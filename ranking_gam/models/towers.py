"""
Base tower architectures: MLP, monotone PWL, concave PWL, and learnable transforms.

These are the building blocks that all GAM models compose.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperTower(nn.Module):
    """
    Per-feature MLP tower (Section 4.1 of Zhuang et al. WSDM 2021).

    Architecture: Dense -> ReLU -> ... -> Dense(1)
        z_{j1} = sigma(W_{j1} x_j + b_{j1}), etc.
        f_j(x_j) = W_j z_{jH} + b_j  (no activation on output, Eq 6)

    Default hidden dims per dataset (Section 6.2):
        YAHOO / WEB30K item features: [16, 8]
        CWS item features: [64, 32]
        CWS context features: [128, 64]

    Optional enhancements:
        residual: add a linear skip connection from input to output
        input_norm: apply BatchNorm to the input before the MLP
    """

    def __init__(self, in_dim, hidden_dims=None, dropout=0.0, residual=False, input_norm=False, activation="relu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]

        self.norm = nn.BatchNorm1d(in_dim) if input_norm else None

        act_map = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
        act_cls = act_map.get(activation, nn.ReLU)

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        self.skip = nn.Linear(in_dim, 1, bias=False) if residual else None

    def forward(self, x):
        """x: [batch * list_size, in_dim] -> [batch * list_size, 1]."""
        h = self.norm(x) if self.norm is not None else x
        out = self.net(h)
        if self.skip is not None:
            out = out + self.skip(x)
        return out


class LearnableMonotoneTransform(nn.Module):
    """
    Learnable monotone feature transformation, initialized from data percentiles.

    Maps raw feature values through a monotone piecewise-linear function
    to [0, 1]. Initialized to approximate the empirical CDF so that
    features start in a well-normalized range. The monotone warp is learned
    end-to-end, preserving interpretability (monotone of f is still single-variable).

    Parameterization:
        - x_knots: fixed at data percentiles (buffer, not learned)
        - y_knots: cumulative softplus deltas, normalized to [0, 1]
          y_i = cumsum(softplus(raw_deltas))_i / total

    Usage:
        transform = LearnableMonotoneTransform(num_knots=20)
        transform.init_from_data(feature_values)  # set knot positions
        normalized = transform(raw_feature)         # [0, 1] output
    """

    def __init__(self, num_knots=20):
        super().__init__()
        self.num_knots = num_knots
        self.register_buffer("x_knots", torch.linspace(0, 1, num_knots))
        self.raw_deltas = nn.Parameter(torch.zeros(num_knots - 1))
        self._init_uniform_deltas()

    def _init_uniform_deltas(self):
        """Initialize deltas so output approximates a uniform CDF."""
        target = 1.0 / (self.num_knots - 1)
        # inverse softplus: x = log(exp(y) - 1)
        init_val = float(np.log(np.exp(target) - 1))
        self.raw_deltas.data.fill_(init_val)

    def init_from_data(self, feature_values):
        """Set knot x-positions at data percentiles.

        Args:
            feature_values: numpy array or tensor of raw feature values (any shape).
        """
        if isinstance(feature_values, torch.Tensor):
            feature_values = feature_values.detach().cpu().numpy()
        vals = feature_values.flatten().astype(np.float64)
        percentiles = np.linspace(0, 100, self.num_knots)
        x_knots = np.percentile(vals, percentiles).astype(np.float32)
        # Ensure strictly increasing for well-defined interpolation
        for i in range(1, len(x_knots)):
            if x_knots[i] <= x_knots[i - 1]:
                x_knots[i] = x_knots[i - 1] + 1e-6
        self.x_knots.copy_(torch.from_numpy(x_knots))
        self._init_uniform_deltas()

    def get_y_knots(self):
        """Return monotone y-knot values in [0, 1]."""
        deltas = F.softplus(self.raw_deltas)
        cum = torch.cat([torch.zeros(1, device=deltas.device), torch.cumsum(deltas, 0)])
        total = cum[-1]
        return cum / (total + 1e-10)

    def forward(self, x):
        """x: any shape tensor -> same shape, values in [0, 1]."""
        y_knots = self.get_y_knots()
        x_knots = self.x_knots

        shape = x.shape
        x_flat = x.reshape(-1)
        x_clamped = x_flat.clamp(x_knots[0], x_knots[-1])

        idx = torch.searchsorted(x_knots, x_clamped).clamp(1, self.num_knots - 1)

        x_lo = x_knots[idx - 1]
        x_hi = x_knots[idx]
        y_lo = y_knots[idx - 1]
        y_hi = y_knots[idx]

        t = (x_clamped - x_lo) / (x_hi - x_lo + 1e-10)
        result = y_lo + t * (y_hi - y_lo)

        return result.reshape(shape)

    def get_curve(self, n_points=200):
        """Return (x, y) numpy arrays for plotting."""
        x = torch.linspace(
            self.x_knots[0].item(), self.x_knots[-1].item(), n_points
        )
        with torch.no_grad():
            y = self.forward(x.to(self.x_knots.device))
        return x.numpy(), y.cpu().numpy()


class ConcavePWL(nn.Module):
    """
    Piecewise-linear function with enforced monotonicity AND concavity.

    Guarantees:  slope_0 >= slope_1 >= ... >= slope_{K-1} >= 0

    This means:
        - Monotone non-decreasing (all slopes >= 0)
        - Concave (slopes are non-increasing -> diminishing returns)

    Useful for diversity features where more novelty always helps but
    marginal benefit shrinks as you add more diverse items.

    Parameterization (build from smallest slope up):
        slope_{K-1} = softplus(raw_{K-1})               >= 0
        slope_{i}   = slope_{i+1} + softplus(raw_{i})    (each earlier slope >= next)

    Submodularity: concave of modular = submodular, so greedy gives (1-1/e).
    """

    def __init__(self, num_knots=10, x_min=0.0, x_max=1.0):
        super().__init__()
        self.num_knots = num_knots
        self.x_min = x_min
        self.x_max = x_max

        init_vals = torch.zeros(num_knots)
        init_vals[-1] = 0.0
        init_vals[:-1] = -1.0
        self.raw_params = nn.Parameter(init_vals)

        self.intercept = nn.Parameter(torch.tensor(0.0))

        knots = torch.linspace(x_min, x_max, num_knots + 1)
        self.register_buffer("knot_edges", knots)
        self.register_buffer("knot_widths", knots[1:] - knots[:-1])

    def get_slopes(self):
        """Return monotone-concave slopes: s_0 >= s_1 >= ... >= s_{K-1} >= 0."""
        sp = F.softplus(self.raw_params)
        slopes = torch.zeros_like(sp)
        slopes[-1] = sp[-1]
        for i in range(self.num_knots - 2, -1, -1):
            slopes[i] = slopes[i + 1] + sp[i]
        return slopes

    def forward(self, x):
        slopes = self.get_slopes()
        x_clamped = x.clamp(self.x_min, self.x_max)
        result = self.intercept.clone().expand_as(x_clamped)
        for i in range(self.num_knots):
            left = self.knot_edges[i]
            width = self.knot_widths[i]
            segment = (x_clamped - left).clamp(0, width)
            result = result + slopes[i] * segment
        return result

    def get_curve(self, n_points=100):
        """Return (x, y) numpy arrays for plotting."""
        x = torch.linspace(self.x_min, self.x_max, n_points)
        with torch.no_grad():
            y = self.forward(x.to(self.raw_params.device))
        return x.numpy(), y.cpu().numpy()


class MonotonePWL(nn.Module):
    """
    Piecewise-linear function with enforced monotonicity (all slopes >= 0).

    Unlike ConcavePWL, slopes are independent -- no concavity constraint.
    Use for pointwise objectives like revenue or freshness where monotonicity
    is the right inductive bias but diminishing returns are not guaranteed.

    Parameterization: slope_i = softplus(raw_i)  >= 0
    """

    def __init__(self, num_knots=10, x_min=0.0, x_max=1.0):
        super().__init__()
        self.num_knots = num_knots
        self.x_min = x_min
        self.x_max = x_max

        self.raw_params = nn.Parameter(torch.zeros(num_knots))
        self.intercept = nn.Parameter(torch.tensor(0.0))

        knots = torch.linspace(x_min, x_max, num_knots + 1)
        self.register_buffer("knot_edges", knots)
        self.register_buffer("knot_widths", knots[1:] - knots[:-1])

    def get_slopes(self):
        return F.softplus(self.raw_params)

    def forward(self, x):
        slopes = self.get_slopes()
        x_clamped = x.clamp(self.x_min, self.x_max)
        result = self.intercept.clone().expand_as(x_clamped)
        for i in range(self.num_knots):
            left = self.knot_edges[i]
            width = self.knot_widths[i]
            segment = (x_clamped - left).clamp(0, width)
            result = result + slopes[i] * segment
        return result

    def get_curve(self, n_points=100):
        x = torch.linspace(self.x_min, self.x_max, n_points)
        with torch.no_grad():
            y = self.forward(x.to(self.raw_params.device))
        return x.numpy(), y.cpu().numpy()
