"""
Base tower architectures: MLP, monotone PWL, and concave PWL.

These are the building blocks that all GAM models compose.
"""

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
    """

    def __init__(self, in_dim, hidden_dims=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: [batch * list_size, in_dim] -> [batch * list_size, 1]."""
        return self.net(x)


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
