"""
Context-present ranking GA2M (Paper Section 4.2 + interactions).

Supports numerical and categorical context features with learned
importance weights over item-level shape functions.

Also provides ContextGAM: a simpler self-context model where all item
features serve as both input and context, producing per-feature importance
weights via a shared context network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .towers import LearnableMonotoneTransform, PaperTower


class ContextWeightNetwork(nn.Module):
    """
    Context feature subnetwork producing importance weights (Eq 8).

    alpha_k = softmax(W_k z_{kT})

    Supports categorical features via embedding (Section 6.2: d=300).
    Output: n-dimensional weight vector (one per item feature).
    """

    def __init__(
        self,
        num_item_features,
        hidden_dims=None,
        dropout=0.0,
        is_categorical=False,
        vocab_size=None,
        embed_dim=300,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.is_categorical = is_categorical

        if is_categorical:
            assert vocab_size is not None
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            in_dim = embed_dim
        else:
            self.embedding = None
            in_dim = 1

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_item_features))
        self.net = nn.Sequential(*layers)

    def forward(self, q_k):
        if self.is_categorical:
            x = self.embedding(q_k.long())
        else:
            x = q_k
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class ContextPresentGA2M(nn.Module):
    """
    Context-present ranking GA2M (Paper Section 4.2 + pairwise interactions).

    y_hat_i = sum_j w_j(q) * f_j(x_{ij}) + sum_{(j,k)} f_{jk}(x_j, x_k) + bias

    where w_j(q) = sum_k alpha_k^(j)  (Eq 9)
    and   alpha_k = softmax(net_k(q_k))  (Eq 8)
    """

    def __init__(
        self,
        num_item_features,
        context_feature_specs,
        interaction_pairs=None,
        item_hidden=None,
        interaction_hidden=None,
        context_hidden=None,
        dropout=0.0,
    ):
        """
        Args:
            num_item_features: n
            context_feature_specs: list of dicts, each with:
                'type': 'numerical' or 'categorical'
                'vocab_size': int (categorical only)
                'embed_dim': int (default 300)
            interaction_pairs: [(j, k), ...]
            item_hidden: CWS uses [64, 32] for item towers
            context_hidden: [128, 64] per paper Section 6.2
        """
        super().__init__()
        if item_hidden is None:
            item_hidden = [64, 32]
        if interaction_hidden is None:
            interaction_hidden = [16, 8]
        if context_hidden is None:
            context_hidden = [128, 64]

        self.num_item_features = num_item_features
        self.num_features = num_item_features  # alias for distill compatibility
        self.context_feature_specs = context_feature_specs
        self.num_context_features = len(context_feature_specs)
        self.interaction_pairs = interaction_pairs or []
        self.num_interactions = len(self.interaction_pairs)

        self.main_towers = nn.ModuleList(
            [PaperTower(1, item_hidden, dropout) for _ in range(num_item_features)]
        )

        self.context_towers = nn.ModuleList(
            [
                ContextWeightNetwork(
                    num_item_features=num_item_features,
                    hidden_dims=context_hidden,
                    dropout=dropout,
                    is_categorical=(spec["type"] == "categorical"),
                    vocab_size=spec.get("vocab_size"),
                    embed_dim=spec.get("embed_dim", 300),
                )
                for spec in context_feature_specs
            ]
        )

        self.interaction_towers = nn.ModuleList(
            [
                PaperTower(2, interaction_hidden, dropout)
                for _ in range(self.num_interactions)
            ]
        )

        self.global_bias = nn.Parameter(torch.zeros(1))

    def _compute_context_weights(self, q):
        """alpha = sum_k alpha_k  (Eq 9). q: list of tensors, each [batch]."""
        alpha = None
        for k, tower in enumerate(self.context_towers):
            q_k = q[k]
            if self.context_feature_specs[k]["type"] != "categorical":
                if q_k.dim() == 1:
                    q_k = q_k.unsqueeze(-1)
            alpha_k = tower(q_k)
            alpha = alpha_k if alpha is None else alpha + alpha_k
        return alpha

    def forward(self, x, q=None):
        """
        Args:
            x: [batch, list_size, num_item_features]
            q: list of context tensors, each [batch], or None for context-absent
        """
        B, L, n = x.shape
        x_flat = x.reshape(B * L, -1)

        sub_scores = []
        for j, tower in enumerate(self.main_towers):
            sub_scores.append(tower(x_flat[:, j : j + 1]))
        sub_scores = torch.cat(sub_scores, dim=-1)

        if q is not None and self.num_context_features > 0:
            alpha = self._compute_context_weights(q)
            alpha_exp = alpha.unsqueeze(1).expand(B, L, n).reshape(B * L, n)
            main_scores = (sub_scores * alpha_exp).sum(dim=-1, keepdim=True)
        else:
            main_scores = sub_scores.sum(dim=-1, keepdim=True)

        int_scores = torch.zeros(B * L, 1, device=x.device)
        for i, tower in enumerate(self.interaction_towers):
            f1, f2 = self.interaction_pairs[i]
            pair = torch.stack([x_flat[:, f1], x_flat[:, f2]], dim=-1)
            int_scores += tower(pair)

        return (main_scores + int_scores).reshape(B, L) + self.global_bias

    def get_main_effect(self, feature_idx, x_values):
        """Evaluate f_j(x) independent of context."""
        self.eval()
        x_t = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        x_t = x_t.to(next(self.parameters()).device)
        with torch.no_grad():
            return self.main_towers[feature_idx](x_t).cpu().numpy().flatten()

    def get_interaction_effect(self, pair_idx, x1_values, x2_values):
        self.eval()
        x1 = torch.tensor(x1_values, dtype=torch.float32).flatten()
        x2 = torch.tensor(x2_values, dtype=torch.float32).flatten()
        pair = torch.stack([x1, x2], dim=-1).to(next(self.parameters()).device)
        with torch.no_grad():
            return self.interaction_towers[pair_idx](pair).cpu().numpy().flatten()

    def get_context_weight_for_feature(self, context_idx, feature_idx, q_values):
        """Evaluate alpha_k^(j) for a single (context, item) pair."""
        self.eval()
        device = next(self.parameters()).device
        spec = self.context_feature_specs[context_idx]
        if spec["type"] == "categorical":
            q_t = torch.tensor(q_values, dtype=torch.long).to(device)
        else:
            q_t = torch.tensor(q_values, dtype=torch.float32).unsqueeze(-1).to(device)
        with torch.no_grad():
            alpha_k = self.context_towers[context_idx](q_t)
            return alpha_k[:, feature_idx].cpu().numpy()


class ContextGAM(nn.Module):
    """
    Self-context GAM: per-feature towers + shared context network.

    score = sum_j w_j(x) * f_j(x_j) + bias

    where w = softmax(context_net(x)) produces per-feature importance weights
    from the full feature vector, and f_j is a per-feature MLP tower.

    Same forward(x) signature as GAM_Paper -- works with existing training loop.

    Interpretability:
      - Tower shapes f_j(x_j) are single-variable functions (distillable to PWL)
      - Context weights w_j(x) are query-dependent but inspectable per-query
      - The context network is a small MLP (~14K params for 136 features)

    Distillation:
      - Towers f_j -> PWL via greedy knot selection (same as GAM_Paper)
      - Context network w_j(x) -> kept as small neural net (136-dim input,
        can't reduce to 1D lookup, but cheap: one matrix multiply at serving)
    """

    def __init__(
        self,
        num_features=136,
        hidden_dims=None,
        context_hidden=None,
        dropout=0.0,
        residual=False,
        input_norm=False,
        feature_transforms=False,
        num_transform_knots=20,
        tower_dropout=0.0,
        output_norm=False,
        activation="relu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]
        if context_hidden is None:
            context_hidden = [64, 32]

        self.num_features = num_features
        self.tower_dropout_rate = tower_dropout
        self.output_norm = output_norm

        # Per-feature towers (same as GAM_Paper)
        self.towers = nn.ModuleList(
            [PaperTower(1, hidden_dims, dropout, residual, input_norm, activation=activation)
             for _ in range(num_features)]
        )

        # Feature transforms (optional)
        self.feature_transforms = None
        if feature_transforms:
            self.feature_transforms = nn.ModuleList(
                [LearnableMonotoneTransform(num_knots=num_transform_knots)
                 for _ in range(num_features)]
            )

        # Shared context network: all features -> per-feature weights
        layers = []
        prev = num_features
        for h in context_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_features))
        self.context_net = nn.Sequential(*layers)

        if output_norm:
            self.tower_norm = nn.BatchNorm1d(num_features, affine=False)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def init_transforms_from_data(self, X):
        """Initialize feature transforms from training data percentiles."""
        if self.feature_transforms is None:
            return
        for j, transform in enumerate(self.feature_transforms):
            transform.init_from_data(X[:, :, j])

    def forward(self, x):
        """x: [batch, list_size, num_features] -> [batch, list_size]."""
        B, L, n = x.shape
        x_flat = x.view(B * L, n)

        # Per-feature tower outputs
        tower_outputs = []
        for i, tower in enumerate(self.towers):
            feat = x_flat[:, i : i + 1]
            if self.feature_transforms is not None:
                feat = self.feature_transforms[i](feat)
            tower_outputs.append(tower(feat))
        tower_out = torch.cat(tower_outputs, dim=-1)  # [B*L, n]

        # Tower dropout
        if self.tower_dropout_rate > 0 and self.training:
            mask = torch.bernoulli(
                torch.full((1, n), 1 - self.tower_dropout_rate, device=tower_out.device)
            )
            tower_out = tower_out * mask / (1 - self.tower_dropout_rate)

        # Output normalization
        if self.output_norm:
            tower_out = self.tower_norm(tower_out)

        # Context weights from all features (softmax -> sums to 1)
        # Scale by num_features so total magnitude is comparable to pure sum
        weights = F.softmax(self.context_net(x_flat), dim=-1)  # [B*L, n]
        scores = n * (weights * tower_out).sum(dim=-1)

        return scores.view(B, L) + self.global_bias

    def get_main_effect(self, feature_idx, x_values):
        """Tower shape without context weighting (for distillation/viz)."""
        self.eval()
        x_t = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        x_t = x_t.to(next(self.parameters()).device)
        with torch.no_grad():
            feat = x_t
            if self.feature_transforms is not None:
                feat = self.feature_transforms[feature_idx](feat)
            return self.towers[feature_idx](feat).cpu().numpy().flatten()

    def get_context_weights(self, x_single):
        """Get context weights for a single document.

        Args:
            x_single: [num_features] numpy array (one document's features)
        Returns:
            [num_features] numpy array of weights (sum to 1)
        """
        self.eval()
        x_t = torch.tensor(x_single, dtype=torch.float32).reshape(1, -1)
        x_t = x_t.to(next(self.parameters()).device)
        with torch.no_grad():
            w = F.softmax(self.context_net(x_t), dim=-1)
            return w.cpu().numpy().flatten()

    def explain(self, x_single):
        """Per-feature attribution for a single document.

        Returns dict with tower outputs, context weights, and weighted contributions.
        """
        self.eval()
        x_t = torch.tensor(x_single, dtype=torch.float32).reshape(1, -1)
        x_t = x_t.to(next(self.parameters()).device)
        n = self.num_features
        with torch.no_grad():
            tower_vals = []
            for i, tower in enumerate(self.towers):
                feat = x_t[:, i : i + 1]
                if self.feature_transforms is not None:
                    feat = self.feature_transforms[i](feat)
                tower_vals.append(tower(feat).item())
            tower_vals = np.array(tower_vals)
            weights = F.softmax(self.context_net(x_t), dim=-1).cpu().numpy().flatten()
            contributions = n * weights * tower_vals
        return {
            "tower_outputs": tower_vals,
            "context_weights": weights,
            "contributions": contributions,
            "score": float(contributions.sum() + self.global_bias.item()),
        }
