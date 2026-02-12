"""
GAM and GA2M ranking models (context-absent).

GAM_Paper: one tower per feature, no interactions.
GA2M_Paper: main effects + pairwise interaction towers.

Matches Zhuang et al. WSDM 2021 Section 4.1.
"""

import numpy as np
import torch
import torch.nn as nn

from .towers import PaperTower, LearnableMonotoneTransform


class GA2M_Paper(nn.Module):
    """
    Context-absent GA2M (paper Section 4.1 + pairwise interactions).

    score = sum f_j(x_j) + sum f_{jk}(x_j, x_k) + bias

    Paper Section 6.2: [16, 8] for WEB30K/YAHOO item towers.
    """

    def __init__(
        self,
        num_features=136,
        interaction_pairs=None,
        hidden_dims=None,
        interaction_hidden=None,
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
        if interaction_hidden is None:
            interaction_hidden = [16, 8]

        self.num_features = num_features
        self.interaction_pairs = interaction_pairs or []
        self.num_interactions = len(self.interaction_pairs)
        self.tower_dropout_rate = tower_dropout
        self.output_norm = output_norm

        self.main_towers = nn.ModuleList(
            [PaperTower(1, hidden_dims, dropout, residual, input_norm, activation=activation)
             for _ in range(num_features)]
        )

        self.interaction_towers = nn.ModuleList(
            [
                PaperTower(2, interaction_hidden, dropout, residual, input_norm, activation=activation)
                for _ in range(self.num_interactions)
            ]
        )

        self.feature_transforms = None
        if feature_transforms:
            self.feature_transforms = nn.ModuleList(
                [LearnableMonotoneTransform(num_knots=num_transform_knots) for _ in range(num_features)]
            )

        n_total_towers = num_features + self.num_interactions
        if output_norm:
            self.tower_norm = nn.BatchNorm1d(n_total_towers, affine=False)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def init_transforms_from_data(self, X):
        """Initialize feature transforms from training data percentiles.

        Args:
            X: [B, L, D] numpy array of training features.
        """
        if self.feature_transforms is None:
            return
        for j, transform in enumerate(self.feature_transforms):
            transform.init_from_data(X[:, :, j])

    def forward(self, x):
        """x: [batch, list_size, num_features] -> [batch, list_size]."""
        batch_size, list_size, _ = x.shape
        x_flat = x.view(batch_size * list_size, -1)

        all_tower_outputs = []

        for i, tower in enumerate(self.main_towers):
            feat = x_flat[:, i : i + 1]
            if self.feature_transforms is not None:
                feat = self.feature_transforms[i](feat)
            all_tower_outputs.append(tower(feat))

        if self.num_interactions > 0:
            for i, tower in enumerate(self.interaction_towers):
                f1, f2 = self.interaction_pairs[i]
                feat1 = x_flat[:, f1]
                feat2 = x_flat[:, f2]
                if self.feature_transforms is not None:
                    feat1 = self.feature_transforms[f1](feat1)
                    feat2 = self.feature_transforms[f2](feat2)
                feat_pair = torch.stack([feat1, feat2], dim=-1)
                all_tower_outputs.append(tower(feat_pair))

        # [B*L, N_towers]
        tower_out = torch.cat(all_tower_outputs, dim=-1)

        if self.tower_dropout_rate > 0 and self.training:
            n_towers = tower_out.shape[-1]
            mask = torch.bernoulli(
                torch.full((1, n_towers), 1 - self.tower_dropout_rate, device=tower_out.device)
            )
            tower_out = tower_out * mask / (1 - self.tower_dropout_rate)

        if self.output_norm:
            tower_out = self.tower_norm(tower_out)

        total = tower_out.sum(dim=-1)
        scores = total.view(batch_size, list_size) + self.global_bias
        return scores

    def get_main_effect(self, feature_idx, x_values):
        """Evaluate f_j(x) for visualization (applies transform if present)."""
        self.eval()
        x_values = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        x_values = x_values.to(next(self.parameters()).device)
        with torch.no_grad():
            feat = x_values
            if self.feature_transforms is not None:
                feat = self.feature_transforms[feature_idx](feat)
            return self.main_towers[feature_idx](feat).cpu().numpy().flatten()

    def get_interaction_effect(self, pair_idx, x1_values, x2_values):
        """Evaluate f_{jk}(x_j, x_k) for visualization."""
        self.eval()
        x1 = torch.tensor(x1_values, dtype=torch.float32).flatten()
        x2 = torch.tensor(x2_values, dtype=torch.float32).flatten()
        dev = next(self.parameters()).device
        with torch.no_grad():
            if self.feature_transforms is not None:
                f1, f2 = self.interaction_pairs[pair_idx]
                x1 = self.feature_transforms[f1](x1.to(dev))
                x2 = self.feature_transforms[f2](x2.to(dev))
            x_pair = torch.stack([x1, x2], dim=-1).to(dev)
            return self.interaction_towers[pair_idx](x_pair).cpu().numpy().flatten()


class GAM_Paper(nn.Module):
    """
    GAM baseline with paper architecture (no interactions).

    score = sum f_j(x_j) + bias

    Optional enhancements (all off by default for backward compat):
        tower_dropout: randomly zero out tower outputs during training
        output_norm: center tower outputs to zero mean (prevents one tower dominating)
    """

    def __init__(
        self,
        num_features=136,
        hidden_dims=None,
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
        self.num_features = num_features
        self.tower_dropout_rate = tower_dropout
        self.output_norm = output_norm

        self.towers = nn.ModuleList(
            [PaperTower(1, hidden_dims, dropout, residual, input_norm, activation=activation)
             for _ in range(num_features)]
        )

        self.feature_transforms = None
        if feature_transforms:
            self.feature_transforms = nn.ModuleList(
                [LearnableMonotoneTransform(num_knots=num_transform_knots) for _ in range(num_features)]
            )

        if output_norm:
            self.tower_norm = nn.BatchNorm1d(num_features, affine=False)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def init_transforms_from_data(self, X):
        """Initialize feature transforms from training data percentiles.

        Args:
            X: [B, L, D] numpy array of training features.
        """
        if self.feature_transforms is None:
            return
        for j, transform in enumerate(self.feature_transforms):
            transform.init_from_data(X[:, :, j])

    def forward(self, x):
        batch_size, list_size, _ = x.shape
        x_flat = x.view(batch_size * list_size, -1)

        scores = []
        for i, tower in enumerate(self.towers):
            feat = x_flat[:, i : i + 1]
            if self.feature_transforms is not None:
                feat = self.feature_transforms[i](feat)
            scores.append(tower(feat))

        # Stack tower outputs: [num_features, B*L, 1] -> [B*L, num_features]
        tower_out = torch.cat(scores, dim=-1)

        # Tower dropout: randomly zero out entire tower outputs during training
        if self.tower_dropout_rate > 0 and self.training:
            mask = torch.bernoulli(
                torch.full((1, self.num_features), 1 - self.tower_dropout_rate, device=tower_out.device)
            )
            tower_out = tower_out * mask / (1 - self.tower_dropout_rate)

        # Output normalization: center tower outputs to prevent dominance
        if self.output_norm:
            tower_out = self.tower_norm(tower_out)

        total = tower_out.sum(dim=-1)
        return total.view(batch_size, list_size) + self.global_bias

    def get_main_effect(self, feature_idx, x_values):
        """Evaluate f_j(x) for visualization (applies transform if present)."""
        self.eval()
        x_values = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        x_values = x_values.to(next(self.parameters()).device)
        with torch.no_grad():
            feat = x_values
            if self.feature_transforms is not None:
                feat = self.feature_transforms[feature_idx](feat)
            return self.towers[feature_idx](feat).cpu().numpy().flatten()
