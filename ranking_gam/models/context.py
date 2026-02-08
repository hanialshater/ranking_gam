"""
Context-present ranking GA2M (Paper Section 4.2 + interactions).

Supports numerical and categorical context features with learned
importance weights over item-level shape functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .towers import PaperTower


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
