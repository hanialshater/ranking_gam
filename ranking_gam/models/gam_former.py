"""
GAMFormer: Hybrid GAM + Inverted Transformer for learning-to-rank.

Combines interpretable GAM main effects with transformer-learned feature
interactions in an additive decomposition:

    score = Σ f_j(x_j) + g(x_1, ..., x_D) + bias
            ↑ GAM towers   ↑ iTransformer residual
            (interpretable)  (captures interactions)

The GAM towers give per-feature shape functions that are:
    - Visualizable via get_main_effect()
    - Distillable to piecewise-linear (PWL) for fast serving
    - Individually interpretable (each f_j is univariate)

The inverted transformer adds a learned interaction residual:
    - Self-attention across features captures higher-order interactions
    - Attention weights reveal which features interact (get_feature_attention)
    - Equivalent to GA2M's pairwise interactions but with arbitrary order

This sits between GAM (fully interpretable, no interactions) and pure
InvertedTransformerRanker (all transformer, no explicit main effects)
on the interpretability spectrum.

Architecture:
    Input: [B, L, D] features
    ┌─ GAM branch: per-feature MLP towers → Σ f_j(x_j)
    │  (optional feature transforms, residual connections)
    ├─ Transformer branch: per-feature embed → self-attention → pool → MLP
    │  (captures what GAM misses: interactions between features)
    └─ Output: gam_score + interaction_score + bias → [B, L]
"""

import torch
import torch.nn as nn

from .towers import LearnableMonotoneTransform, PaperTower


class GAMFormer(nn.Module):
    """Hybrid GAM + Inverted Transformer ranker.

    GAM main effects provide interpretable per-feature shape functions.
    The inverted transformer captures feature interactions as a residual.
    Scores each document independently (no cross-document attention).
    """

    def __init__(
        self,
        num_features=136,
        # GAM tower config
        hidden_dims=None,
        tower_dropout=0.0,
        residual=False,
        feature_transforms=False,
        num_transform_knots=20,
        activation="relu",
        # Transformer config
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.1,
        pooling="cls",
        # Combination
        interaction_weight=1.0,
    ):
        """
        Args:
            num_features: number of input features per document.
            hidden_dims: GAM tower hidden dimensions (default: [16, 8]).
            tower_dropout: dropout rate for GAM tower outputs.
            residual: add skip connection in GAM towers.
            feature_transforms: use learnable monotone input transforms.
            num_transform_knots: knots for monotone transforms.
            activation: GAM tower activation ("relu", "silu", "gelu").
            d_model: transformer hidden dimension per feature token.
            nhead: number of attention heads.
            num_layers: number of transformer encoder layers.
            dim_feedforward: FFN hidden dim (default: 4 * d_model).
            dropout: transformer dropout rate.
            pooling: "cls" or "mean" for transformer aggregation.
            interaction_weight: initial weight for interaction term.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.num_features = num_features
        self.d_model = d_model
        self.pooling = pooling
        self.tower_dropout_rate = tower_dropout

        # ── GAM branch: per-feature MLP towers ──
        self.towers = nn.ModuleList([
            PaperTower(1, hidden_dims, dropout=0.0, residual=residual,
                       activation=activation)
            for _ in range(num_features)
        ])

        self.feature_transforms = None
        if feature_transforms:
            self.feature_transforms = nn.ModuleList([
                LearnableMonotoneTransform(num_knots=num_transform_knots)
                for _ in range(num_features)
            ])

        self.global_bias = nn.Parameter(torch.zeros(1))

        # ── Transformer branch: feature-level self-attention ──
        self.feature_embeds = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.embed_norm = nn.LayerNorm(d_model)

        n_tokens = num_features + 1 if pooling == "cls" else num_features
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.interaction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Learnable weight for the interaction term
        self.interaction_weight = nn.Parameter(
            torch.tensor(float(interaction_weight))
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "pos_embed" in name or "cls_token" in name or "interaction_weight" in name:
                continue
            # Only xavier-init transformer weights, not GAM towers
            # (GAM towers use PyTorch default init which works well)
            if any(prefix in name for prefix in
                   ["feature_embeds", "encoder", "interaction_head"]):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def init_transforms_from_data(self, X):
        """Initialize feature transforms from training data percentiles.

        Args:
            X: [B, L, D] numpy array of training features.
        """
        if self.feature_transforms is None:
            return
        for j, transform in enumerate(self.feature_transforms):
            transform.init_from_data(X[:, :, j])

    def _embed_features(self, x_flat):
        """Embed each scalar feature with its own projection.

        Args:
            x_flat: [N, D] where N = B*L.

        Returns:
            [N, D, d_model] feature tokens.
        """
        tokens = torch.stack([
            self.feature_embeds[i](x_flat[:, i:i+1])
            for i in range(self.num_features)
        ], dim=1)
        return self.embed_norm(tokens)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] document features.

        Returns:
            [B, L] relevance scores.
        """
        batch_size, list_size, num_features = x.shape
        x_flat = x.reshape(batch_size * list_size, num_features)

        # ── GAM branch ──
        tower_outputs = []
        for i, tower in enumerate(self.towers):
            feat = x_flat[:, i:i+1]
            if self.feature_transforms is not None:
                feat = self.feature_transforms[i](feat)
            tower_outputs.append(tower(feat))

        tower_out = torch.cat(tower_outputs, dim=-1)  # [B*L, num_features]

        if self.tower_dropout_rate > 0 and self.training:
            mask = torch.bernoulli(
                torch.full((1, num_features), 1 - self.tower_dropout_rate,
                           device=tower_out.device)
            )
            tower_out = tower_out * mask / (1 - self.tower_dropout_rate)

        gam_score = tower_out.sum(dim=-1)  # [B*L]

        # ── Transformer branch ──
        x_tokens = self._embed_features(x_flat)  # [B*L, D, d_model]

        if self.pooling == "cls":
            n = x_tokens.shape[0]
            cls = self.cls_token.expand(n, -1, -1)
            x_tokens = torch.cat([cls, x_tokens], dim=1)

        x_tokens = x_tokens + self.pos_embed
        h = self.encoder(x_tokens)

        if self.pooling == "cls":
            doc_repr = h[:, 0]
        else:
            doc_repr = h.mean(dim=1)

        interaction_score = self.interaction_head(doc_repr).squeeze(-1)  # [B*L]

        # ── Combine ──
        scores = gam_score + self.interaction_weight * interaction_score
        return scores.reshape(batch_size, list_size) + self.global_bias

    def get_main_effect(self, feature_idx, x_values):
        """Evaluate f_j(x) for visualization (GAM tower only).

        This shows the interpretable main effect — the transformer
        interaction residual is NOT included here.

        Args:
            feature_idx: which feature tower to evaluate.
            x_values: 1D array of feature values to evaluate at.

        Returns:
            numpy array of tower outputs.
        """
        self.eval()
        x_values = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        x_values = x_values.to(next(self.parameters()).device)
        with torch.no_grad():
            feat = x_values
            if self.feature_transforms is not None:
                feat = self.feature_transforms[feature_idx](feat)
            return self.towers[feature_idx](feat).cpu().numpy().flatten()

    def explain(self, x_single):
        """Decompose a single document's score into GAM + interaction.

        Args:
            x_single: [D] or [1, 1, D] feature vector for one document.

        Returns:
            dict with:
                tower_outputs: [D] per-feature GAM contributions
                interaction_score: scalar interaction residual
                interaction_weight: current weight for interaction term
                gam_score: sum of tower outputs + bias
                total_score: final combined score
        """
        self.eval()
        if x_single.dim() == 1:
            x_single = x_single.unsqueeze(0).unsqueeze(0)
        elif x_single.dim() == 2:
            x_single = x_single.unsqueeze(0)

        device = next(self.parameters()).device
        x_single = x_single.to(device)
        x_flat = x_single.reshape(1, -1)

        with torch.no_grad():
            # GAM contributions
            tower_vals = []
            for i, tower in enumerate(self.towers):
                feat = x_flat[:, i:i+1]
                if self.feature_transforms is not None:
                    feat = self.feature_transforms[i](feat)
                tower_vals.append(tower(feat).item())

            tower_outputs = torch.tensor(tower_vals)
            gam_score = tower_outputs.sum().item() + self.global_bias.item()

            # Transformer interaction
            x_tokens = self._embed_features(x_flat)
            if self.pooling == "cls":
                cls = self.cls_token.expand(1, -1, -1)
                x_tokens = torch.cat([cls, x_tokens], dim=1)
            x_tokens = x_tokens + self.pos_embed
            h = self.encoder(x_tokens)
            if self.pooling == "cls":
                doc_repr = h[:, 0]
            else:
                doc_repr = h.mean(dim=1)
            interaction = self.interaction_head(doc_repr).item()
            weight = self.interaction_weight.item()

        return {
            "tower_outputs": tower_outputs,
            "interaction_score": interaction,
            "interaction_weight": weight,
            "gam_score": gam_score,
            "total_score": gam_score + weight * interaction,
        }

    def get_feature_attention(self, x):
        """Extract attention weights from the transformer branch.

        Args:
            x: [B, L, D] or [1, 1, D] document features.

        Returns:
            attn_weights: [T, T] average attention matrix (T = D or D+1 with CLS).
        """
        self.eval()
        batch_size, list_size, num_features = x.shape
        x_flat = x.reshape(batch_size * list_size, num_features)
        x_flat = x_flat.to(next(self.parameters()).device)
        x_tokens = self._embed_features(x_flat)

        if self.pooling == "cls":
            n = x_tokens.shape[0]
            cls = self.cls_token.expand(n, -1, -1)
            x_tokens = torch.cat([cls, x_tokens], dim=1)

        x_tokens = x_tokens + self.pos_embed

        all_attn = []
        h = x_tokens
        with torch.no_grad():
            for layer in self.encoder.layers:
                h_norm = layer.norm1(h)
                _, attn_w = layer.self_attn(
                    h_norm, h_norm, h_norm, need_weights=True,
                    average_attn_weights=True,
                )
                all_attn.append(attn_w)
                h = layer(h)

        attn_stack = torch.stack(all_attn, dim=0)
        avg_attn = attn_stack.mean(dim=(0, 1))
        return avg_attn.cpu()

    def gam_param_count(self):
        """Count parameters in GAM branch only."""
        count = sum(p.numel() for p in self.towers.parameters())
        if self.feature_transforms is not None:
            count += sum(p.numel() for p in self.feature_transforms.parameters())
        count += self.global_bias.numel()
        return count

    def transformer_param_count(self):
        """Count parameters in transformer branch only."""
        count = sum(p.numel() for p in self.feature_embeds.parameters())
        count += sum(p.numel() for p in self.encoder.parameters())
        count += sum(p.numel() for p in self.interaction_head.parameters())
        count += self.embed_norm.weight.numel() + self.embed_norm.bias.numel()
        count += self.pos_embed.numel()
        count += self.interaction_weight.numel()
        if self.pooling == "cls":
            count += self.cls_token.numel()
        return count
