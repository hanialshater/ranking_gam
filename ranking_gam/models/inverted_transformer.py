"""
Inverted Transformer baseline for learning-to-rank.

Unlike the standard TransformerRanker (which applies self-attention across
documents in the list), this model applies self-attention across *features*.
Each feature is treated as a token, and the transformer learns which feature
interactions matter for scoring.

This design is inspired by iTransformer (Liu et al., ICLR 2024) which inverts
the attention axis for time series. Applied to ranking:

    Standard Transformer: attention across documents (cross-document interactions)
    Inverted Transformer: attention across features (cross-feature interactions)

Key properties:
    - Scores each document INDEPENDENTLY (no cross-doc interactions), like GAM.
    - Captures arbitrary feature interactions via self-attention, unlike GAM
      (no interactions) or GA2M (pairwise only).
    - Attention weights reveal which features interact, providing some
      interpretability.
    - Useful baseline to isolate the value of feature interactions from
      cross-document interactions.

Architecture (aligned with iTransformer paper):
    Input: [B, L, D] features
    -> Per-feature embedding: each feature gets its own Linear(1, d_model) + LayerNorm
    -> Add learnable feature position embeddings
    -> N pre-norm Transformer encoder layers (self-attention across features)
    -> Pool across features (CLS token or mean)
    -> Linear score head -> [B, L] scores

Changes from original implementation to match the paper:
    1. Per-feature embeddings (not shared) — each feature gets a distinct projection
    2. Pre-norm transformer blocks (norm_first=True) — more stable training
    3. LayerNorm after embedding — acts as instance norm on feature representations
    4. 4x FFN multiplier (dim_feedforward=4*d_model) — paper default
"""

import torch
import torch.nn as nn


class InvertedTransformerRanker(nn.Module):
    """Inverted Transformer ranker: self-attention across features.

    Each document is scored independently. Features are tokens, and the
    transformer learns cross-feature interactions through self-attention.
    This sits between GAM (no feature interactions) and TransformerRanker
    (full cross-document interactions) in terms of model expressiveness.
    """

    def __init__(
        self,
        num_features=136,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.1,
        pooling="cls",
    ):
        """
        Args:
            num_features: number of input features per document.
            d_model: transformer hidden dimension per feature token.
            nhead: number of attention heads.
            num_layers: number of transformer encoder layers.
            dim_feedforward: FFN hidden dimension inside each layer.
                Default: 4 * d_model (paper default).
            dropout: dropout rate.
            pooling: how to aggregate feature tokens into a score.
                "cls" — prepend a learnable [CLS] token, use its output.
                "mean" — average pool over all feature token outputs.
        """
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.pooling = pooling

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Per-feature embedding: each feature gets its own learned projection.
        # In the paper, each variate's full series is projected to d_model;
        # for ranking (scalar features), we use per-feature Linear(1, d_model)
        # so each feature learns a distinct embedding (analogous to GAM towers).
        self.feature_embeds = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

        # LayerNorm after embedding (paper: normalizes variate token
        # representations, acts as instance normalization across features)
        self.embed_norm = nn.LayerNorm(d_model)

        # Learnable position embedding so the model knows which feature
        # is which (features have no inherent ordering).
        n_tokens = num_features + 1 if pooling == "cls" else num_features
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Pre-norm Transformer encoder (paper uses pre-norm: norm before attn/FFN)
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

        # Score head
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "pos_embed" in name or "cls_token" in name:
                continue  # already initialized
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed_features(self, x_flat):
        """Embed each scalar feature with its own projection.

        Args:
            x_flat: [N, D] where N = B*L, D = num_features.

        Returns:
            [N, D, d_model] feature tokens.
        """
        # x_flat[:, i] is the i-th feature for all documents
        tokens = torch.stack([
            self.feature_embeds[i](x_flat[:, i:i+1])
            for i in range(self.num_features)
        ], dim=1)  # [N, D, d_model]
        return self.embed_norm(tokens)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] document features.

        Returns:
            [B, L] relevance scores.
        """
        batch_size, list_size, num_features = x.shape

        # Flatten batch and list dims: [B*L, D]
        x_flat = x.reshape(batch_size * list_size, num_features)

        # Per-feature embedding + LayerNorm: [B*L, D, d_model]
        x_tokens = self._embed_features(x_flat)

        # Prepend CLS token if using CLS pooling
        if self.pooling == "cls":
            n = x_tokens.shape[0]
            cls = self.cls_token.expand(n, -1, -1)  # [B*L, 1, d_model]
            x_tokens = torch.cat([cls, x_tokens], dim=1)  # [B*L, D+1, d_model]

        # Add positional embeddings
        x_tokens = x_tokens + self.pos_embed

        # Self-attention across features (pre-norm encoder)
        h = self.encoder(x_tokens)  # [B*L, D(+1), d_model]

        # Pool to single vector per document
        if self.pooling == "cls":
            doc_repr = h[:, 0]  # [B*L, d_model] — CLS token output
        else:
            doc_repr = h.mean(dim=1)  # [B*L, d_model] — mean pool

        # Score
        scores = self.score_head(doc_repr).squeeze(-1)  # [B*L]
        return scores.reshape(batch_size, list_size)

    def get_feature_attention(self, x):
        """Extract attention weights for interpretability.

        Returns the average attention paid between each pair of features
        across all layers and heads. Useful for understanding which feature
        interactions the model learned.

        Args:
            x: [B, L, D] or [1, 1, D] document features.

        Returns:
            attn_weights: [D, D] average attention matrix across layers/heads.
                If using CLS pooling, returns [D+1, D+1] with row/col 0 = CLS.
        """
        self.eval()
        batch_size, list_size, num_features = x.shape
        x_flat = x.reshape(batch_size * list_size, num_features)
        x_tokens = self._embed_features(x_flat)

        if self.pooling == "cls":
            n = x_tokens.shape[0]
            cls = self.cls_token.expand(n, -1, -1)
            x_tokens = torch.cat([cls, x_tokens], dim=1)

        x_tokens = x_tokens + self.pos_embed

        # Collect attention weights from each layer
        all_attn = []
        h = x_tokens
        with torch.no_grad():
            for layer in self.encoder.layers:
                # Pre-norm: norm1 is applied before self-attention
                h_norm = layer.norm1(h)
                _, attn_w = layer.self_attn(
                    h_norm, h_norm, h_norm, need_weights=True,
                    average_attn_weights=True,
                )
                all_attn.append(attn_w)
                # Run full layer forward for next layer's input
                h = layer(h)

        # Average across layers and batch: [n_tokens, n_tokens]
        attn_stack = torch.stack(all_attn, dim=0)  # [layers, B*L, T, T]
        avg_attn = attn_stack.mean(dim=(0, 1))  # [T, T]
        return avg_attn.cpu()
