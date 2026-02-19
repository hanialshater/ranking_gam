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

Architecture:
    Input: [B, L, D] features
    -> Per-feature linear embedding: scalar -> d_model   [B*L, D, d_model]
    -> Add learnable feature position embeddings
    -> N Transformer encoder layers (self-attention across features)
    -> Pool across features (CLS token or mean)
    -> Linear score head -> [B, L] scores
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
        dim_feedforward=128,
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
            dropout: dropout rate.
            pooling: how to aggregate feature tokens into a score.
                "cls" — prepend a learnable [CLS] token, use its output.
                "mean" — average pool over all feature token outputs.
        """
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.pooling = pooling

        # Embed each scalar feature value into d_model dimensions.
        # Separate projection per feature so the model can learn
        # feature-specific embeddings (like per-feature towers in GAM).
        self.feature_embed = nn.Linear(1, d_model)

        # Learnable position embedding so the model knows which feature
        # is which (features have no inherent ordering).
        n_tokens = num_features + 1 if pooling == "cls" else num_features
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder: self-attention across features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
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

        # Each feature becomes a token: [B*L, D, 1] -> [B*L, D, d_model]
        x_tokens = x_flat.unsqueeze(-1)  # [B*L, D, 1]
        x_tokens = self.feature_embed(x_tokens)  # [B*L, D, d_model]

        # Prepend CLS token if using CLS pooling
        if self.pooling == "cls":
            n = x_tokens.shape[0]
            cls = self.cls_token.expand(n, -1, -1)  # [B*L, 1, d_model]
            x_tokens = torch.cat([cls, x_tokens], dim=1)  # [B*L, D+1, d_model]

        # Add positional embeddings
        x_tokens = x_tokens + self.pos_embed

        # Self-attention across features
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
        x_tokens = self.feature_embed(x_flat.unsqueeze(-1))

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
                # Manually call self-attention to get weights
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
