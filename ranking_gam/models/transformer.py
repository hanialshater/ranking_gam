"""
Black-box Transformer baseline for learning-to-rank.

A standard self-attention model over the list of documents. Unlike GAM,
this model allows cross-document interactions (each document attends to
all others in the list), making it a full black-box. This serves as an
upper-bound comparison: how much NDCG are we leaving on the table by
requiring interpretability?

Architecture:
    Input: [B, L, D] features
    -> Linear projection to d_model
    -> N Transformer encoder layers (self-attention + FFN)
    -> Linear(d_model, 1) per-document score
    Output: [B, L] scores
"""

import torch
import torch.nn as nn
import math


class TransformerRanker(nn.Module):
    """Black-box Transformer ranker (non-interpretable baseline).

    Uses self-attention so each document's score depends on ALL other
    documents in the list. This is the key difference from GAM: the model
    can learn cross-document interactions, not just per-feature effects.
    """

    def __init__(
        self,
        num_features=136,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        """
        Args:
            num_features: input feature dimension per document
            d_model: transformer hidden dimension
            nhead: number of attention heads
            num_layers: number of transformer encoder layers
            dim_feedforward: FFN hidden dimension
            dropout: dropout rate
        """
        super().__init__()
        self.num_features = num_features

        # Project features to d_model
        self.input_proj = nn.Linear(num_features, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Score head: per-document score
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] document features

        Returns:
            [B, L] relevance scores
        """
        # Project input features
        h = self.input_proj(x)  # [B, L, d_model]

        # Self-attention across documents
        h = self.encoder(h)  # [B, L, d_model]

        # Per-document scores
        scores = self.score_head(h).squeeze(-1)  # [B, L]
        return scores
