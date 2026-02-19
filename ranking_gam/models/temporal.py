"""
Temporal iTransformer + GAM for ranking with behavioral time series.

Implements the iTransformer × GAM architecture from the design doc:

    Option 1 ("extract"): iTransformer encodes temporal signals → per-variate
        representations fed into GAM towers → additive score.
    Option 2 ("integrated"): Pre-attention embeddings give additive base (like
        neural GAM shape functions), attention adds interaction residual.

Key components:
    RevIN: Reversible Instance Normalization for distribution shift (Kim et al.)
    TemporalEncoder: iTransformer core with optional 2D time patching
    TemporalGAMFormer: Full ranking model combining static GAM + temporal encoder

Input convention:
    The model takes a single flat tensor [B, L, D_total] where:
        D_total = num_static_features + num_variates * time_steps
    First num_static_features columns are scalar features (GAM towers).
    Remaining columns are temporal features (reshaped to [V, T] per document).

References:
    Liu et al., "iTransformer: Inverted Transformers Are Effective for Time
        Series Forecasting," ICLR 2024 (Spotlight).
    Kim et al., "Reversible Instance Normalization for Accurate Time-Series
        Forecasting against Distribution Shift," ICLR 2022.
"""

import torch
import torch.nn as nn

from .towers import LearnableMonotoneTransform, PaperTower


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalizes each variate's time series independently to zero mean and
    unit variance, handling distribution shift across variates. Learnable
    affine parameters allow the model to recover signal scale when needed.
    """

    def __init__(self, num_variates, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_variates, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_variates, 1))

    def forward(self, x):
        """Normalize each variate's time series.

        Args:
            x: [N, V, T] — N documents, V variates, T time steps.

        Returns:
            [N, V, T] normalized.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x


class TemporalEncoder(nn.Module):
    """iTransformer encoder for multivariate behavioral time series.

    Inverts the standard transformer axis: tokenizes per-variate (each
    behavioral signal's full history = one token), then applies self-attention
    across variates to learn cross-signal correlations.

    With num_patches > 1 (iTransformer2D), each variate's time series is
    split into non-overlapping patches, giving V * P tokens that capture
    both cross-variate AND cross-temporal patterns.

    Architecture:
        Input: [N, V, T]
        → RevIN normalization (optional)
        → Per-variate embedding: Linear(T or patch_size, d_model)
        → LayerNorm + positional embedding
        → Pre-norm Transformer encoder (attention across variate tokens)
        → Output: [N, V, d_model] per-variate representations
    """

    def __init__(
        self,
        num_variates,
        time_steps,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.1,
        num_patches=1,
        shared_embedding=False,
        use_revin=True,
    ):
        """
        Args:
            num_variates: number of behavioral signal channels (V).
            time_steps: length of each signal's time series (T).
            d_model: transformer hidden dimension.
            nhead: number of attention heads.
            num_layers: number of transformer encoder layers.
            dim_feedforward: FFN hidden dim (default: 4 * d_model).
            dropout: dropout rate.
            num_patches: number of time patches per variate.
                1 = standard iTransformer (full series per token).
                >1 = iTransformer2D (V * P tokens).
            shared_embedding: if True, all variates share one Linear projection.
                If False, each variate gets its own embedding (recommended when
                signals are semantically different).
            use_revin: apply Reversible Instance Normalization.
        """
        super().__init__()
        self.num_variates = num_variates
        self.time_steps = time_steps
        self.d_model = d_model
        self.num_patches = num_patches
        self.shared_embedding = shared_embedding

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # RevIN
        self.revin = RevIN(num_variates) if use_revin else None

        # Embedding dimension
        if num_patches > 1:
            assert time_steps % num_patches == 0, (
                f"time_steps ({time_steps}) must be divisible by "
                f"num_patches ({num_patches})"
            )
            self.patch_size = time_steps // num_patches
        else:
            self.patch_size = time_steps
        embed_input_dim = self.patch_size

        # Per-variate or shared embedding
        if shared_embedding:
            self.embed = nn.Linear(embed_input_dim, d_model)
        else:
            self.embed = nn.ModuleList([
                nn.Linear(embed_input_dim, d_model)
                for _ in range(num_variates)
            ])

        self.embed_norm = nn.LayerNorm(d_model)

        # Position embeddings
        n_tokens = num_variates * num_patches if num_patches > 1 else num_variates
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Pre-norm transformer encoder
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

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "pos_embed" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, x):
        """Embed variate time series into tokens.

        Args:
            x: [N, V, T] time series (after RevIN).

        Returns:
            [N, num_tokens, d_model] embedded tokens.
        """
        N, V, T = x.shape

        if self.num_patches > 1:
            # Split into patches: [N, V, P, patch_size]
            P = self.num_patches
            x_patches = x.reshape(N, V, P, self.patch_size)

            if self.shared_embedding:
                # [N, V, P, patch_size] → [N, V*P, d_model]
                tokens = self.embed(x_patches.reshape(N, V * P, self.patch_size))
            else:
                # Per-variate embedding, shared across patches of same variate
                patch_tokens = []
                for v in range(V):
                    vt = self.embed[v](x_patches[:, v])  # [N, P, d_model]
                    patch_tokens.append(vt)
                tokens = torch.cat(patch_tokens, dim=1)  # [N, V*P, d_model]
        else:
            if self.shared_embedding:
                tokens = self.embed(x)  # [N, V, d_model]
            else:
                tokens = torch.stack([
                    self.embed[v](x[:, v])
                    for v in range(V)
                ], dim=1)  # [N, V, d_model]

        return self.embed_norm(tokens)

    def forward(self, x, return_pre_attention=False):
        """Encode multivariate time series.

        Args:
            x: [N, V, T] — N documents, V variates, T time steps.
            return_pre_attention: if True, also return pre-attention embeddings
                for the "integrated" mode's additive base.

        Returns:
            h: [N, V, d_model] per-variate representations (post-attention).
            h_pre: [N, V, d_model] pre-attention embeddings (only if requested).
        """
        if self.revin is not None:
            x = self.revin(x)

        tokens = self._embed(x)  # [N, num_tokens, d_model]
        tokens = tokens + self.pos_embed

        h_pre = tokens  # before attention (additive base)
        h = self.encoder(tokens)  # after attention (with interactions)

        if self.num_patches > 1:
            # Aggregate patches back to per-variate: [N, V*P, d_model] → [N, V, d_model]
            N = h.shape[0]
            h = h.reshape(N, self.num_variates, self.num_patches, self.d_model)
            h = h.mean(dim=2)  # [N, V, d_model]

            if return_pre_attention:
                h_pre = h_pre.reshape(N, self.num_variates, self.num_patches, self.d_model)
                h_pre = h_pre.mean(dim=2)

        if return_pre_attention:
            return h, h_pre
        return h

    def get_variate_attention(self, x):
        """Extract cross-variate attention weights for interpretability.

        Returns the average attention matrix showing which behavioral
        signals attend to each other — the learned correlation structure.

        Args:
            x: [N, V, T] time series.

        Returns:
            [V, V] average attention matrix across layers and batch.
        """
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)

        if self.revin is not None:
            x = self.revin(x)

        tokens = self._embed(x)
        tokens = tokens + self.pos_embed

        all_attn = []
        h = tokens
        with torch.no_grad():
            for layer in self.encoder.layers:
                h_norm = layer.norm1(h)
                _, attn_w = layer.self_attn(
                    h_norm, h_norm, h_norm, need_weights=True,
                    average_attn_weights=True,
                )
                all_attn.append(attn_w)
                h = layer(h)

        attn_stack = torch.stack(all_attn, dim=0)  # [layers, N, T, T]
        avg_attn = attn_stack.mean(dim=(0, 1))  # [T, T]

        if self.num_patches > 1:
            # Aggregate patch-level attention to variate-level
            V = self.num_variates
            P = self.num_patches
            avg_attn = avg_attn.reshape(V, P, V, P).mean(dim=(1, 3))

        return avg_attn.cpu()


class TemporalGAMFormer(nn.Module):
    """GAM + iTransformer for ranking with static and temporal features.

    Combines interpretable GAM main effects on static scalar features with
    iTransformer-encoded temporal behavioral signals:

        score = Σ f_j(x_static_j) + temporal_score + bias
                ↑ static GAM         ↑ from iTransformer

    Input is a single flat tensor [B, L, D_total] where:
        D_total = num_static_features + num_variates * time_steps
    This works with the existing train_model / DataLoader pipeline unchanged.

    Two modes for the temporal branch:

        "extract" (Option 1): iTransformer encoder → per-variate MLP towers
            → additive. Each variate gets its own interpretable tower on
            the learned representation. Shape functions are univariate.

        "integrated" (Option 2): Pre-attention embeddings → per-variate
            score heads (additive base, no cross-signal info). Post-attention
            → pooled interaction head (captures cross-signal effects).
            score = additive_base + weight * interaction_residual.
            L1 penalty on interaction_residual pushes toward the additive solution.
    """

    def __init__(
        self,
        # Static features
        num_static_features=0,
        static_hidden_dims=None,
        static_residual=False,
        static_feature_transforms=False,
        num_transform_knots=20,
        activation="relu",
        # Temporal features
        num_variates=10,
        time_steps=30,
        num_patches=1,
        shared_embedding=False,
        use_revin=True,
        # Transformer config
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=None,
        dropout=0.1,
        # Temporal tower config (for "extract" mode)
        temporal_hidden_dims=None,
        # Combination
        mode="extract",
        interaction_weight=1.0,
    ):
        """
        Args:
            num_static_features: number of scalar static features (0 = temporal only).
            static_hidden_dims: GAM tower hidden dims (default: [16, 8]).
            static_residual: add skip connection in static GAM towers.
            static_feature_transforms: use learnable monotone transforms on static features.
            num_transform_knots: knots for monotone transforms.
            activation: activation function for MLP towers.
            num_variates: number of behavioral signal channels (V).
            time_steps: time series length per variate (T).
            num_patches: iTransformer2D patches (1 = standard).
            shared_embedding: share embedding across variates.
            use_revin: apply RevIN normalization.
            d_model: transformer hidden dimension.
            nhead: attention heads.
            num_layers: transformer layers.
            dim_feedforward: FFN hidden dim (default: 4 * d_model).
            dropout: transformer dropout.
            temporal_hidden_dims: tower dims for "extract" mode (default: [32, 16]).
            mode: "extract" (Option 1) or "integrated" (Option 2).
            interaction_weight: initial weight for interaction term (Option 2).
        """
        super().__init__()
        if static_hidden_dims is None:
            static_hidden_dims = [16, 8]
        if temporal_hidden_dims is None:
            temporal_hidden_dims = [32, 16]
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.num_static_features = num_static_features
        self.num_variates = num_variates
        self.time_steps = time_steps
        self.mode = mode
        self.d_model = d_model

        # ── Static GAM branch ──
        if num_static_features > 0:
            self.static_towers = nn.ModuleList([
                PaperTower(1, static_hidden_dims, dropout=0.0,
                           residual=static_residual, activation=activation)
                for _ in range(num_static_features)
            ])
            self.feature_transforms = None
            if static_feature_transforms:
                self.feature_transforms = nn.ModuleList([
                    LearnableMonotoneTransform(num_knots=num_transform_knots)
                    for _ in range(num_static_features)
                ])
        else:
            self.static_towers = None
            self.feature_transforms = None

        # ── Temporal iTransformer branch ──
        self.temporal_encoder = TemporalEncoder(
            num_variates=num_variates,
            time_steps=time_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_patches=num_patches,
            shared_embedding=shared_embedding,
            use_revin=use_revin,
        )

        if mode == "extract":
            # Option 1: per-variate MLP towers on encoded representations
            self.variate_towers = nn.ModuleList([
                PaperTower(d_model, temporal_hidden_dims, dropout=0.0,
                           residual=False, activation=activation)
                for _ in range(num_variates)
            ])
        elif mode == "integrated":
            # Option 2: additive base + interaction residual
            self.variate_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                )
                for _ in range(num_variates)
            ])
            self.interaction_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.interaction_weight = nn.Parameter(
                torch.tensor(float(interaction_weight))
            )
        else:
            raise ValueError(f"mode must be 'extract' or 'integrated', got {mode!r}")

        self.global_bias = nn.Parameter(torch.zeros(1))

    @property
    def total_input_features(self):
        """Total expected input dimension."""
        return self.num_static_features + self.num_variates * self.time_steps

    def init_transforms_from_data(self, X):
        """Initialize static feature transforms from training data percentiles.

        Args:
            X: [B, L, D] numpy array (only first num_static_features columns used).
        """
        if self.feature_transforms is None:
            return
        for j, transform in enumerate(self.feature_transforms):
            transform.init_from_data(X[:, :, j])

    def forward(self, x):
        """
        Args:
            x: [B, L, D_total] where D_total = D_static + V * T.

        Returns:
            [B, L] relevance scores.
        """
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)

        score = torch.zeros(B * L, device=x.device)

        # ── Static GAM branch ──
        if self.static_towers is not None and self.num_static_features > 0:
            x_static = x_flat[:, :self.num_static_features]
            for i, tower in enumerate(self.static_towers):
                feat = x_static[:, i:i+1]
                if self.feature_transforms is not None:
                    feat = self.feature_transforms[i](feat)
                score = score + tower(feat).squeeze(-1)

        # ── Temporal branch ──
        x_temporal = x_flat[:, self.num_static_features:]
        x_temporal = x_temporal.reshape(B * L, self.num_variates, self.time_steps)

        if self.mode == "extract":
            # Option 1: full encoder → per-variate towers → additive
            h = self.temporal_encoder(x_temporal)  # [B*L, V, d_model]
            for v in range(self.num_variates):
                score = score + self.variate_towers[v](h[:, v]).squeeze(-1)

        elif self.mode == "integrated":
            # Option 2: additive base (pre-attention) + interaction (post-attention)
            h_post, h_pre = self.temporal_encoder(
                x_temporal, return_pre_attention=True
            )

            # Additive base: per-variate scores from pre-attention embeddings
            for v in range(self.num_variates):
                score = score + self.variate_heads[v](h_pre[:, v]).squeeze(-1)

            # Interaction residual: pooled post-attention representation
            pooled = h_post.mean(dim=1)  # [B*L, d_model]
            interaction = self.interaction_head(pooled).squeeze(-1)
            score = score + self.interaction_weight * interaction

        return score.reshape(B, L) + self.global_bias

    def get_static_main_effect(self, feature_idx, x_values):
        """Evaluate static GAM tower f_j(x) for visualization.

        Args:
            feature_idx: which static feature tower.
            x_values: 1D array of feature values.

        Returns:
            numpy array of tower outputs.
        """
        if self.static_towers is None:
            raise ValueError("No static features configured")
        self.eval()
        x_values = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
        x_values = x_values.to(next(self.parameters()).device)
        with torch.no_grad():
            feat = x_values
            if self.feature_transforms is not None:
                feat = self.feature_transforms[feature_idx](feat)
            return self.static_towers[feature_idx](feat).cpu().numpy().flatten()

    def get_variate_attention(self, x):
        """Get cross-variate attention matrix from temporal encoder.

        Args:
            x: [B, L, D_total] full input tensor.

        Returns:
            [V, V] average attention matrix.
        """
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        x_temporal = x_flat[:, self.num_static_features:]
        x_temporal = x_temporal.reshape(B * L, self.num_variates, self.time_steps)
        return self.temporal_encoder.get_variate_attention(x_temporal)

    def explain(self, x_single):
        """Decompose a single document's score into components.

        Args:
            x_single: [D_total] or [1, 1, D_total] feature vector.

        Returns:
            dict with:
                static_contributions: [D_static] per-feature GAM scores
                static_score: sum of static contributions
                variate_contributions: [V] per-variate temporal scores
                temporal_score: total temporal contribution
                interaction_score: interaction term (integrated mode only)
                interaction_weight: weight of interaction (integrated mode only)
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
            result = {}

            # Static contributions
            static_score = 0.0
            if self.static_towers is not None and self.num_static_features > 0:
                static_vals = []
                x_static = x_flat[:, :self.num_static_features]
                for i, tower in enumerate(self.static_towers):
                    feat = x_static[:, i:i+1]
                    if self.feature_transforms is not None:
                        feat = self.feature_transforms[i](feat)
                    static_vals.append(tower(feat).item())
                result["static_contributions"] = torch.tensor(static_vals)
                static_score = sum(static_vals)
                result["static_score"] = static_score

            # Temporal contributions
            x_temporal = x_flat[:, self.num_static_features:]
            x_temporal = x_temporal.reshape(1, self.num_variates, self.time_steps)

            if self.mode == "extract":
                h = self.temporal_encoder(x_temporal)
                variate_vals = []
                for v in range(self.num_variates):
                    variate_vals.append(
                        self.variate_towers[v](h[:, v]).item()
                    )
                result["variate_contributions"] = torch.tensor(variate_vals)
                result["temporal_score"] = sum(variate_vals)

            elif self.mode == "integrated":
                h_post, h_pre = self.temporal_encoder(
                    x_temporal, return_pre_attention=True
                )
                variate_vals = []
                for v in range(self.num_variates):
                    variate_vals.append(
                        self.variate_heads[v](h_pre[:, v]).item()
                    )
                result["variate_contributions"] = torch.tensor(variate_vals)
                additive_base = sum(variate_vals)
                result["additive_base"] = additive_base

                pooled = h_post.mean(dim=1)
                interaction = self.interaction_head(pooled).item()
                weight = self.interaction_weight.item()
                result["interaction_score"] = interaction
                result["interaction_weight"] = weight
                result["temporal_score"] = additive_base + weight * interaction

            result["bias"] = self.global_bias.item()
            result["total_score"] = (
                static_score + result["temporal_score"] + result["bias"]
            )

        return result

    def static_param_count(self):
        """Count parameters in static GAM branch."""
        count = 0
        if self.static_towers is not None:
            count += sum(p.numel() for p in self.static_towers.parameters())
        if self.feature_transforms is not None:
            count += sum(p.numel() for p in self.feature_transforms.parameters())
        return count

    def temporal_param_count(self):
        """Count parameters in temporal encoder + towers."""
        count = sum(p.numel() for p in self.temporal_encoder.parameters())
        if self.mode == "extract":
            count += sum(p.numel() for p in self.variate_towers.parameters())
        elif self.mode == "integrated":
            count += sum(p.numel() for p in self.variate_heads.parameters())
            count += sum(p.numel() for p in self.interaction_head.parameters())
            count += self.interaction_weight.numel()
        return count
