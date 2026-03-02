"""
PiRank: differentiable NDCG via NeuralSort permutation matrices.

Based on:
  - NeuralSort (Grover et al., NeurIPS 2019)
  - PiRank (Swezey et al., NeurIPS 2021)

NeuralSort produces a soft permutation matrix P where P[i,j] is the
probability that item i is placed at position j (descending by score).
DCG is computed by routing gains through P to positions with discounts.
"""

import torch
import torch.nn as nn


def neural_sort(scores, tau=1.0):
    """
    NeuralSort: differentiable relaxation of argsort (descending).

    Given scores s ∈ R^n, produces a soft permutation matrix P where
    P[i,j] ≈ 1 if item i should be at position j.

    For position j (1-indexed, descending):
        logit_i = ((n+1-2j) * s_i - Σ_k |s_i - s_k|) / τ
        P[:, j] = softmax(logit / τ)

    As τ→0, P converges to the hard permutation matrix of argsort.

    Args:
        scores: [B, N] predicted scores
        tau: temperature (lower = sharper, default 1.0)

    Returns:
        [B, N, N] soft permutation matrix (rows = items, cols = positions)
    """
    B, N = scores.shape

    # Pairwise absolute differences: A[b,i,j] = |s_i - s_j|
    A = torch.abs(scores.unsqueeze(-1) - scores.unsqueeze(-2))  # [B, N, N]
    A_sum = A.sum(dim=-1)  # [B, N]

    # Position scaling: (N+1-2j) for j=1..N
    positions = torch.arange(1, N + 1, device=scores.device, dtype=scores.dtype)
    scaling = N + 1 - 2 * positions  # [N]

    # logits[b, i, j] = (scaling[j] * s[b,i] - A_sum[b,i]) / tau
    logits = (
        scaling.unsqueeze(0).unsqueeze(0) * scores.unsqueeze(-1)
        - A_sum.unsqueeze(-1)
    ) / tau  # [B, N, N]

    # Softmax over items (dim=1) for each position
    P = torch.softmax(logits, dim=1)  # [B, N, N]
    return P


class PiRankNDCGLoss(nn.Module):
    """
    Differentiable NDCG loss via NeuralSort permutation matrices.

    Computes a soft DCG by routing item gains through the soft permutation
    matrix to position-based discounts. The loss is 1 - NDCG (averaged
    over queries with non-zero IDCG).

    Properties:
      - Produces doubly-stochastic-like permutation matrices (proper
        probability distribution over item-position assignments)
      - As tau→0, converges to exact NDCG
      - O(n^2) memory and compute per query (permutation matrix is n×n)

    Args:
        k: NDCG cutoff (positions beyond k get zero discount)
        tau: NeuralSort temperature (default 1.0)
    """

    def __init__(self, k=10, tau=1.0):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, N] predicted scores
            y_true: [B, N] relevance labels (-1 = padding)

        Returns:
            scalar loss (1 - mean NDCG)
        """
        device = y_pred.device
        dtype = y_pred.dtype
        eps = 1e-10

        # Handle padding (label -1)
        mask = (y_true >= 0).float()
        y_true_clean = y_true.clamp(min=0)

        # Push padded items to very low scores
        pred_min = y_pred.min().detach()
        y_pred_masked = y_pred * mask + (1 - mask) * (pred_min - 100.0)

        B, N = y_pred.shape
        k = min(self.k, N)

        # Soft permutation matrix: P[b,i,j] = prob(item i at position j)
        P = neural_sort(y_pred_masked, self.tau)  # [B, N, N]

        # Gains per item
        gains = (torch.pow(2.0, y_true_clean) - 1.0) * mask  # [B, N]

        # Position discounts (zero beyond k)
        positions = torch.arange(1, N + 1, device=device, dtype=dtype)
        discounts = torch.where(
            positions <= k,
            1.0 / torch.log2(1.0 + positions),
            torch.zeros_like(positions),
        )  # [N]

        # Soft DCG: each item's expected discount = sum_j P[i,j] * discount[j]
        expected_discount = torch.matmul(P, discounts)  # [B, N]
        dcg = (gains * expected_discount).sum(dim=-1)  # [B]

        # IDCG (hard, exact)
        ideal_sorted = y_true_clean.sort(descending=True, dim=-1)[0]
        ideal_gains = torch.pow(2.0, ideal_sorted[:, :k]) - 1.0
        ideal_disc = 1.0 / torch.log2(
            torch.arange(2, k + 2, device=device, dtype=dtype)
        )
        idcg = (ideal_gains * ideal_disc).sum(dim=-1).clamp(min=eps)

        ndcg = dcg / idcg
        valid = idcg > eps
        num_valid = valid.float().sum().clamp(min=1.0)
        return ((1.0 - ndcg) * valid.float()).sum() / num_valid
