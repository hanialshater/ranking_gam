"""
Differentiable NDCG via soft ranking.

Includes:
  - soft_rank: pairwise sigmoid soft ranking
  - DiffSortNDCGLoss: differentiable NDCG using soft ranks
  - Isotonic regression helpers (PAVA) for Blondel et al. 2020 approach
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pava_l2_non_decreasing(y_np):
    """
    Pool Adjacent Violators for non-decreasing L2 isotonic regression.

    Solves: min_s ||s - y||^2  s.t. s_1 <= s_2 <= ... <= s_n

    Returns (solution, partition).
    O(n) time, O(n) space.
    """
    n = len(y_np)
    starts, ends, sums, counts = [0], [1], [float(y_np[0])], [1]

    for i in range(1, n):
        starts.append(i)
        ends.append(i + 1)
        sums.append(float(y_np[i]))
        counts.append(1)

        while len(starts) > 1 and (sums[-2] / counts[-2]) > (sums[-1] / counts[-1]):
            sums[-2] += sums[-1]
            counts[-2] += counts[-1]
            ends[-2] = ends[-1]
            starts.pop()
            ends.pop()
            sums.pop()
            counts.pop()

    result = np.empty(n, dtype=np.float64)
    partition = []
    for s, e, sm, c in zip(starts, ends, sums, counts):
        result[s:e] = sm / c
        partition.append((s, e))

    return result, partition


def _pava_l2_non_increasing(y_np):
    """Non-increasing isotonic regression via negate -> PAVA -> negate."""
    neg_sol, partition = _pava_l2_non_decreasing(-y_np)
    return -neg_sol, partition


class _IsotonicL2(torch.autograd.Function):
    """Batched L2 isotonic regression (non-decreasing) with analytic backward."""

    @staticmethod
    def forward(ctx, input_tensor):
        B, N = input_tensor.shape
        output = torch.empty_like(input_tensor)
        all_partitions = []

        input_np = input_tensor.detach().cpu().numpy()
        for b in range(B):
            sol, partition = _pava_l2_non_decreasing(input_np[b])
            output[b] = torch.tensor(
                sol, dtype=input_tensor.dtype, device=input_tensor.device
            )
            all_partitions.append(partition)

        ctx.all_partitions = all_partitions
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        for b, partition in enumerate(ctx.all_partitions):
            for s, e in partition:
                if e - s > 1:
                    grad_input[b, s:e] = grad_output[b, s:e].mean()
        return grad_input


class _IsotonicNonIncreasing(torch.autograd.Function):
    """Batched non-increasing isotonic regression with analytic backward."""

    @staticmethod
    def forward(ctx, input_tensor):
        B, N = input_tensor.shape
        output = torch.empty_like(input_tensor)
        all_partitions = []

        input_np = input_tensor.detach().cpu().numpy()
        for b in range(B):
            sol, partition = _pava_l2_non_increasing(input_np[b])
            output[b] = torch.tensor(
                sol, dtype=input_tensor.dtype, device=input_tensor.device
            )
            all_partitions.append(partition)

        ctx.all_partitions = all_partitions
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        for b, partition in enumerate(ctx.all_partitions):
            for s, e in partition:
                if e - s > 1:
                    grad_input[b, s:e] = grad_output[b, s:e].mean()
        return grad_input


def soft_rank(values, regularization_strength=1.0):
    """
    Differentiable soft ranking via pairwise sigmoid comparisons.

    For item i, descending soft rank:
        rank_i = 1 + sum_{j!=i} sigma((v_j - v_i) / tau)

    Properties:
      - Ranks always in [1, N] (bounded by construction)
      - tau->0: converges to hard ranks
      - tau->inf: all ranks -> (N+1)/2
      - Fully differentiable, O(n^2) per sample

    Args:
        values: [B, N] tensor of scores
        regularization_strength: temperature tau

    Returns:
        [B, N] tensor of 1-indexed descending soft ranks
    """
    B, N = values.shape
    tau = regularization_strength

    v_i = values.unsqueeze(-1)
    v_j = values.unsqueeze(-2)
    diff = (v_j - v_i) / tau

    eye = torch.eye(N, device=values.device, dtype=values.dtype).unsqueeze(0)
    sig = torch.sigmoid(diff) * (1.0 - eye)

    ranks = 1.0 + sig.sum(dim=-1)
    return ranks


class DiffSortNDCGLoss(nn.Module):
    """
    Differentiable NDCG via sigmoid soft ranking.

    Properties:
      - Ranks bounded to [1, N]
      - Temperature tau controls smoothness
      - Fully vectorized, O(n^2) per sample
    """

    def __init__(self, k=10, regularization_strength=1.0):
        super().__init__()
        self.k = k
        self.reg = regularization_strength

    def forward(self, y_pred, y_true):
        device = y_pred.device
        dtype = y_pred.dtype
        eps = 1e-10

        mask = (y_true >= 0).float()
        y_true_clean = y_true.clamp(min=0)

        pred_min = y_pred.min().detach()
        y_pred_masked = y_pred * mask + (1 - mask) * (pred_min - 100.0)

        sranks = soft_rank(y_pred_masked, self.reg)

        gains = (torch.pow(2.0, y_true_clean) - 1.0) * mask
        discounts = 1.0 / torch.log2(1.0 + sranks)

        topk_weight = torch.sigmoid(5.0 * (self.k + 0.5 - sranks)) * mask
        dcg = (gains * discounts * topk_weight).sum(dim=-1)

        ideal_sorted = y_true_clean.sort(descending=True, dim=-1)[0]
        k = min(self.k, ideal_sorted.shape[-1])
        ideal_gains = torch.pow(2.0, ideal_sorted[:, :k]) - 1.0
        ideal_disc = 1.0 / torch.log2(
            torch.arange(2, k + 2, device=device, dtype=dtype)
        )
        idcg = (ideal_gains * ideal_disc).sum(dim=-1).clamp(min=eps)

        ndcg = dcg / idcg
        valid = idcg > eps
        num_valid = valid.float().sum().clamp(min=1.0)
        return ((1.0 - ndcg) * valid.float()).sum() / num_valid
