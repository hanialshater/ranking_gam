"""
ApproxNDCG loss -- allRank-faithful implementation.

Based on "A General Approximation Framework for Direct Optimization of
Information Retrieval Measures" (Qin et al. 2010).

Algorithm (follows allRank/allegro exactly):
  1. Sort predictions descending
  2. Gather true labels in that order
  3. Mask padded pairs via isfinite + diagonal zeroing
  4. approx_rank_i = 1 + sum_{j!=i} sigma(-alpha * (s_i - s_j))
  5. DCG with approximate ranks, normalize by IDCG
"""

import torch
import torch.nn as nn


class ApproxNDCGLoss(nn.Module):
    """
    Two conventions exist for the sharpness parameter:
      - allRank: alpha (multiplier on score diffs).  Default alpha=1.
      - TF-Ranking: temperature (divisor). T = 1/alpha.

    For GAM architectures with small score spreads, alpha=10 (T=0.1) helps.
    """

    def __init__(self, alpha=10.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [batch, list_size] predicted scores
            y_true: [batch, list_size] relevance labels (-1 = padded)
        """
        device = y_pred.device
        eps = 1e-10

        y_pred_c = y_pred.clone()
        y_true_c = y_true.clone().float()

        padded = y_true_c < 0
        y_pred_c[padded] = float("-inf")
        y_true_c[padded] = float("-inf")

        y_pred_sorted, indices = y_pred_c.sort(descending=True, dim=-1)
        true_sorted_by_preds = torch.gather(y_true_c, dim=1, index=indices)

        true_diffs = (
            true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        )
        pairs_mask = torch.isfinite(true_diffs).float()
        pairs_mask = pairs_mask * (
            1.0 - torch.eye(y_pred.shape[1], device=device).unsqueeze(0)
        )

        scores_diffs = y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]
        pairs_bool = pairs_mask > 0.5
        scores_diffs = torch.where(
            pairs_bool, scores_diffs, torch.zeros_like(scores_diffs)
        )
        approx_pos = 1.0 + torch.sum(
            pairs_mask * torch.sigmoid(-self.alpha * scores_diffs).clamp(min=eps),
            dim=-1,
        )

        true_clean = torch.clamp(true_sorted_by_preds, min=0)
        gains = torch.pow(2.0, true_clean) - 1.0
        gains[~torch.isfinite(true_sorted_by_preds)] = 0.0

        discounts = 1.0 / torch.log2(approx_pos + 1.0)
        approx_dcg = (gains * discounts).sum(dim=-1)

        sorted_gains, _ = torch.sort(gains, dim=-1, descending=True)
        positions = torch.arange(1, y_pred.shape[1] + 1, device=device).float()
        ideal_discounts = 1.0 / torch.log2(positions + 1.0)
        ideal_dcg = (sorted_gains * ideal_discounts).sum(dim=-1)

        valid = ideal_dcg > 0
        ndcg = torch.where(
            valid, approx_dcg / (ideal_dcg + eps), torch.zeros_like(approx_dcg)
        )

        num_valid = valid.float().sum().clamp(min=1.0)
        return ((1.0 - ndcg) * valid.float()).sum() / num_valid
