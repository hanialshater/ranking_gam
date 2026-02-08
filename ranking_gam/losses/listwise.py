"""
Listwise ranking losses: ListMLE and ListNet.

ListMLE: "Listwise Approach to Learning to Rank" (Xia et al. 2008)
ListNet: "Learning to Rank: From Pairwise to Listwise" (Cao et al. 2007)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ListMLELoss(nn.Module):
    """
    allRank-faithful ListMLE.

    Shuffles for randomized tie resolution, then sorts by true labels.
    Loss = -sum_i (s_{pi(i)} - log sum_{j>=i} exp(s_{pi(j)}))
    """

    def forward(self, y_pred, y_true):
        mask = (y_true >= 0).float()

        valid_queries = mask.sum(dim=-1) > 0
        if not valid_queries.any():
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        y_pred_c = y_pred.clone()
        y_true_c = y_true.clone().float()
        y_pred_c[y_true < 0] = float("-inf")
        y_true_c[y_true < 0] = float("-inf")

        random_indices = torch.randperm(y_pred.shape[-1], device=y_pred.device)
        y_pred_shuffled = y_pred_c[:, random_indices]
        y_true_shuffled = y_true_c[:, random_indices]
        mask_shuffled = mask[:, random_indices]

        _, indices = y_true_shuffled.sort(descending=True, dim=-1)
        preds_sorted_by_true = torch.gather(y_pred_shuffled, 1, indices)
        mask_sorted = torch.gather(mask_shuffled, 1, indices)

        preds_sorted_by_true[mask_sorted < 0.5] = float("-inf")

        safe_preds = torch.where(
            mask_sorted > 0.5,
            preds_sorted_by_true,
            torch.full_like(preds_sorted_by_true, -1e9),
        )
        max_pred = safe_preds.max(dim=-1, keepdim=True)[0]
        preds_stable = preds_sorted_by_true - max_pred

        preds_stable = preds_stable.clamp(min=-1e6)

        cumsums = torch.logcumsumexp(preds_stable.flip(dims=[-1]), dim=-1).flip(
            dims=[-1]
        )

        observation_loss = -(preds_stable - cumsums) * mask_sorted

        query_losses = observation_loss.sum(dim=-1)
        return query_losses[valid_queries].mean()


class ListNetLoss(nn.Module):
    """
    allRank-faithful ListNet (top-1 probability variant).

    Cross-entropy between softmax(labels) and softmax(predictions).
    """

    def forward(self, y_pred, y_true):
        y_pred_c = y_pred.clone()
        y_true_c = y_true.clone().float()

        padded = y_true_c < 0
        y_pred_c[padded] = float("-inf")
        y_true_c[padded] = float("-inf")

        preds_smax = F.softmax(y_pred_c, dim=1)
        true_smax = F.softmax(y_true_c, dim=1)

        preds_log = torch.log(preds_smax + 1e-10)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
