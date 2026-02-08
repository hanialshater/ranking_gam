"""Pairwise logistic loss (RankNet)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseLoss(nn.Module):
    """
    For each pair (i,j) where y_i > y_j:
        loss += log(1 + exp(-sigma * (s_i - s_j)))
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, y_pred, y_true):
        mask = (y_true >= 0).float()
        y_true = torch.clamp(y_true, min=0) * mask
        y_pred = y_pred * mask

        label_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)

        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)
        pair_mask = mask_2d * (label_diff > 0).float()

        loss = F.softplus(-self.sigma * pred_diff) * pair_mask
        num_pairs = pair_mask.sum(dim=[1, 2]).clamp(min=1e-10)
        return (loss.sum(dim=[1, 2]) / num_pairs).mean()
