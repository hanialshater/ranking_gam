"""
LambdaLoss framework -- allRank-faithful implementation.

"The LambdaLoss Framework for Ranking Metric Optimization" (Wang et al. 2018).

Unifies multiple pairwise losses via different weighting schemes:
  - None / 'rankNet':   plain pairwise logistic
  - 'lambdaRank':       classic LambdaRank with |delta-DCG| weights
  - 'ndcgLoss1':        NDCGLoss1 from the paper
  - 'ndcgLoss2':        NDCGLoss2 from the paper
  - 'ndcgLoss2++':      NDCGLoss2++ (best in paper)
"""

import torch
import torch.nn as nn


class LambdaLoss(nn.Module):

    def __init__(
        self,
        weighing_scheme="lambdaRank",
        sigma=1.0,
        k=10,
        mu=10.0,
        reduction="sum",
        reduction_log="binary",
    ):
        super().__init__()
        self.weighing_scheme = weighing_scheme
        self.sigma = sigma
        self.k = k
        self.mu = mu
        self.reduction = reduction
        self.reduction_log = reduction_log

    def forward(self, y_pred, y_true):
        device = y_pred.device
        eps = 1e-10

        y_pred_c = y_pred.clone()
        y_true_c = y_true.clone().float()

        padded = y_true_c < 0
        y_pred_c[padded] = float("-inf")
        y_true_c[padded] = float("-inf")

        y_pred_sorted, indices_pred = y_pred_c.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true_c.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true_c, dim=1, index=indices_pred)

        true_diffs = (
            true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        )
        padded_pairs_mask = torch.isfinite(true_diffs)

        if self.weighing_scheme != "ndcgLoss1":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        L = y_pred.shape[1]
        ndcg_at_k_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
        k = min(self.k, L)
        ndcg_at_k_mask[:k, :k] = True

        true_sorted_by_preds_clean = true_sorted_by_preds.clamp(min=0.0)
        y_true_sorted_clean = y_true_sorted.clamp(min=0.0)

        pos_idxs = torch.arange(1, L + 1, device=device).float()
        D = torch.log2(1.0 + pos_idxs)[None, :]
        maxDCGs = (
            torch.sum(
                ((torch.pow(2, y_true_sorted_clean) - 1) / D)[:, :k], dim=-1
            ).clamp(min=eps)
        )
        G = (torch.pow(2, true_sorted_by_preds_clean) - 1) / maxDCGs[:, None]

        weights = self._compute_weights(G, D, true_sorted_by_preds_clean)

        scores_diffs = (
            y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]
        ).clamp(-1e8, 1e8)
        scores_diffs = torch.where(
            torch.isfinite(scores_diffs), scores_diffs, torch.zeros_like(scores_diffs)
        )

        weighted_probas = (
            torch.sigmoid(self.sigma * scores_diffs).clamp(min=eps) ** weights
        ).clamp(min=eps)

        if self.reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            losses = torch.log(weighted_probas)

        combined_mask = padded_pairs_mask & ndcg_at_k_mask

        if self.reduction == "sum":
            loss = -losses[combined_mask].sum()
        else:
            loss = (
                -losses[combined_mask].mean()
                if combined_mask.any()
                else torch.tensor(0.0, device=device)
            )

        return loss / y_pred.shape[0]

    def _compute_weights(self, G, D, true_sorted):
        if self.weighing_scheme is None or self.weighing_scheme == "rankNet":
            return 1.0

        elif self.weighing_scheme == "ndcgLoss1":
            return (G / D)[:, :, None]

        elif self.weighing_scheme == "ndcgLoss2":
            return self._ndcg2_weights(G, D)

        elif self.weighing_scheme == "lambdaRank":
            return torch.abs(
                torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
            ) * torch.abs(G[:, :, None] - G[:, None, :])

        elif self.weighing_scheme == "ndcgLoss2++":
            ndcg2 = self._ndcg2_weights(G, D)
            lambdarank = torch.abs(
                torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
            ) * torch.abs(G[:, :, None] - G[:, None, :])
            return self.mu * ndcg2 + lambdarank

        else:
            raise ValueError(f"Unknown weighing scheme: {self.weighing_scheme}")

    def _ndcg2_weights(self, G, D):
        L = G.shape[1]
        pos_idxs = torch.arange(1, L + 1, device=G.device)
        delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
        deltas = torch.abs(
            torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.0)
            - torch.pow(torch.abs(D[0, delta_idxs]), -1.0)
        )
        deltas.diagonal().zero_()
        return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])
