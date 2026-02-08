"""
SubmodularRankingGAM -- interpretable groupwise ranking with greedy guarantees.

Architecture:
    score(item_i | S) = sum_j f_j(x_ij)                   <- base GAM (unconstrained)
                      + sum_k g_k(diversity_k(item_i, S))  <- concave PWL (submodular)

Key properties:
    1. Base item features scored by standard GAM towers (interpretable curves)
    2. Groupwise features scored by CONCAVE PWL curves
    3. Concavity enforces submodularity -> greedy reranking is (1-1/e) optimal
    4. Every component is independently interpretable

References:
    - Nemhauser et al. 1978: greedy (1-1/e) guarantee
    - Yue & Joachims 2008: submodular diversification for IR
    - Tschiatschek et al. 2014: learning submodular functions from data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .towers import PaperTower, ConcavePWL
from .groupwise import GroupwiseFeatureComputer


class SubmodularRankingGAM(nn.Module):

    def __init__(
        self,
        num_item_features,
        groupwise_specs,
        item_hidden=None,
        num_knots=10,
        dropout=0.0,
        train_mode="pointwise",
        gumbel_tau=1.0,
    ):
        """
        Args:
            num_item_features: number of item-level features
            groupwise_specs: list of dicts for GroupwiseFeatureComputer
            item_hidden: hidden dims for item feature towers
            num_knots: knots per concave PWL
            dropout: dropout rate for item towers
            train_mode: 'pointwise' or 'soft_greedy'
            gumbel_tau: temperature for Gumbel-softmax (soft_greedy mode)
        """
        super().__init__()
        if item_hidden is None:
            item_hidden = [16, 8]

        self.num_item_features = num_item_features
        self.num_features = num_item_features  # alias
        self.groupwise_specs = groupwise_specs
        self.train_mode = train_mode
        self.gumbel_tau = gumbel_tau

        self.item_towers = nn.ModuleList(
            [
                PaperTower(1, hidden_dims=item_hidden, dropout=dropout)
                for _ in range(num_item_features)
            ]
        )

        self.diversity_towers = nn.ModuleList(
            [
                ConcavePWL(
                    num_knots=num_knots,
                    x_min=spec.get("x_min", 0.0),
                    x_max=spec.get("x_max", 1.0),
                )
                for spec in groupwise_specs
            ]
        )

        self.feature_computer = GroupwiseFeatureComputer(groupwise_specs)

    def base_scores(self, x):
        """Compute base GAM scores (item-level, set-independent).

        Args:
            x: [B, L, D] item features
        Returns:
            [B, L] base scores
        """
        B, L, D = x.shape
        x_flat = x.view(B * L, D)

        sub_scores = []
        for j in range(self.num_item_features):
            sub_scores.append(self.item_towers[j](x_flat[:, j : j + 1]))

        total = torch.stack(sub_scores, dim=-1).sum(dim=-1).squeeze(-1)
        return total.view(B, L)

    def diversity_scores(self, groupwise_features):
        """Score groupwise features through concave PWL towers.

        Args:
            groupwise_features: [n_items, num_groupwise] tensor
        Returns:
            [n_items] diversity contribution scores
        """
        total = torch.zeros(
            groupwise_features.shape[0], device=groupwise_features.device
        )
        for k, tower in enumerate(self.diversity_towers):
            total = total + tower(groupwise_features[:, k])
        return total

    def greedy_rerank(self, x, k=None, return_scores=False):
        """
        Greedy submodular reranking (inference).

        At each step, selects the item maximizing:
            marginal_gain(i|S) = base_score(i) + diversity_score(i|S)

        Args:
            x: [B, L, D] item features
            k: number of items to select (default: L)
            return_scores: if True, return (order, scores_at_selection)

        Returns:
            selected_order: [B, k] indices in selection order
        """
        B, L, D = x.shape
        if k is None:
            k = L

        base = self.base_scores(x)

        all_orders = []
        all_scores = []

        for b in range(B):
            selected = []
            remaining = list(range(L))
            item_feats = x[b]
            scores_at_sel = []

            for step in range(min(k, L)):
                if not remaining:
                    break

                cand_idx = torch.tensor(remaining, device=x.device, dtype=torch.long)
                sel_idx = (
                    torch.tensor(selected, device=x.device, dtype=torch.long)
                    if selected
                    else torch.tensor([], device=x.device, dtype=torch.long)
                )

                gw_feats = self.feature_computer.compute(cand_idx, sel_idx, item_feats)

                base_cand = base[b, cand_idx]
                div_cand = self.diversity_scores(gw_feats)
                marginal = base_cand + div_cand

                best_local = marginal.argmax().item()
                best_global = remaining[best_local]

                selected.append(best_global)
                scores_at_sel.append(marginal[best_local].item())
                remaining.remove(best_global)

            all_orders.append(selected)
            all_scores.append(scores_at_sel)

        max_len = max(len(o) for o in all_orders)
        padded = torch.full((B, max_len), -1, device=x.device, dtype=torch.long)
        for b, order in enumerate(all_orders):
            padded[b, : len(order)] = torch.tensor(order, device=x.device)

        if return_scores:
            return padded, all_scores
        return padded

    def forward(self, x, y_true=None):
        """
        Forward pass.

        pointwise mode: returns base_scores [B, L].
        soft_greedy mode: returns soft-reranked scores [B, L].
        """
        if self.train_mode == "pointwise" or not self.training:
            return self.base_scores(x)
        elif self.train_mode == "soft_greedy":
            return self._soft_greedy_forward(x)

    def _soft_greedy_forward(self, x):
        """Differentiable greedy via Gumbel-softmax sequential selection."""
        B, L, D = x.shape
        base = self.base_scores(x)

        output_scores = torch.zeros(B, L, device=x.device)
        remaining_prob = torch.ones(B, L, device=x.device)

        for step in range(L):
            diversity_bonus = remaining_prob * 0.1
            marginal = (base + diversity_bonus) * remaining_prob
            marginal = marginal + (1 - remaining_prob) * (-1e9)

            if self.training:
                selection = F.gumbel_softmax(
                    marginal, tau=self.gumbel_tau, hard=False
                )
            else:
                selection = F.softmax(marginal / self.gumbel_tau, dim=-1)

            output_scores = output_scores + selection * (L - step)
            remaining_prob = remaining_prob * (1 - selection)

        return output_scores

    def explain(self, x_single, selected_order):
        """
        Explain the contribution of each component at each selection step.

        Args:
            x_single: [L, D] features for a single query's items
            selected_order: [K] indices in selection order

        Returns:
            list of dicts per step with base_score, diversity_score,
            breakdowns, and total_marginal_gain.
        """
        self.eval()
        explanations = []
        selected_so_far = []

        with torch.no_grad():
            for step, item_idx in enumerate(selected_order):
                item_idx = int(item_idx)

                base_total = 0.0
                base_breakdown = {}
                x_item = x_single[item_idx]

                for j in range(self.num_item_features):
                    feat_val = x_item[j : j + 1].unsqueeze(0)
                    contrib = self.item_towers[j](feat_val).item()
                    base_breakdown[f"feature_{j}"] = contrib
                    base_total += contrib

                cand_idx = torch.tensor([item_idx], device=x_single.device)
                sel_idx = (
                    torch.tensor(
                        selected_so_far, device=x_single.device, dtype=torch.long
                    )
                    if selected_so_far
                    else torch.tensor([], device=x_single.device, dtype=torch.long)
                )

                gw_feats = self.feature_computer.compute(
                    cand_idx, sel_idx, x_single
                )

                div_total = 0.0
                div_breakdown = {}
                for k, (spec, tower) in enumerate(
                    zip(self.groupwise_specs, self.diversity_towers)
                ):
                    input_val = gw_feats[0, k].item()
                    contrib = tower(gw_feats[0, k : k + 1]).item()
                    div_breakdown[spec["name"]] = (input_val, contrib)
                    div_total += contrib

                explanations.append(
                    {
                        "step": step,
                        "item_idx": item_idx,
                        "base_score": base_total,
                        "base_breakdown": base_breakdown,
                        "diversity_score": div_total,
                        "diversity_breakdown": div_breakdown,
                        "total_marginal_gain": base_total + div_total,
                    }
                )

                selected_so_far.append(item_idx)

        return explanations
