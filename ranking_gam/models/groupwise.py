"""
Groupwise (set-dependent) feature computation for diversity-aware ranking.

Each feature is a function: f(item_i, selected_set_S) -> scalar
measuring the marginal contribution of item_i given what is already selected.
"""

import torch
import torch.nn.functional as F


class GroupwiseFeatureComputer:
    """
    Computes groupwise (set-dependent) features for candidate items.

    Built-in feature types:
        'category_novelty':  1 - |same_cat_in_S| / max(|S|, 1)
        'brand_novelty':     1 - |same_brand_in_S| / max(|S|, 1)
        'visual_diversity':  min cosine distance to items in S (or 1.0 if S empty)
        'price_spread':      |price_i - mean_price_S| / (std_price_S + eps)
        'attribute_coverage': fraction of item's attributes not yet covered by S

    Custom features: pass a callable(item_features, selected_features) -> tensor.
    """

    def __init__(self, feature_specs):
        """
        Args:
            feature_specs: list of dicts, each with:
                'name': str identifier
                'type': one of built-in types or 'custom'
                'column': int, column index in item features (for category/brand/price)
                'columns': list of int (for visual_diversity, attribute_coverage)
                'fn': callable (only for type='custom')
                'x_min': float (default 0.0)
                'x_max': float (default 1.0)
        """
        self.specs = feature_specs

    def compute(self, candidates, selected_indices, all_features):
        """
        Compute groupwise features for all candidate items.

        Args:
            candidates: indices of candidate items (not yet selected)
            selected_indices: indices of already-selected items
            all_features: [N, D] tensor of all item features

        Returns:
            [len(candidates), num_groupwise_features] tensor
        """
        device = all_features.device
        n_cand = len(candidates)
        n_specs = len(self.specs)

        result = torch.zeros(n_cand, n_specs, device=device)

        if len(selected_indices) == 0:
            for j, spec in enumerate(self.specs):
                result[:, j] = spec.get("x_max", 1.0)
            return result

        cand_feats = all_features[candidates]
        sel_feats = all_features[selected_indices]

        for j, spec in enumerate(self.specs):
            ftype = spec["type"]
            col = spec.get("column", None)

            if ftype in ("category_novelty", "brand_novelty"):
                cand_vals = cand_feats[:, col]
                sel_vals = sel_feats[:, col]
                matches = (
                    (cand_vals.unsqueeze(1) == sel_vals.unsqueeze(0)).float().sum(dim=1)
                )
                result[:, j] = 1.0 - matches / len(selected_indices)

            elif ftype == "visual_diversity":
                cols = spec["columns"]
                cand_emb = cand_feats[:, cols]
                sel_emb = sel_feats[:, cols]
                cand_norm = F.normalize(cand_emb, dim=-1)
                sel_norm = F.normalize(sel_emb, dim=-1)
                cos_sim = torch.mm(cand_norm, sel_norm.t())
                result[:, j] = 1.0 - cos_sim.max(dim=1)[0]

            elif ftype == "price_spread":
                cand_price = cand_feats[:, col]
                sel_price = sel_feats[:, col]
                mean_p = sel_price.mean()
                if sel_price.numel() <= 1:
                    std_p = torch.tensor(1.0, device=cand_price.device)
                else:
                    std_p = sel_price.std(unbiased=False).clamp(min=1e-6)
                result[:, j] = ((cand_price - mean_p).abs() / std_p).clamp(
                    0, spec.get("x_max", 3.0)
                )

            elif ftype == "attribute_coverage":
                cols = spec["columns"]
                cand_attrs = (cand_feats[:, cols] > 0).float()
                sel_covered = (sel_feats[:, cols] > 0).float().max(dim=0)[0]
                new_attrs = cand_attrs * (1.0 - sel_covered.unsqueeze(0))
                total_attrs = cand_attrs.sum(dim=1).clamp(min=1.0)
                result[:, j] = new_attrs.sum(dim=1) / total_attrs

            elif ftype == "custom":
                result[:, j] = spec["fn"](cand_feats, sel_feats)

        return result
