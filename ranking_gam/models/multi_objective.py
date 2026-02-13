"""
Multi-Objective Ranking GAM -- Pareto-controllable greedy reranking.

Each objective gets its own towers. Objectives can be:
    'pointwise':  scored from item features only (MLP or MonotonePWL)
    'groupwise':  scored from set-dependent features (ConcavePWL)

At serving time, product managers slide the weights and greedy
reranking finds the best list under those tradeoffs -- no retraining.

Theoretical guarantees:
    - ConcavePWL groupwise objectives are submodular
    - Sum of (weighted) submodular + modular = submodular
    - Greedy gives (1-1/e) ~ 63% of optimal
"""

import torch
import torch.nn as nn

from .groupwise import GroupwiseFeatureComputer
from .towers import ConcavePWL, MonotonePWL, PaperTower


class MultiObjectiveRankingGAM(nn.Module):

    def __init__(
        self,
        objectives,
        num_item_features=136,
        hidden_dims=None,
        num_knots=8,
        dropout=0.0,
    ):
        """
        Args:
            objectives: list of objective specs, each a dict:
                'name': str
                'type': 'pointwise' or 'groupwise'
                'weight': float (default scalarization weight)
                For pointwise:
                    'features': list of int column indices
                    'tower': 'mlp' or 'monotone'
                For groupwise:
                    'groupwise_specs': list of GroupwiseFeatureComputer specs
            num_item_features: total item feature columns
            hidden_dims: MLP architecture for unconstrained towers
            num_knots: knots per PWL curve
            dropout: dropout for MLP towers
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]

        self.objectives = objectives
        self.num_item_features = num_item_features
        self.obj_names = [o["name"] for o in objectives]

        default_w = torch.tensor([o.get("weight", 1.0) for o in objectives])
        self.register_buffer("default_weights", default_w)

        self.obj_towers = nn.ModuleDict()
        self.obj_types = {}
        self.obj_features = {}
        self.feature_computers = {}

        for obj in objectives:
            name = obj["name"]
            otype = obj["type"]
            self.obj_types[name] = otype

            if otype == "pointwise":
                features = obj.get("features", list(range(num_item_features)))
                self.obj_features[name] = features
                tower_type = obj.get("tower", "mlp")

                if tower_type == "mlp":
                    towers = nn.ModuleList(
                        [
                            PaperTower(1, hidden_dims=hidden_dims, dropout=dropout)
                            for _ in features
                        ]
                    )
                elif tower_type == "monotone":
                    towers = nn.ModuleList(
                        [
                            MonotonePWL(
                                num_knots=num_knots,
                                x_min=obj.get("x_min", 0.0),
                                x_max=obj.get("x_max", 1.0),
                            )
                            for _ in features
                        ]
                    )
                else:
                    raise ValueError(f"Unknown tower type: {tower_type}")

                self.obj_towers[name] = towers

            elif otype == "groupwise":
                gw_specs = obj["groupwise_specs"]
                towers = nn.ModuleList(
                    [
                        ConcavePWL(
                            num_knots=num_knots,
                            x_min=s.get("x_min", 0.0),
                            x_max=s.get("x_max", 1.0),
                        )
                        for s in gw_specs
                    ]
                )
                self.obj_towers[name] = towers
                self.feature_computers[name] = GroupwiseFeatureComputer(gw_specs)
                self.obj_features[name] = gw_specs

    def objective_scores(
        self, name, x, cand_idx=None, sel_idx=None, item_feats=None
    ):
        """
        Compute scores for a single objective.

        Args:
            name: objective name
            x: [B, L, D] for pointwise, or ignored for groupwise
            cand_idx: candidate indices (groupwise only)
            sel_idx: selected indices (groupwise only)
            item_feats: [L, D] all item features (groupwise only)

        Returns:
            [B, L] for pointwise, [n_cand] for groupwise
        """
        otype = self.obj_types[name]
        towers = self.obj_towers[name]

        if otype == "pointwise":
            features = self.obj_features[name]
            B, L, D = x.shape
            x_flat = x.view(B * L, D)

            sub = []
            for idx, feat_col in enumerate(features):
                sub.append(towers[idx](x_flat[:, feat_col : feat_col + 1]))

            total = torch.stack(sub, dim=-1).sum(dim=-1).squeeze(-1)
            return total.view(B, L)

        elif otype == "groupwise":
            computer = self.feature_computers[name]
            gw_feats = computer.compute(cand_idx, sel_idx, item_feats)

            sub = []
            for k, tower in enumerate(towers):
                sub.append(tower(gw_feats[:, k]))
            return torch.stack(sub, dim=-1).sum(dim=-1)

    def forward(self, x, weights=None):
        """
        Forward pass -- returns weighted sum of pointwise objectives.
        Groupwise objectives only matter during greedy reranking.
        """
        if weights is None:
            weights = self.default_weights

        B, L, D = x.shape
        total = torch.zeros(B, L, device=x.device)

        for i, obj in enumerate(self.objectives):
            if obj["type"] == "pointwise":
                total = total + weights[i] * self.objective_scores(obj["name"], x)

        return total

    def greedy_rerank(self, x, k=None, weights=None, return_scores=False):
        """
        Multi-objective greedy reranking.

        At each step, selects item maximizing:
            sum_obj w_obj * obj_score(item_i, S)
        """
        B, L, D = x.shape
        if k is None:
            k = L

        if weights is None:
            w = self.default_weights
        elif isinstance(weights, dict):
            w = torch.tensor(
                [
                    weights.get(n, self.default_weights[i].item())
                    for i, n in enumerate(self.obj_names)
                ],
                device=x.device,
            )
        else:
            w = weights.to(x.device)

        pw_scores = {}
        for i, obj in enumerate(self.objectives):
            if obj["type"] == "pointwise":
                pw_scores[obj["name"]] = (w[i], self.objective_scores(obj["name"], x))

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

                marginal = torch.zeros(len(remaining), device=x.device)

                for name, (wi, scores) in pw_scores.items():
                    marginal = marginal + wi * scores[b, cand_idx]

                for i, obj in enumerate(self.objectives):
                    if obj["type"] == "groupwise":
                        gw_score = self.objective_scores(
                            obj["name"], x, cand_idx, sel_idx, item_feats
                        )
                        marginal = marginal + w[i] * gw_score

                best_local = marginal.argmax().item()
                best_global = remaining.pop(best_local)
                selected.append(best_global)
                scores_at_sel.append(marginal[best_local].item())

            all_orders.append(selected)
            all_scores.append(scores_at_sel)

        max_len = max(len(o) for o in all_orders)
        padded = torch.full((B, max_len), -1, device=x.device, dtype=torch.long)
        for b, order in enumerate(all_orders):
            padded[b, : len(order)] = torch.tensor(order, device=x.device)

        if return_scores:
            return padded, all_scores
        return padded

    def explain(self, x_single, selected_order, weights=None):
        """
        Full attribution at each greedy step, broken down by objective.

        Returns:
            list of dicts per step with objective-level breakdowns.
        """
        self.eval()
        if weights is None:
            w = self.default_weights
        elif isinstance(weights, dict):
            w = torch.tensor(
                [
                    weights.get(n, self.default_weights[i].item())
                    for i, n in enumerate(self.obj_names)
                ],
                device=x_single.device,
            )
        else:
            w = weights.to(x_single.device)

        explanations = []
        selected = []

        with torch.no_grad():
            x_batch = x_single.unsqueeze(0)

            for step, item_idx in enumerate(selected_order):
                item_idx = int(item_idx)
                cand_idx = torch.tensor(
                    [item_idx], device=x_single.device, dtype=torch.long
                )
                sel_idx = (
                    torch.tensor(
                        selected, device=x_single.device, dtype=torch.long
                    )
                    if selected
                    else torch.tensor([], device=x_single.device, dtype=torch.long)
                )

                step_info = {"step": step, "item_idx": item_idx, "objectives": {}}
                total = 0.0

                for i, obj in enumerate(self.objectives):
                    name = obj["name"]
                    wi = w[i].item()

                    if obj["type"] == "pointwise":
                        score = self.objective_scores(name, x_batch)[
                            0, item_idx
                        ].item()
                        step_info["objectives"][name] = {
                            "score": score,
                            "weight": wi,
                            "weighted": wi * score,
                        }

                    elif obj["type"] == "groupwise":
                        computer = self.feature_computers[name]
                        gw_feats = computer.compute(cand_idx, sel_idx, x_single)
                        towers = self.obj_towers[name]

                        breakdown = {}
                        gw_total = 0.0
                        for k, (spec, tower) in enumerate(
                            zip(obj["groupwise_specs"], towers)
                        ):
                            val = gw_feats[0, k].item()
                            contrib = tower(gw_feats[0, k : k + 1]).item()
                            breakdown[spec["name"]] = (val, contrib)
                            gw_total += contrib

                        step_info["objectives"][name] = {
                            "score": gw_total,
                            "weight": wi,
                            "weighted": wi * gw_total,
                            "breakdown": breakdown,
                        }

                    total += wi * step_info["objectives"][name]["score"]

                step_info["total"] = total
                explanations.append(step_info)
                selected.append(item_idx)

        return explanations
