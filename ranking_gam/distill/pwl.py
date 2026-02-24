"""
Greedy PWL distillation (Algorithm 1 from Zhuang et al.).

Distills neural GAM towers into piecewise-linear functions for
fast serving and full interpretability.

Steps:
  1. Percentile candidate knots (0%, 1%, ..., 100%)
  2. Greedy selection minimizing MSE
  3. Refinement: swap each knot for a better candidate until convergence
  4. y-values via least squares
"""

import numpy as np


def _build_interpolation_matrix(x_data, x_knots):
    """Build matrix A such that PWL(x_data) = A @ y_knots."""
    K = len(x_knots)
    N = len(x_data)
    A = np.zeros((N, K), dtype=np.float64)
    indices = np.searchsorted(x_knots, x_data, side="right") - 1
    for i in range(N):
        k = indices[i]
        if k < 0:
            A[i, 0] = 1.0
        elif k >= K - 1:
            A[i, K - 1] = 1.0
        else:
            span = x_knots[k + 1] - x_knots[k]
            if span < 1e-12:
                A[i, k] = 1.0
            else:
                t = (x_data[i] - x_knots[k]) / span
                A[i, k] = 1.0 - t
                A[i, k + 1] = t
    return A


def _solve_y_least_squares(x_data, y_data, x_knots):
    """Optimal y-knots via least squares (paper Section 5.2)."""
    A = _build_interpolation_matrix(x_data, x_knots)
    y_knots, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
    return y_knots


def greedy_knot_selection(x_data, y_data, num_knots, max_refine=50):
    """
    Algorithm 1 from Zhuang et al. -- faithful to the paper.

    - Candidates from percentile boundaries (0%, 1%, ..., 100%)
    - Greedy selection minimizing MSE
    - Refinement: swap each knot for a better candidate until convergence
    - y-values via least squares
    """
    percentiles = np.linspace(0, 100, 101)
    candidates = np.unique(np.percentile(x_data, percentiles))

    if len(candidates) < num_knots:
        candidates = np.unique(x_data)
        if len(candidates) < num_knots:
            num_knots = len(candidates)

    num_cands = len(candidates)

    def _loss(idx_set):
        idx_sorted = sorted(idx_set)
        xk = candidates[idx_sorted]
        yk = _solve_y_least_squares(x_data, y_data, xk)
        y_pwl = np.interp(x_data, xk, yk)
        return np.mean((y_data - y_pwl) ** 2)

    # Greedy initialization
    S = {0}
    for _k in range(1, num_knots):
        best_idx, best_loss = None, float("inf")
        for i in range(num_cands):
            if i in S:
                continue
            trial = _loss(S | {i})
            if trial < best_loss:
                best_loss, best_idx = trial, i
        if best_idx is not None:
            S.add(best_idx)

    # Refinement: swap each knot for a better candidate
    for _ in range(max_refine):
        improved = False
        for current_idx in sorted(S):
            current_loss = _loss(S)
            S_without = S - {current_idx}
            best_swap, best_swap_loss = None, current_loss
            for i in range(num_cands):
                if i in S_without:
                    continue
                trial = _loss(S_without | {i})
                if trial < best_swap_loss:
                    best_swap_loss, best_swap = trial, i
            if best_swap is not None and best_swap != current_idx:
                S = S_without | {best_swap}
                improved = True
        if not improved:
            break

    final = sorted(S)
    x_knots = candidates[final]
    y_knots = _solve_y_least_squares(x_data, y_data, x_knots)
    return x_knots, y_knots


def _check_distillable(model):
    """Validate that a model supports distillation to PWL.

    Supported models: GAM_Paper, GA2M_Paper, ContextGAM, SubmodularRankingGAM.
    Not supported: MultiObjectiveRankingGAM (multiple objective towers, no single
    set of main effects), ContextPresentGA2M (use distill_context_model instead).

    Raises:
        TypeError: if the model lacks required attributes.
    """
    model_cls = type(model).__name__
    missing = []
    for attr in ("num_features", "get_main_effect", "global_bias"):
        if not hasattr(model, attr):
            missing.append(attr)
    if missing:
        raise TypeError(
            f"{model_cls} does not support distill_to_pwl "
            f"(missing: {', '.join(missing)}). "
            f"Supported models: GAM_Paper, GA2M_Paper, ContextGAM, "
            f"SubmodularRankingGAM."
        )


def distill_to_pwl(model, num_knots=5, grid_size=30, train_X=None):
    """
    Distill GAM/GA2M/ContextGAM to PWL with Algorithm 1.

    Paper Section 5.1: "a small K (e.g. around 3 to 5) is usually sufficient"

    Supported models:
        - GAM_Paper: full distillation
        - GA2M_Paper: main effects + interaction grids
        - ContextGAM: tower shapes only (context weights are dropped;
          use the neural context_net at serving time alongside PWL towers)
        - SubmodularRankingGAM: item towers only (diversity towers are
          already piecewise-linear)

    Not supported:
        - MultiObjectiveRankingGAM: multiple objective tower sets, no
          single set of main effects to distill
        - ContextPresentGA2M: use distill_context_model() instead

    Args:
        model: trained model with get_main_effect(), num_features, global_bias
        num_knots: knots per feature
        grid_size: grid resolution for 2D interactions
        train_X: [B, L, D] training data for percentile sampling

    Returns:
        dict with 'main_effects', 'interactions', 'bias'

    Raises:
        TypeError: if the model does not support distillation
    """
    _check_distillable(model)

    model_cls = type(model).__name__
    if model_cls == "ContextGAM":
        import warnings
        warnings.warn(
            "Distilling ContextGAM: tower shapes are distilled to PWL but "
            "context weights (w_j(x)) are dropped. The distilled model scores "
            "as sum(f_j(x_j)) instead of sum(w_j(x)*f_j(x_j)). For full "
            "ContextGAM serving, keep the context_net (~14K params) alongside "
            "the PWL towers.",
            stacklevel=2,
        )

    num_features = model.num_features

    main_pwl = []
    total_mse = 0

    for i in range(num_features):
        if train_X is not None:
            X_flat = train_X.reshape(-1, train_X.shape[-1])
            feat_vals = X_flat[:, i]
            x_samples = np.unique(
                np.percentile(feat_vals, np.linspace(0, 100, 500))
            )
            x_samples = x_samples.astype(np.float32)
        else:
            x_samples = np.linspace(-4, 4, 500).astype(np.float32)

        y_samples = model.get_main_effect(i, x_samples)
        x_knots, y_knots = greedy_knot_selection(x_samples, y_samples, num_knots)

        y_pwl = np.interp(x_samples, x_knots, y_knots)
        mse = np.mean((y_samples - y_pwl) ** 2)
        total_mse += mse

        main_pwl.append(
            {"feature": i, "x": x_knots.tolist(), "y": y_knots.tolist(), "mse": float(mse)}
        )

    print(f"  Main effects distillation: avg MSE = {total_mse / num_features:.8f}")

    interaction_pwl = []
    if hasattr(model, "interaction_pairs") and model.num_interactions > 0:
        X_flat = train_X.reshape(-1, train_X.shape[-1]) if train_X is not None else None

        for i in range(model.num_interactions):
            f1, f2 = model.interaction_pairs[i]

            if X_flat is not None:
                x1_grid = np.unique(
                    np.percentile(X_flat[:, f1], np.linspace(0, 100, grid_size))
                ).astype(np.float32)
                x2_grid = np.unique(
                    np.percentile(X_flat[:, f2], np.linspace(0, 100, grid_size))
                ).astype(np.float32)
            else:
                x1_grid = np.linspace(-4, 4, grid_size).astype(np.float32)
                x2_grid = np.linspace(-4, 4, grid_size).astype(np.float32)

            xx, yy = np.meshgrid(x1_grid, x2_grid, indexing="ij")
            z = model.get_interaction_effect(i, xx.flatten(), yy.flatten())
            z = z.reshape(len(x1_grid), len(x2_grid))
            interaction_pwl.append(
                {
                    "features": (f1, f2),
                    "x1_grid": x1_grid.tolist(),
                    "x2_grid": x2_grid.tolist(),
                    "z": z.tolist(),
                }
            )

    bias = float(model.global_bias.detach().cpu().numpy().item())

    # Extract diversity towers for SubmodularRankingGAM (already PWL, no distillation)
    diversity_towers = []
    groupwise_specs_out = []
    if hasattr(model, "diversity_towers") and hasattr(model, "groupwise_specs"):
        import torch

        with torch.no_grad():
            for k, (spec, tower) in enumerate(
                zip(model.groupwise_specs, model.diversity_towers)
            ):
                slopes = tower.get_slopes().cpu().numpy().tolist()
                knot_edges = tower.knot_edges.cpu().numpy().tolist()
                knot_widths = tower.knot_widths.cpu().numpy().tolist()
                diversity_towers.append(
                    {
                        "name": spec.get("name", f"diversity_{k}"),
                        "x_min": float(tower.x_min),
                        "x_max": float(tower.x_max),
                        "intercept": float(tower.intercept.cpu()),
                        "knot_edges": knot_edges,
                        "knot_widths": knot_widths,
                        "slopes": slopes,
                    }
                )
                # Serialize spec (omit callable 'fn' for custom types)
                spec_out = {
                    k_: v
                    for k_, v in spec.items()
                    if k_ != "fn" and not callable(v)
                }
                groupwise_specs_out.append(spec_out)
        print(f"  Diversity towers: {len(diversity_towers)} exported (already PWL)")

    return {
        "main_effects": main_pwl,
        "interactions": interaction_pwl,
        "bias": bias,
        "context_weights": [],
        "has_context": False,
        "diversity_towers": diversity_towers,
        "groupwise_specs": groupwise_specs_out,
    }


def _distill_context_weights(model, train_q, num_knots_context=5):
    """Distill context weight functions."""
    if not hasattr(model, "context_feature_specs"):
        return []

    n = model.num_item_features
    context_pwl = []

    for k, spec in enumerate(model.context_feature_specs):
        q_vals = train_q[k]

        if spec["type"] == "categorical":
            unique_cats = np.unique(q_vals).astype(int)
            weight_table = {}
            for cat_val in unique_cats:
                weight_vec = []
                for j in range(n):
                    w = model.get_context_weight_for_feature(
                        k, j, np.array([cat_val])
                    )
                    weight_vec.append(float(w[0]))
                weight_table[int(cat_val)] = weight_vec
            context_pwl.append(
                {"context_feature": k, "type": "categorical", "weight_table": weight_table}
            )
        else:
            q_samples = np.unique(
                np.percentile(q_vals, np.linspace(0, 100, 500))
            ).astype(np.float32)
            feature_pwls = []
            for j in range(n):
                w_values = model.get_context_weight_for_feature(k, j, q_samples)
                q_knots, w_knots = greedy_knot_selection(
                    q_samples, w_values, num_knots_context
                )
                feature_pwls.append(
                    {"item_feature": j, "q_knots": q_knots.tolist(), "w_knots": w_knots.tolist()}
                )
            context_pwl.append(
                {"context_feature": k, "type": "numerical", "feature_pwls": feature_pwls}
            )

    return context_pwl


def distill_context_model(
    model, train_X, train_q, num_knots=5, grid_size=30, num_knots_context=5,
):
    """
    Full distillation for context-present GA2M.

    Distills: main effects + interactions + context weight functions.
    """
    pwl = distill_to_pwl(model, num_knots=num_knots, grid_size=grid_size, train_X=train_X)
    pwl["context_weights"] = _distill_context_weights(model, train_q, num_knots_context)
    pwl["has_context"] = True
    print(f"  Distilled {len(pwl['context_weights'])} context weight functions")
    return pwl


def pwl_predict(pwl, X, q=None):
    """
    Predict using distilled PWL model. Supports context weights.

    Args:
        pwl: distilled model dict from distill_to_pwl or distill_context_model
        X: [B, L, D] numpy feature array
        q: list of context arrays (optional)

    Returns:
        [B, L] numpy score array
    """
    from scipy.interpolate import RegularGridInterpolator

    batch_size, list_size, num_features = X.shape
    scores = np.full((batch_size, list_size), pwl["bias"], dtype=np.float32)

    has_context = pwl.get("has_context", False) and q is not None
    if has_context:
        weights = np.zeros((batch_size, num_features), dtype=np.float32)
        for ctx in pwl["context_weights"]:
            k = ctx["context_feature"]
            if ctx["type"] == "categorical":
                table = ctx["weight_table"]
                for b in range(batch_size):
                    cat_val = int(q[k][b])
                    if cat_val in table:
                        weights[b] += np.array(table[cat_val], dtype=np.float32)
            else:
                for j_info in ctx["feature_pwls"]:
                    j = j_info["item_feature"]
                    q_knots = np.array(j_info["q_knots"])
                    w_knots = np.array(j_info["w_knots"])
                    weights[:, j] += np.interp(q[k], q_knots, w_knots)

    for p in pwl["main_effects"]:
        j = p["feature"]
        feat_vals = X[:, :, j].flatten()
        contrib = np.interp(feat_vals, p["x"], p["y"]).reshape(batch_size, list_size)
        if has_context:
            contrib = contrib * weights[:, j : j + 1]
        scores += contrib

    for p in pwl["interactions"]:
        f1, f2 = p["features"]
        interp = RegularGridInterpolator(
            (np.array(p["x1_grid"]), np.array(p["x2_grid"])),
            np.array(p["z"]),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        x1 = np.clip(X[:, :, f1].flatten(), p["x1_grid"][0], p["x1_grid"][-1])
        x2 = np.clip(X[:, :, f2].flatten(), p["x2_grid"][0], p["x2_grid"][-1])
        scores += interp(np.stack([x1, x2], axis=-1)).reshape(batch_size, list_size)

    return scores


def save_pwl_json(pwl, path):
    """Save distilled PWL model to JSON for Java/serving inference.

    Args:
        pwl: distilled model dict from distill_to_pwl or distill_context_model
        path: output file path (e.g. "model.json")
    """
    import json

    # Convert interaction tuple keys to lists for JSON serialization
    out = {
        "bias": pwl["bias"],
        "has_context": pwl.get("has_context", False),
        "main_effects": pwl["main_effects"],
        "interactions": [
            {
                "features": list(p["features"]),
                "x1_grid": p["x1_grid"],
                "x2_grid": p["x2_grid"],
                "z": p["z"],
            }
            for p in pwl.get("interactions", [])
        ],
        "context_weights": pwl.get("context_weights", []),
        "diversity_towers": pwl.get("diversity_towers", []),
        "groupwise_specs": pwl.get("groupwise_specs", []),
    }
    with open(path, "w") as f:
        json.dump(out, f)


def load_pwl_json(path):
    """Load distilled PWL model from JSON.

    Args:
        path: JSON file path saved by save_pwl_json

    Returns:
        dict compatible with pwl_predict
    """
    import json

    with open(path) as f:
        pwl = json.load(f)
    # Convert interaction feature lists back to tuples for pwl_predict compat
    for p in pwl.get("interactions", []):
        p["features"] = tuple(p["features"])
    return pwl


def evaluate_pwl(pwl, X, y, q=None, k=10):
    """Evaluate PWL NDCG. Supports context."""
    pred = pwl_predict(pwl, X, q)
    ndcg_list = []
    for p, t in zip(pred, y):
        valid = t >= 0
        if not valid.any():
            continue
        pv, tv = p[valid], t[valid]
        idx = np.argsort(-pv)[:k]
        disc = np.log2(np.arange(len(idx)) + 2)
        dcg = np.sum((2 ** tv[idx] - 1) / disc)
        iidx = np.argsort(-tv)[:k]
        idcg = np.sum((2 ** tv[iidx] - 1) / disc[: len(iidx)])
        if idcg > 0:
            ndcg_list.append(dcg / idcg)
    return np.mean(ndcg_list) if ndcg_list else 0.0
