"""
Training loops for GAM, SubmodularRankingGAM, and MultiObjectiveRankingGAM.

Includes:
  - train_model: standard training with early stopping
  - train_diversity_towers: phase-2 step-wise diversity tower training
  - train_multi_objective: joint multi-objective step-wise training
  - get_base_ranking / get_greedy_ranking: inference helpers
"""

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import compute_ndcg


def train_model(
    model, train_loader, val_loader, loss_fn, epochs=30, lr=0.001,
    patience=7, grad_clip=1.0, device=None,
    lr_schedule="constant", l1_output_reg=0.0,
    transform_lr_mult=0.1, warmup_epochs=1,
    weight_decay=0.0, eval_k=10,
):
    """
    Standard training loop with early stopping on NDCG.

    Args:
        model: any ranking model (GAM, GA2M, SubmodularRankingGAM, etc.)
        train_loader: DataLoader of (X, y) batches
        val_loader: DataLoader of (X, y) batches
        loss_fn: ranking loss module
        epochs: max training epochs
        lr: learning rate
        patience: early stopping patience
        grad_clip: max gradient norm (0 to disable)
        device: torch device (auto-detected if None)
        lr_schedule: "constant" or "cosine" (CosineAnnealingLR over epochs)
        l1_output_reg: L1 penalty weight on predicted scores (0 to disable)
        transform_lr_mult: LR multiplier for feature transforms (default 0.1x)
        warmup_epochs: linear LR warmup epochs (default 1, 0 to disable)
        weight_decay: L2 weight decay for AdamW (default 0, uses Adam)
        eval_k: NDCG cutoff for validation (default 10)

    Returns:
        best validation NDCG@eval_k
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Separate param groups: transforms get a lower LR to fine-tune gently
    transform_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "feature_transform" in name:
            transform_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": lr}]
    if transform_params:
        param_groups.append({"params": transform_params, "lr": lr * transform_lr_mult})

    OptimClass = torch.optim.AdamW if weight_decay > 0 else torch.optim.Adam
    optimizer = OptimClass(param_groups, weight_decay=weight_decay)

    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

    best_ndcg = 0
    best_state = None
    patience_counter = 0

    # Linear warmup scheduler (ramps LR from 0 to target over warmup_epochs)
    warmup_scheduler = None
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs,
        )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            if l1_output_reg > 0:
                loss = loss + l1_output_reg * pred.abs().mean()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Step schedulers: warmup first, then cosine
        if warmup_scheduler is not None and epoch < warmup_epochs:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_ndcg.append(compute_ndcg(pred, y, k=eval_k))

        val_ndcg = np.mean(val_ndcg)
        print(f"Epoch {epoch + 1:2d}: loss={train_loss:.4f}, val_ndcg@{eval_k}={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    return best_ndcg


def train_diversity_towers(
    model, X_aug, y, epochs=10, lr=0.003, k=10, queries_per_epoch=300,
):
    """
    Phase 2: Train diversity towers via step-wise ranking loss.

    Freezes base towers, then at each greedy step computes ListNet loss
    between marginal gains and relevance labels.

    Args:
        model: SubmodularRankingGAM with trained base towers
        X_aug: [B, L, D_aug] numpy array of features (with groupwise columns)
        y: [B, L] numpy array of relevance labels
        epochs: training epochs
        lr: learning rate for diversity towers
        k: number of greedy steps per query
        queries_per_epoch: subsample for speed

    Returns:
        model with trained diversity towers
    """
    device = next(model.parameters()).device

    for p in model.item_towers.parameters():
        p.requires_grad_(False)

    div_params = list(model.diversity_towers.parameters())
    optimizer = torch.optim.Adam(div_params, lr=lr)

    print(f"\n  Phase 2: Training diversity towers ({len(div_params)} param groups)")

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_queries = 0

        indices = np.random.permutation(len(X_aug))[:queries_per_epoch]

        for qi in indices:
            x_q = torch.from_numpy(X_aug[qi]).float().to(device)
            y_q = torch.from_numpy(y[qi]).float().to(device)

            valid_mask = y_q >= 0
            if valid_mask.sum() < 5:
                continue

            with torch.no_grad():
                base = model.base_scores(x_q.unsqueeze(0)).squeeze(0)

            selected = []
            remaining = [i for i in range(len(y_q)) if valid_mask[i]]
            query_loss = torch.tensor(0.0, device=device)

            for step in range(min(k, len(remaining))):
                if len(remaining) < 2:
                    break

                cand_idx = torch.tensor(remaining, device=device, dtype=torch.long)
                sel_idx = (
                    torch.tensor(selected, device=device, dtype=torch.long)
                    if selected
                    else torch.tensor([], device=device, dtype=torch.long)
                )

                gw_feats = model.feature_computer.compute(cand_idx, sel_idx, x_q)
                div_scores = model.diversity_scores(gw_feats)
                marginal = base[cand_idx] + div_scores
                rel_labels = y_q[cand_idx]

                p_true = F.softmax(rel_labels, dim=0)
                p_pred = F.softmax(marginal, dim=0)
                step_loss = -(p_true * torch.log(p_pred + 1e-10)).sum()

                position_weight = 1.0 / np.log2(step + 2)
                query_loss = query_loss + step_loss * position_weight

                with torch.no_grad():
                    best_local = marginal.argmax().item()
                    best_global = remaining[best_local]
                    selected.append(best_global)
                    remaining.remove(best_global)

            if query_loss.requires_grad:
                optimizer.zero_grad()
                query_loss.backward()
                torch.nn.utils.clip_grad_norm_(div_params, 1.0)
                optimizer.step()
                total_loss += query_loss.item()
                n_queries += 1

        avg_loss = total_loss / max(n_queries, 1)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            model.eval()
            sample_ndcg = _quick_greedy_eval(model, X_aug, y, n_queries=100, k=k)
            model.train()
            print(
                f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, "
                f"sample_ndcg@{k}={sample_ndcg:.4f}"
            )
        else:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    for p in model.item_towers.parameters():
        p.requires_grad_(True)

    model.eval()
    return model


def train_multi_objective(
    model, X_aug, y, epochs=10, lr=0.003, k=10, queries_per_epoch=300,
):
    """
    Train all towers of a MultiObjectiveRankingGAM via step-wise ranking loss.

    At each greedy step, computes the weighted multi-objective marginal gain
    and trains all towers to make greedy selection agree with relevance ordering.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Training {n_params} parameters, epochs={epochs}, lr={lr}")

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_queries = 0

        indices = np.random.permutation(len(X_aug))[:queries_per_epoch]

        for qi in indices:
            x_q = torch.from_numpy(X_aug[qi]).float().to(device)
            y_q = torch.from_numpy(y[qi]).float().to(device)

            valid_mask = y_q >= 0
            if valid_mask.sum() < 5:
                continue

            x_batch = x_q.unsqueeze(0)

            selected = []
            remaining = [i for i in range(len(y_q)) if valid_mask[i]]
            query_loss = torch.tensor(0.0, device=device)

            for step in range(min(k, len(remaining))):
                if len(remaining) < 2:
                    break

                cand_idx = torch.tensor(remaining, device=device, dtype=torch.long)
                sel_idx = (
                    torch.tensor(selected, device=device, dtype=torch.long)
                    if selected
                    else torch.tensor([], device=device, dtype=torch.long)
                )

                marginal = torch.zeros(len(remaining), device=device)

                for i, obj in enumerate(model.objectives):
                    name = obj["name"]
                    wi = model.default_weights[i]

                    if obj["type"] == "pointwise":
                        scores = model.objective_scores(name, x_batch)
                        marginal = marginal + wi * scores[0, cand_idx]
                    elif obj["type"] == "groupwise":
                        gw_scores = model.objective_scores(
                            name, x_batch, cand_idx, sel_idx, x_q
                        )
                        marginal = marginal + wi * gw_scores

                rel_labels = y_q[cand_idx]
                p_true = F.softmax(rel_labels, dim=0)
                p_pred = F.softmax(marginal, dim=0)
                step_loss = -(p_true * torch.log(p_pred + 1e-10)).sum()

                position_weight = 1.0 / np.log2(step + 2)
                query_loss = query_loss + step_loss * position_weight

                with torch.no_grad():
                    best_local = marginal.argmax().item()
                    best_global = remaining[best_local]
                    selected.append(best_global)
                    remaining.remove(best_global)

            if query_loss.requires_grad:
                optimizer.zero_grad()
                query_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += query_loss.item()
                n_queries += 1

        avg_loss = total_loss / max(n_queries, 1)
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    model.eval()
    return model


def _quick_greedy_eval(model, X_aug, y, n_queries=100, k=10, device=None):
    """Quick NDCG eval of greedy reranking on a sample of queries."""
    if device is None:
        device = next(model.parameters()).device

    ndcg_list = []
    indices = np.random.RandomState(42).permutation(len(X_aug))[:n_queries]

    with torch.no_grad():
        for qi in indices:
            x_q = torch.from_numpy(X_aug[qi : qi + 1]).float().to(device)
            y_q = y[qi]

            valid = y_q >= 0
            if valid.sum() < 2:
                continue

            order = model.greedy_rerank(x_q, k=k)
            order_list = [o for o in order[0].tolist() if o >= 0]

            if len(order_list) > 0:
                reranked_labels = [
                    float(y_q[o]) if o < len(y_q) and y_q[o] >= 0 else 0.0
                    for o in order_list
                ]
                disc = np.log2(np.arange(len(reranked_labels)) + 2)
                dcg = np.sum((2 ** np.array(reranked_labels) - 1) / disc)

                ideal = np.sort(y_q[valid])[::-1][:k]
                idcg = np.sum((2**ideal - 1) / np.log2(np.arange(len(ideal)) + 2))
                if idcg > 0:
                    ndcg_list.append(dcg / idcg)

    return np.mean(ndcg_list) if ndcg_list else 0.0


def get_base_ranking(model, X_aug, y, k=10, device=None):
    """Get top-k by base scores only (no diversity reranking)."""
    if device is None:
        device = next(model.parameters()).device

    all_orders = []
    with torch.no_grad():
        for qi in range(len(X_aug)):
            x_q = torch.from_numpy(X_aug[qi : qi + 1]).float().to(device)
            base = model.base_scores(x_q).squeeze(0).cpu().numpy()

            valid = y[qi] >= 0
            base[~valid] = -1e9
            order = np.argsort(-base)[:k].tolist()
            all_orders.append(order)

    return all_orders


def get_greedy_ranking(model, X_aug, k=10, device=None):
    """Get top-k via greedy submodular reranking."""
    if device is None:
        device = next(model.parameters()).device

    all_orders = []
    with torch.no_grad():
        for qi in range(len(X_aug)):
            x_q = torch.from_numpy(X_aug[qi : qi + 1]).float().to(device)
            order = model.greedy_rerank(x_q, k=k)
            order_list = [o for o in order[0].tolist() if o >= 0]
            all_orders.append(order_list)

    return all_orders
