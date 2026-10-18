from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from .core import ACTIVE, Move, State


@dataclass
class _Fit:
    deleted: tuple[int, ...]
    updated: tuple[int, ...]
    delta: torch.Tensor
    damage: float
    objective: float
    savings: int


def _layout(state: State):
    names = list(state.params)
    offsets = {}
    start = 0
    for name in names:
        stop = start + state.params[name].numel()
        offsets[name] = (start, stop)
        start = stop
    theta = torch.cat([state.params[name].reshape(-1) for name in names]).double()
    live = torch.cat([state.masks[name].reshape(-1) for name in names])
    return names, offsets, theta, live


def _fisher_design(adapter, state: State, cfg, names):
    X = adapter.data["train"][0]
    if cfg.screen_n is not None and cfg.screen_n < len(X):
        generator = torch.Generator(device=X.device).manual_seed(cfg.seed + 7)
        X = X[torch.randperm(len(X), generator=generator, device=X.device)[:cfg.screen_n]]
    params = {name: state.params[name].detach() for name in names}
    masks = {name: state.masks[name].to(state.params[name].dtype) for name in names}
    struct = state.struct

    def output_single(values, x):
        effective = {name: values[name] * masks[name] for name in names}
        return adapter.logits(effective, struct, x.unsqueeze(0)).reshape(())

    rows = []
    for (xb,) in DataLoader(TensorDataset(X), batch_size=cfg.fisher_batch):
        with torch.enable_grad():
            jacobian = vmap(grad(output_single), in_dims=(None, 0))(params, xb)
        with torch.no_grad():
            effective = {name: params[name] * masks[name] for name in names}
            probability = torch.sigmoid(adapter.logits(effective, struct, xb))
            scale = (probability * (1 - probability) / len(X)).sqrt()
            flat = torch.cat([jacobian[name].reshape(len(xb), -1) for name in names], 1)
            rows.append(scale[:, None] * flat)
    return torch.cat(rows).double()


def _split_indices(indices, offsets):
    values = {}
    if not indices:
        return values
    index = torch.tensor(indices, dtype=torch.long)
    for name, (start, stop) in offsets.items():
        local = index[(index >= start) & (index < stop)] - start
        if local.numel():
            values[name] = local
    return values


def graph_aware_fisher_moves(adapter, state: State, cfg) -> list[Move]:
    if adapter.loss_type != "bernoulli":
        raise ValueError("fisher_general currently supports Bernoulli MLPs")
    names, offsets, theta, live = _layout(state)
    design = _fisher_design(adapter, state, cfg, names)
    diagonal = design.square().sum(0)
    k_current = adapter.k_eff(state)
    closure_cache = {}
    target_cache = {}

    def closure(deleted):
        deleted = tuple(sorted(set(deleted)))
        if deleted not in closure_cache:
            candidate = state.clone()
            for name, local in _split_indices(deleted, offsets).items():
                candidate.masks[name].view(-1)[local] = False
                candidate.params[name].view(-1)[local] = 0
            adapter.canonicalize(candidate)
            survivor = torch.cat([candidate.masks[name].reshape(-1) for name in names])
            closure_cache[deleted] = survivor, k_current - adapter.k_eff(candidate)
        return closure_cache[deleted]

    def fit(deleted, updated=()):
        deleted = tuple(sorted(set(deleted)))
        if not deleted:
            return None
        survivor, savings = closure(deleted)
        updated = tuple(sorted(i for i in set(updated)
                               if survivor[i] and i not in deleted))
        d = torch.tensor(deleted, device=design.device)
        if deleted not in target_cache:
            target_cache[deleted] = design[:, d] @ theta[d]
        target = target_cache[deleted]
        if updated:
            u = torch.tensor(updated, device=design.device)
            z = design[:, u]
            matrix = z.T @ z
            if cfg.fisher_general_mu:
                matrix.diagonal().add_(cfg.fisher_general_mu)
            rhs = z.T @ target
            factor, info = torch.linalg.cholesky_ex(matrix)
            if int(info) == 0:
                delta = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
            else:
                rtol = torch.finfo(torch.float32).eps * max(len(updated), 1)
                delta = torch.linalg.pinv(matrix, rtol=rtol, hermitian=True) @ rhs
            residual = z @ delta - target
        else:
            delta = target.new_empty(0)
            residual = -target
        damage = .5 * residual.square().sum()
        objective = (damage + cfg.fisher_general_lambda_u * len(updated)
                     + .5 * cfg.fisher_general_mu * delta.square().sum()
                     - cfg.fisher_general_lambda_d * savings)
        return _Fit(deleted, updated, delta, float(damage), float(objective), savings)

    deletion_ids = []
    for name in adapter.prunable:
        start, _ = offsets[name]
        local = state.masks[name].reshape(-1).nonzero().flatten()
        deletion_ids.append(local + start)
    deletion_ids = torch.cat(deletion_ids)
    scalar_score = .5 * theta[deletion_ids].square() * diagonal[deletion_ids]
    scalar_order = deletion_ids[torch.argsort(scalar_score)]
    atoms = [tuple(chunk.tolist()) for chunk in scalar_order.split(
        cfg.fisher_general_window)[:cfg.fisher_general_pool] if chunk.numel()]
    atoms.extend((int(i),) for i in scalar_order[:cfg.fisher_general_pool])

    groups = []
    for layer in range(adapter.L - 1):
        name = f"W{layer + 1}"
        start, _ = offsets[name]
        mask = state.masks[name]
        for unit in (state.struct[f"type{layer}"] == ACTIVE).nonzero().flatten().tolist():
            local = (mask[:, unit].nonzero().flatten() * mask.shape[1] + unit)
            if local.numel():
                group = tuple((local + start).tolist())
                candidate = fit(group)
                groups.append((candidate.objective / max(candidate.savings, 1), group))
    groups.sort(key=lambda item: item[0])
    atoms.extend(group for _, group in groups[:cfg.fisher_general_pool])
    atoms = list(dict.fromkeys(atoms))

    initial_fits = [fit(atom) for atom in atoms]
    initial_fits = [item for item in initial_fits
                    if item.savings >= cfg.fisher_general_min_savings]
    starts = sorted(initial_fits, key=lambda item: item.objective)[:cfg.fisher_general_starts]
    found = {(item.deleted, item.updated): item for item in initial_fits}
    for initial in starts:
        current = initial
        for _ in range(cfg.fisher_general_iters):
            alternatives = []
            for atom in atoms:
                deletion = tuple(sorted(set(current.deleted).union(atom)))
                if deletion != current.deleted:
                    alternatives.append(fit(deletion, current.updated))
            restore = sorted(current.deleted,
                             key=lambda i: float(theta[i].square() * diagonal[i]),
                             reverse=True)[:cfg.fisher_general_update_pool]
            alternatives.extend(fit(tuple(i for i in current.deleted if i != restored),
                                    current.updated) for restored in restore
                                if len(current.deleted) > 1)

            d = torch.tensor(current.deleted, device=design.device)
            target = design[:, d] @ theta[d]
            if current.updated:
                u = torch.tensor(current.updated, device=design.device)
                residual = design[:, u] @ current.delta - target
            else:
                residual = -target
            survivor, _ = closure(current.deleted)
            available = survivor.clone()
            available[d] = False
            if current.updated:
                available[u] = False
            available_ids = available.nonzero().flatten()
            if available_ids.numel():
                gradient = (design[:, available_ids].T @ residual).abs()
                scale = (diagonal[available_ids] + cfg.fisher_general_mu).sqrt().clamp_min(1e-15)
                add = available_ids[torch.argsort(gradient / scale, descending=True)
                                    [:cfg.fisher_general_update_pool]]
                alternatives.extend(fit(current.deleted, current.updated + (int(i),))
                                    for i in add)
            alternatives.extend(fit(current.deleted,
                                    tuple(i for i in current.updated if i != removed))
                                for removed in current.updated)
            alternatives = [candidate for candidate in alternatives if candidate is not None]
            if not alternatives:
                break
            best = min(alternatives, key=lambda item: item.objective)
            if best.objective >= current.objective - 1e-14:
                break
            current = best
            found[(current.deleted, current.updated)] = current
        found[(current.deleted, current.updated)] = current

    moves = []
    for candidate in sorted(found.values(), key=lambda item: item.objective):
        if (candidate.objective >= 0
                or candidate.savings < cfg.fisher_general_min_savings):
            continue
        deleted = _split_indices(candidate.deleted, offsets)
        updated_ids = _split_indices(candidate.updated, offsets)
        updates = {}
        position = {index: j for j, index in enumerate(candidate.updated)}
        for name, local in updated_ids.items():
            start, _ = offsets[name]
            updates[name] = (local, torch.stack([
                candidate.delta[position[start + int(i)]] for i in local]))
        layers = [adapter.site_layer[name] for name in deleted if name in adapter.site_layer]
        moves.append(Move(
            "fisher_general", "parameter_graph", min(layers, default=0),
            {"deleted": deleted, "updates": updates,
             "n_deleted": len(candidate.deleted), "n_updates": len(candidate.updated),
             "closure_savings": candidate.savings,
             "fisher_damage": candidate.damage,
             "search_objective": candidate.objective,
             "screen_cost": "fisher"},
            pred_cost=candidate.objective))
    return moves
