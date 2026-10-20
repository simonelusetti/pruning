from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import ACTIVE, Move, State
from .fisher_general import _fisher_design, _layout, _split_indices


@dataclass
class _Fit:
    deleted: tuple[int, ...]
    active: tuple[int, ...]
    delta: torch.Tensor
    damage: float
    objective: float
    savings: int


def _solve_psd(matrix, rhs):
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info) == 0:
        return torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
    rtol = torch.finfo(torch.float32).eps * max(matrix.shape[0], 1)
    return torch.linalg.pinv(matrix, rtol=rtol, hermitian=True) @ rhs


def _ridge_compensation(design, target, removed, mu):
    keep = torch.ones(design.shape[1], dtype=torch.bool, device=design.device)
    keep[removed] = False
    x = design[:, keep]
    if not x.numel():
        return design.new_zeros(design.shape[1])
    if mu:
        if x.shape[1] > x.shape[0]:
            matrix = x @ x.T
            matrix.diagonal().add_(mu)
            value = x.T @ _solve_psd(matrix, target)
        else:
            matrix = x.T @ x
            matrix.diagonal().add_(mu)
            value = _solve_psd(matrix, x.T @ target)
    else:
        value = torch.linalg.lstsq(x, target).solution
    result = design.new_zeros(design.shape[1])
    result[keep] = value
    return result


def learned_basis_fisher_moves(adapter, state: State, cfg) -> list[Move]:
    if adapter.loss_type != "bernoulli":
        raise ValueError("fisherT currently supports Bernoulli MLPs")
    names, offsets, theta, live = _layout(state)
    design = _fisher_design(adapter, state, cfg, names)
    diagonal = design.square().sum(0)
    k_current = adapter.k_eff(state)
    closure_cache = {}
    target_cache = {}

    spaces = []
    for layer in range(adapter.L):
        ids = []
        for name in (f"W{layer}", f"b{layer}"):
            start, stop = offsets[name]
            local = live[start:stop].nonzero().flatten() + start
            ids.append(local)
        spaces.append(torch.cat(ids))

    def closure(deleted):
        deleted = tuple(sorted(set(deleted)))
        if deleted not in closure_cache:
            candidate = state.clone()
            for name, local in _split_indices(deleted, offsets).items():
                local = local.to(candidate.params[name].device)
                candidate.masks[name].view(-1)[local] = False
                candidate.params[name].view(-1)[local] = 0
            adapter.canonicalize(candidate)
            survivor = torch.cat([candidate.masks[name].reshape(-1) for name in names])
            closure_cache[deleted] = survivor, k_current - adapter.k_eff(candidate)
        return closure_cache[deleted]

    def target(deleted):
        deleted = tuple(sorted(set(deleted)))
        if deleted not in target_cache:
            ids = torch.tensor(deleted, device=design.device)
            target_cache[deleted] = design[:, ids] @ theta[ids]
        return target_cache[deleted]

    scalar_singles = []
    scalar_chunks = []
    for layer, name in enumerate(f"W{i}" for i in range(adapter.L)):
        start, _ = offsets[name]
        ids = state.masks[name].reshape(-1).nonzero().flatten() + start
        order = ids[torch.argsort(.5 * theta[ids].square() * diagonal[ids])]
        scalar_singles.extend(
            (float(.5 * theta[i].square() * diagonal[i]), (int(i),), layer)
            for i in order[:cfg.fisher_general_pool])
        for chunk in order.split(cfg.fisher_general_window):
            if chunk.numel():
                score = float((.5 * theta[chunk].square() * diagonal[chunk]).sum())
                scalar_chunks.append((score, tuple(chunk.tolist()), layer))

    structural = []
    for layer in range(adapter.L - 1):
        name = f"W{layer + 1}"
        start, _ = offsets[name]
        mask = state.masks[name]
        for unit in (state.struct[f"type{layer}"] == ACTIVE).nonzero().flatten().tolist():
            local = mask[:, unit].nonzero().flatten() * mask.shape[1] + unit
            if local.numel():
                deleted = tuple((local + start).tolist())
                _, savings = closure(deleted)
                score = float(.5 * target(deleted).square().sum()) / max(savings, 1)
                structural.append((score, deleted, layer + 1))

    pool = cfg.fisher_general_pool
    representatives = (sorted(scalar_singles)[:pool]
                       + sorted(scalar_chunks)[:pool]
                       + sorted(structural)[:pool])
    unique = {}
    for score, deleted, space in representatives:
        unique.setdefault(deleted, (score, deleted, space))
    representatives = list(unique.values())

    basis_blocks = []
    basis_layers = []
    for layer, space_ids in enumerate(spaces):
        candidates = [deleted for _, deleted, space in representatives if space == layer]
        if not candidates or not space_ids.numel():
            continue
        position = {int(index): local for local, index in enumerate(space_ids.tolist())}
        local_design = design[:, space_ids]
        repairs = []
        for deleted in candidates:
            removed = torch.tensor([position[i] for i in deleted], device=design.device)
            compensation = _ridge_compensation(
                local_design, target(deleted), removed, cfg.fisher_general_mu)
            norm = compensation.norm()
            if float(norm) > torch.finfo(compensation.dtype).eps:
                repairs.append(compensation / norm)
        if not repairs:
            continue
        repair_matrix = torch.stack(repairs, 1)
        vectors, singular, _ = torch.linalg.svd(repair_matrix, full_matrices=False)
        rank = min(cfg.fisher_t_rank, int((singular > singular[0] * 1e-10).sum()))
        if rank:
            block = theta.new_zeros(theta.numel(), rank)
            block[space_ids] = vectors[:, :rank]
            basis_blocks.append(block)
            basis_layers.extend([layer] * rank)

    if not basis_blocks:
        return []
    basis = torch.cat(basis_blocks, 1)
    basis_rank = basis.shape[1]

    fit_cache = {}

    def optimize(deleted):
        deleted = tuple(sorted(set(deleted)))
        if not deleted:
            return None
        if deleted in fit_cache:
            return fit_cache[deleted]
        survivor, closure_savings = closure(deleted)
        restricted = basis * survivor[:, None].to(basis.dtype)
        usable = (torch.linalg.vector_norm(restricted, dim=0)
                  > torch.finfo(restricted.dtype).eps)
        available = usable.nonzero().flatten().tolist()
        projected = design @ restricted
        wanted = target(deleted)

        def fit(active):
            active = tuple(sorted(set(active)))
            if active:
                ids = torch.tensor(active, device=design.device)
                t = restricted[:, ids]
                z = projected[:, ids]
                matrix = z.T @ z
                if cfg.fisher_general_mu:
                    matrix += cfg.fisher_general_mu * (t.T @ t)
                alpha = _solve_psd(matrix, z.T @ wanted)
                delta = t @ alpha
                residual = z @ alpha - wanted
                regularizer = .5 * cfg.fisher_general_mu * delta.square().sum()
            else:
                delta = theta.new_zeros(theta.numel())
                residual = -wanted
                regularizer = theta.new_zeros(())
            damage = .5 * residual.square().sum()
            # T is zero outside canonical survivors and never changes masks, so
            # every support S has exactly the deletion closure's parameter count.
            savings = closure_savings
            objective = (damage + regularizer
                         + cfg.fisher_t_lambda_m * len(active)
                         - cfg.fisher_general_lambda_d * savings)
            return _Fit(deleted, active, delta, float(damage),
                        float(objective), savings)

        current = fit(())
        for _ in range(2 * len(available) + 1):
            alternatives = [fit(current.active + (index,))
                            for index in available if index not in current.active]
            alternatives += [fit(tuple(index for index in current.active if index != removed))
                             for removed in current.active]
            if not alternatives:
                break
            best = min(alternatives, key=lambda item: item.objective)
            if best.objective >= current.objective - 1e-14:
                break
            current = best
        fit_cache[deleted] = current
        return current

    atoms = [deleted for _, deleted, _ in representatives]
    initial = [optimize(atom) for atom in atoms]
    initial = [item for item in initial
               if item.savings >= cfg.fisher_general_min_savings]
    starts = sorted(initial, key=lambda item: item.objective)[:cfg.fisher_general_starts]
    found = {item.deleted: item for item in initial}
    for start in starts:
        current = start
        for _ in range(cfg.fisher_general_iters):
            alternatives = []
            for atom in atoms:
                deleted = tuple(sorted(set(current.deleted).union(atom)))
                if deleted != current.deleted:
                    alternatives.append(optimize(deleted))
            restore = sorted(current.deleted,
                             key=lambda i: float(theta[i].square() * diagonal[i]),
                             reverse=True)[:cfg.fisher_general_update_pool]
            alternatives.extend(optimize(tuple(i for i in current.deleted if i != restored))
                                for restored in restore if len(current.deleted) > 1)
            alternatives = [item for item in alternatives if item is not None]
            if not alternatives:
                break
            best = min(alternatives, key=lambda item: item.objective)
            if best.objective >= current.objective - 1e-14:
                break
            current = best
            found[current.deleted] = current

    moves = []
    for candidate in sorted(found.values(), key=lambda item: item.objective):
        if (candidate.objective >= 0
                or candidate.savings < cfg.fisher_general_min_savings):
            continue
        deleted = _split_indices(candidate.deleted, offsets)
        updates = {}
        for name, (start, stop) in offsets.items():
            value = candidate.delta[start:stop]
            local = (value != 0).nonzero().flatten()
            if local.numel():
                updates[name] = (local, value[local])
        layers = [adapter.site_layer[name] for name in deleted if name in adapter.site_layer]
        active_layers = sorted(set(basis_layers[index] for index in candidate.active))
        moves.append(Move(
            "fisher_t", "parameter_graph", min(layers, default=0),
            {"deleted": deleted, "updates": updates,
             "n_deleted": len(candidate.deleted),
             "n_updates": sum(len(local) for local, _ in updates.values()),
             "closure_savings": candidate.savings,
             "fisher_damage": candidate.damage,
             "search_objective": candidate.objective,
             "basis_rank": basis_rank,
             "active_directions": len(candidate.active),
             "active_basis_layers": active_layers,
             "screen_cost": "fisher"},
            pred_cost=candidate.objective))
    return moves
