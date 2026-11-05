from __future__ import annotations

import torch
from torch.func import grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from .core import ACTIVE, REMOVED, Move, State
from .fisher import resolve_strategy, validate_strategy


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
    index = torch.tensor(indices, dtype=torch.long)
    for name, (start, stop) in offsets.items():
        local = index[(index >= start) & (index < stop)] - start
        if local.numel():
            values[name] = local
    return values


def _closure_details(adapter, state, offsets, deleted):
    candidate = state.clone()
    for name, local in _split_indices(deleted, offsets).items():
        candidate.masks[name].view(-1)[local] = False
        candidate.params[name].view(-1)[local] = 0
    adapter.canonicalize(candidate)
    survivor = torch.cat([candidate.masks[name].reshape(-1) for name in offsets])
    return adapter.k_eff(state) - adapter.k_eff(candidate), survivor


def _closure(adapter, state, offsets, deleted):
    return _closure_details(adapter, state, offsets, deleted)[0]


def _candidate_pool(adapter, state, cfg, names, offsets, theta, live, design):
    diagonal = design.square().sum(0)
    eligible = []
    for name in adapter.prunable:
        start, _ = offsets[name]
        eligible.append(state.masks[name].reshape(-1).nonzero().flatten() + start)
    eligible = torch.cat(eligible)
    order = eligible[torch.argsort(.5 * theta[eligible].square() * diagonal[eligible])]

    groups = []
    for layer in range(adapter.L - 1):
        name = f"W{layer + 1}"
        start, _ = offsets[name]
        mask = state.masks[name]
        for unit in (state.struct[f"type{layer}"] == ACTIVE).nonzero().flatten().tolist():
            local = mask[:, unit].nonzero().flatten() * mask.shape[1] + unit
            if local.numel():
                group = local + start
                score = float((.5 * theta[group].square() * diagonal[group]).sum())
                groups.append((score / len(group), tuple(int(i) for i in group)))
    groups.sort(key=lambda item: item[0])

    limit = cfg.fisher_pool
    deleted = set(int(i) for i in order[:max(1, limit // 2)])
    for _, group in groups:
        if len(deleted | set(group)) <= limit:
            deleted.update(group)
    for index in order.tolist():
        if len(deleted) >= limit:
            break
        deleted.add(index)
    deletion_ids = tuple(sorted(deleted))

    target = design[:, list(deletion_ids)] @ theta[list(deletion_ids)]
    scale = (diagonal + cfg.fisher_mu).sqrt().clamp_min(1e-15)
    score = (design.T @ target).abs() / scale
    live_ids = live.nonzero().flatten()
    update_ids = tuple(int(i) for i in live_ids[
        torch.argsort(score[live_ids], descending=True)[:limit]])
    coordinates = tuple(sorted(set(deletion_ids) | set(update_ids)))
    return deletion_ids, update_ids, coordinates


def _logical_and(model, terms, name):
    if any(isinstance(term, int) and term == 0 for term in terms):
        return 0
    variables = [term for term in terms if not isinstance(term, int)]
    if not variables:
        return 1
    if len(variables) == 1:
        return variables[0]
    result = model.addVar(vtype="B", name=name)
    model.addGenConstrAnd(result, variables)
    return result


def _logical_or(model, terms, name):
    if any(isinstance(term, int) and term == 1 for term in terms):
        return 1
    variables = [term for term in terms if not isinstance(term, int)]
    if not variables:
        return 0
    if len(variables) == 1:
        return variables[0]
    result = model.addVar(vtype="B", name=name)
    model.addGenConstrOr(result, variables)
    return result


def _canonical_retained(model, adapter, state, offsets, d_by_global):
    availability = []
    for layer in range(adapter.L):
        name = f"W{layer}"
        start, _ = offsets[name]
        mask = state.masks[name]
        rows = []
        for target in range(mask.shape[0]):
            row = []
            for source in range(mask.shape[1]):
                local = target * mask.shape[1] + source
                if not bool(mask[target, source]):
                    row.append(0)
                elif start + local in d_by_global:
                    value = model.addVar(vtype="B", name=f"keep_explicit_{layer}_{target}_{source}")
                    model.addConstr(value + d_by_global[start + local] == 1)
                    row.append(value)
                else:
                    row.append(1)
            rows.append(row)
        availability.append(rows)

    forward = [[1] * adapter.widths[0]]
    for layer in range(adapter.L - 1):
        values = []
        for target in range(adapter.widths[layer + 1]):
            if int(state.struct[f"type{layer}"][target]) == REMOVED:
                values.append(0)
                continue
            terms = [_logical_and(model, (availability[layer][target][source],
                                          forward[layer][source]),
                                  f"forward_edge_{layer}_{target}_{source}")
                     for source in range(adapter.widths[layer])]
            values.append(_logical_or(model, terms, f"forward_{layer}_{target}"))
        forward.append(values)

    backward = [None] * adapter.L + [[1] * adapter.widths[-1]]
    for layer in range(adapter.L - 2, -1, -1):
        values = []
        for source in range(adapter.widths[layer + 1]):
            if int(state.struct[f"type{layer}"][source]) == REMOVED:
                values.append(0)
                continue
            terms = [_logical_and(model, (availability[layer + 1][target][source],
                                          backward[layer + 2][target]),
                                  f"backward_edge_{layer}_{target}_{source}")
                     for target in range(adapter.widths[layer + 2])]
            values.append(_logical_or(model, terms, f"backward_{layer}_{source}"))
        backward[layer + 1] = values

    live = [[1] * adapter.widths[0]]
    for layer in range(adapter.L - 1):
        live.append([_logical_and(model, (forward[layer + 1][unit],
                                          backward[layer + 1][unit]),
                                  f"live_{layer}_{unit}")
                     for unit in range(adapter.widths[layer + 1])])
    live.append([1] * adapter.widths[-1])

    retained = []
    retained_by_global = {}
    for layer in range(adapter.L):
        weight_name = f"W{layer}"
        weight_start, _ = offsets[weight_name]
        mask = state.masks[weight_name]
        for target in range(mask.shape[0]):
            for source in range(mask.shape[1]):
                if bool(mask[target, source]):
                    value = _logical_and(
                        model, (availability[layer][target][source], live[layer][source],
                                live[layer + 1][target]),
                        f"retained_{layer}_{target}_{source}")
                    retained.append(value)
                    retained_by_global[
                        weight_start + target * mask.shape[1] + source] = value
        bias_name = f"b{layer}"
        bias_start, _ = offsets[bias_name]
        bias_mask = state.masks[bias_name]
        if layer < adapter.L - 1:
            for target in range(len(bias_mask)):
                if bool(bias_mask[target]):
                    value = live[layer + 1][target]
                    retained.append(value)
                    retained_by_global[bias_start + target] = value
        else:
            for target in range(len(bias_mask)):
                if bool(bias_mask[target]):
                    retained.append(1)
                    retained_by_global[bias_start + target] = 1
    return retained, retained_by_global


def miqp_moves(adapter, state, cfg) -> list[Move]:
    strategy = validate_strategy(
        cfg.fisher_strategy or resolve_strategy("fisher_general"), "mlp")
    if adapter.loss_type != "bernoulli":
        raise ValueError("Fisher General currently supports Bernoulli MLPs")
    if any(bool(((state.struct[f"type{layer}"] != ACTIVE)
                 & (state.struct[f"type{layer}"] != REMOVED)).any())
           for layer in range(adapter.L - 1)):
        raise ValueError("Fisher General does not support nonstandard hidden-unit states")
    try:
        import gurobipy as gp
    except ImportError as error:
        raise RuntimeError("Fisher General requires gurobipy") from error

    names, offsets, theta, live = _layout(state)
    design = _fisher_design(adapter, state, cfg, names)
    deletion_ids, update_ids, coordinates = _candidate_pool(
        adapter, state, cfg, names, offsets, theta, live, design)
    if not deletion_ids:
        return []
    coordinate_index = {global_index: local for local, global_index in enumerate(coordinates)}
    deletion_set, update_set = set(deletion_ids), set(update_ids)
    local_design = design[:, list(coordinates)]
    q = local_design.T @ local_design
    q = (q + q.T) / 2
    values, vectors = torch.linalg.eigh(q)
    q = (vectors * values.clamp_min(0)) @ vectors.T

    solver = gp.Model("fisher")
    solver.Params.OutputFlag = int(cfg.fisher_log)
    solver.Params.TimeLimit = cfg.fisher_time_limit
    solver.Params.MIPGap = cfg.fisher_gap
    solver.Params.NonConvex = 0
    median = float(theta[live].abs().median())
    sparse = strategy.compensation == "sparse_miqp"
    d, u, c, displacement, bounds = [], [], [], [], []
    d_by_global = {}
    for local, global_index in enumerate(coordinates):
        can_delete, can_update = global_index in deletion_set, global_index in update_set
        dv = solver.addVar(vtype=gp.GRB.BINARY, ub=int(can_delete), name=f"d_{global_index}")
        uv = (solver.addVar(vtype=gp.GRB.BINARY, ub=int(can_update),
                            name=f"u_{global_index}") if sparse else None)
        bound = cfg.fisher_bound * max(abs(float(theta[global_index])), median, 1e-6)
        cv = solver.addVar(lb=-bound if can_update else 0.,
                           ub=bound if can_update else 0., name=f"c_{global_index}")
        xv = solver.addVar(lb=-abs(float(theta[global_index])) - bound,
                           ub=abs(float(theta[global_index])) + bound,
                           name=f"delta_{global_index}")
        if sparse:
            solver.addConstr(dv + uv <= 1)
            solver.addGenConstrIndicator(uv, 0, cv == 0.)
        elif can_delete and can_update:
            solver.addGenConstrIndicator(dv, 1, cv == 0.)
        solver.addConstr(xv == -float(theta[global_index]) * dv + cv)
        d.append(dv)
        u.append(uv)
        c.append(cv)
        displacement.append(xv)
        bounds.append(bound)
        if can_delete:
            d_by_global[global_index] = dv

    if strategy.structure == "reward":
        retained, retained_by_global = _canonical_retained(
            solver, adapter, state, offsets, d_by_global)
        for local, global_index in enumerate(coordinates):
            if global_index not in update_set:
                continue
            survives = retained_by_global[global_index]
            if isinstance(survives, int):
                if not survives:
                    c[local].LB = c[local].UB = 0
                    if sparse:
                        u[local].UB = 0
            elif sparse:
                solver.addConstr(u[local] <= survives)
            else:
                solver.addConstr(c[local] <= bounds[local] * survives)
                solver.addConstr(c[local] >= -bounds[local] * survives)
        retained_expression = gp.quicksum(
            value for value in retained if not isinstance(value, int))
        retained_constant = sum(value for value in retained if isinstance(value, int))
        savings = adapter.k_eff(state) - retained_constant - retained_expression
    else:
        savings = gp.quicksum(d)

    scale = max(float(q.diag().max()),
                cfg.fisher_mu, cfg.fisher_lambda_u,
                cfg.fisher_lambda_d, 1e-12)
    objective = gp.QuadExpr()
    for row in range(len(coordinates)):
        for column in range(len(coordinates)):
            coefficient = .5 * float(q[row, column]) / scale
            if coefficient:
                objective += coefficient * displacement[row] * displacement[column]
        objective += (.5 * cfg.fisher_mu / scale) * c[row] * c[row]
        if sparse:
            objective += (cfg.fisher_lambda_u / scale) * u[row]
    objective -= (cfg.fisher_lambda_d / scale) * savings
    solver.setObjective(objective, gp.GRB.MINIMIZE)

    warm_deleted = deletion_ids[0]
    for local, global_index in enumerate(coordinates):
        chosen = global_index == warm_deleted
        d[local].Start = int(chosen)
        if sparse:
            u[local].Start = 0
        c[local].Start = 0.
        displacement[local].Start = -float(theta[global_index]) if chosen else 0.
    solver.optimize()
    if solver.SolCount == 0:
        status = solver.Status
        solver.dispose()
        raise RuntimeError(f"Gurobi found no feasible Fisher General solution (status {status})")

    deleted = tuple(coordinates[i] for i, value in enumerate(d) if value.X > .5)
    updated = tuple(coordinates[i] for i, value in enumerate(c)
                    if coordinates[i] in update_set
                    and (u[i].X > .5 if sparse else value.X != 0.))
    compensation = torch.tensor([c[coordinate_index[index]].X for index in updated],
                                dtype=design.dtype, device=design.device)
    if not deleted:
        solver.dispose()
        return []
    d_index = torch.tensor(deleted, device=design.device)
    effect = -(design[:, d_index] @ theta[d_index])
    if updated:
        effect += design[:, list(updated)] @ compensation
    damage = float(.5 * effect.square().sum())
    actual_savings = _closure(adapter, state, offsets, deleted)
    rewarded_savings = actual_savings if strategy.structure == "reward" else len(deleted)
    direct_objective = (damage + .5 * cfg.fisher_mu * float(compensation.square().sum())
                        + (cfg.fisher_lambda_u * len(updated) if sparse else 0.)
                        - cfg.fisher_lambda_d * rewarded_savings)
    solver_objective = solver.ObjVal * scale
    tolerance = 1e-7 * max(1., abs(direct_objective), abs(solver_objective))
    if abs(direct_objective - solver_objective) > tolerance:
        solver.dispose()
        raise RuntimeError("Fisher General objective does not agree with direct evaluation")
    modeled_savings = round(float(savings.getValue()))
    if modeled_savings != rewarded_savings:
        solver.dispose()
        raise RuntimeError("Fisher savings term does not match direct evaluation")

    deleted_by_name = _split_indices(deleted, offsets)
    updated_by_name = _split_indices(updated, offsets)
    position = {index: position for position, index in enumerate(updated)}
    updates = {}
    for name, indices in updated_by_name.items():
        start, _ = offsets[name]
        updates[name] = (indices, torch.stack([
            compensation[position[start + int(index)]] for index in indices]))
    gap = float(solver.MIPGap) if solver.IsMIP else 0.
    runtime = float(solver.Runtime)
    status = int(solver.Status)
    bound_active = any(
        abs(c[coordinate_index[index]].X) >= .999 * bounds[coordinate_index[index]]
        for index in updated)
    solver.dispose()
    return [Move(
        "fisher", "parameter_graph", 0,
        {"deleted": deleted_by_name, "updates": updates,
         "n_deleted": len(deleted), "n_updates": len(updated),
         "closure_savings": actual_savings, "fisher_damage": damage,
         "search_objective": direct_objective, "solver_objective": solver_objective,
         "solver_status": status, "solver_gap": gap, "solver_seconds": runtime,
         "compensation_bound_active": bound_active,
         "candidate_deletions": len(deletion_ids),
         "candidate_updates": len(update_ids), "screen_cost": "fisher",
         "fisher_strategy": strategy.as_dict()},
        pred_cost=direct_objective)]
