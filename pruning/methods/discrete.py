from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from pruning.models.mnist import linears
from pruning.nnprune.canonicalize.mlp import canonicalize_mlp_tensors
from pruning.nnprune.fisher_miqp import add_quadratic_terms


def theta_of(linear):
    return torch.cat((linear.weight, linear.bias[:, None]), 1).double()


def set_theta(linear, theta):
    linear.weight.copy_(theta[:, :-1])
    linear.bias.copy_(theta[:, -1])


def fisher_factors(net, x):
    """Inputs A_l = [a_l, 1] of each linear layer and rows g_l[x, c] = sqrt(p_c)(e_c - p)^T dz/dpre_l.
    F of layer l = mean_x sum_c (g a^T)(g a^T)^T, since sum_c p_c (e_c - p)(e_c - p)^T = diag p - p p^T."""
    l0, l1, l2 = linears(net)
    a0 = x.flatten(1).double()
    h1 = F.linear(a0, l0.weight.double(), l0.bias.double())
    a1 = h1.relu()
    h2 = F.linear(a1, l1.weight.double(), l1.bias.double())
    a2 = h2.relu()
    p = F.linear(a2, l2.weight.double(), l2.bias.double()).softmax(-1)
    g2 = p.sqrt()[:, :, None] * (torch.eye(p.shape[1], dtype=p.dtype, device=p.device)
                                 - p[:, None, :])
    g1 = (g2 @ l2.weight.double()) * (h2 > 0)[:, None]
    g0 = (g1 @ l1.weight.double()) * (h1 > 0)[:, None]
    ones = torch.ones(len(a0), 1, dtype=a0.dtype, device=a0.device)
    return [torch.cat((a, ones), 1) for a in (a0, a1, a2)], [g0, g1, g2]


def weight_zero_candidates(net, A, G, per_family):
    """One scalar deletion with no compensation, ranked by the exact Fisher diagonal."""
    out = []
    for layer, (linear, a, g) in enumerate(zip(linears(net), A, G)):
        theta = linear.weight.double()
        diagonal = g.square().sum(1).T @ a[:, :-1].square() / len(a)
        cost = (.5 * theta.square() * diagonal).masked_fill(theta == 0, torch.inf)
        for flat in torch.argsort(cost.flatten())[:per_family].tolist():
            row, column = divmod(flat, linear.in_features)
            def apply(net, layer=layer, row=row, column=column):
                linear = linears(net)[layer]
                linear.weight[row, column] = 0
            c = float(cost[row, column])
            out.append({"kind": "weight_zero",
                        "label": f"weight_zero:L{layer}:w{row},{column}",
                        "apply": apply, "fisher_cost": c,
                        "fisher_cost_per_removed": c})
    return out


@torch.inference_mode()
def _logic_and(model, terms, name):
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


def _logic_or(model, terms, name):
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


@torch.inference_mode()
def discrete_candidates(net, A, G, cfg):
    try:
        import gurobipy as gp
    except ImportError as error:
        raise RuntimeError("Discrete MIQP requires gurobipy") from error

    matrices = [theta_of(linear) for linear in linears(net)]
    offsets = [0]
    for matrix in matrices:
        offsets.append(offsets[-1] + matrix.numel())
    theta = torch.cat([matrix.flatten() for matrix in matrices])
    live = theta != 0
    n, classes = len(A[0]), G[0].shape[1]

    def columns(indices):
        indices = torch.as_tensor(indices, device=theta.device, dtype=torch.long)
        result = theta.new_empty(n * classes, len(indices))
        for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            positions = ((indices >= start) & (indices < stop)).nonzero().flatten()
            if not positions.numel():
                continue
            local = indices[positions] - start
            width = matrices[layer].shape[1]
            rows, cols = torch.div(local, width, rounding_mode="floor"), local % width
            result[:, positions] = (
                G[layer][:, :, rows] * A[layer][:, None, cols]
            ).reshape(n * classes, -1) / math.sqrt(n)
        return result

    def closure_details(deleted):
        values = [matrix.clone() for matrix in matrices]
        for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            local = torch.tensor([index - start for index in deleted
                                  if start <= index < stop], device=theta.device)
            if local.numel():
                values[layer].view(-1)[local] = 0
        canonicalize_mlp_tensors([value[:, :-1] for value in values],
                                 [value[:, -1] for value in values],
                                 (F.relu, F.relu))
        survivor = torch.cat([(value != 0).flatten() for value in values])
        removed = set((live & ~survivor).nonzero().flatten().tolist())
        return len(removed), removed

    def closure(deleted):
        return closure_details(deleted)[0]

    diagonal = torch.cat([
        torch.einsum("ncr,nf->rf", g.square(), a.square()).flatten() / n
        for a, g in zip(A, G)])
    weight_ids = []
    for layer, linear in enumerate(linears(net)):
        width = linear.in_features + 1
        rows = torch.arange(linear.out_features, device=theta.device)[:, None]
        cols = torch.arange(linear.in_features, device=theta.device)[None]
        ids = offsets[layer] + rows * width + cols
        weight_ids.append(ids[linear.weight != 0])
    weight_ids = torch.cat(weight_ids)
    order = weight_ids[torch.argsort(.5 * theta[weight_ids].square() * diagonal[weight_ids])]

    groups = []
    for layer in (0, 1):
        outgoing = linears(net)[layer + 1]
        width = outgoing.in_features + 1
        for unit in (outgoing.weight != 0).any(0).nonzero().flatten().tolist():
            rows = (outgoing.weight[:, unit] != 0).nonzero().flatten()
            group = tuple((offsets[layer + 1] + rows * width + unit).tolist())
            if len(group) > cfg.max_group:
                continue
            raw = float((.5 * theta[list(group)].square() * diagonal[list(group)]).sum())
            groups.append((raw / max(closure(group), 1), group))
    groups.sort(key=lambda item: item[0])
    structural_blocks = [group for _, group in groups[:cfg.group_pool]]
    structural_closures = [closure_details(group)[1] for group in structural_blocks]
    reserved = (set().union(*structural_closures) if structural_closures else set())
    guards = set()
    for layer in (0, 1):
        incoming = linears(net)[layer]
        incoming_width = incoming.in_features + 1
        for unit in (incoming.weight != 0).any(1).nonzero().flatten().tolist():
            ids = [offsets[layer] + unit * incoming_width + source
                   for source in (incoming.weight[unit] != 0).nonzero().flatten().tolist()]
            available = [index for index in ids if index not in reserved]
            if available:
                guards.add(available[-1])
        outgoing = linears(net)[layer + 1]
        width = outgoing.in_features + 1
        for unit in (outgoing.weight != 0).any(0).nonzero().flatten().tolist():
            ids = [offsets[layer + 1] + row * width + unit
                   for row in (outgoing.weight[:, unit] != 0).nonzero().flatten().tolist()]
            available = [index for index in ids if index not in reserved]
            if available:
                guards.add(available[-1])
    scalar_ids = [int(index) for index in order
                  if int(index) not in reserved and int(index) not in guards]
    scalar_blocks = [tuple(block) for block in
                     (scalar_ids[start:start + cfg.window]
                      for start in range(0, len(scalar_ids), cfg.window))
                     if block][:cfg.pool]
    blocks = scalar_blocks + structural_blocks
    if not blocks:
        return []
    block_savings = [len(block) for block in scalar_blocks]
    block_savings += [closure(block) for block in structural_blocks]
    deletion_pool = set().union(*map(set, blocks))

    target = sum((columns(block) @ theta[list(block)] for block in blocks),
                 theta.new_zeros(n * classes))
    live_ids = live.nonzero().flatten()
    scale = (diagonal + cfg.mu).sqrt().clamp_min(1e-15)
    score = (columns(live_ids).T @ target).abs()
    ranked_updates = live_ids[torch.argsort(score / scale[live_ids], descending=True)]
    update_ids = tuple(int(index) for index in ranked_updates
                       if int(index) not in deletion_pool and int(index) not in reserved
                       )[:cfg.pool]
    block_effects = torch.stack([-(columns(block) @ theta[list(block)])
                                 for block in blocks], 1)
    update_design = columns(update_ids)
    design = torch.cat((block_effects, update_design), 1)
    q = design.T @ design
    q = (q + q.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(q)
    q = (eigenvectors * eigenvalues.clamp_min(0)) @ eigenvectors.T

    solver = gp.Model("discrete")
    solver.Params.OutputFlag = int(cfg.log)
    solver.Params.TimeLimit = cfg.time_limit
    solver.Params.MIPGap = cfg.gap
    solver.Params.NonConvex = 0
    selected = [solver.addVar(vtype=gp.GRB.BINARY, name=f"block_{index}")
                for index in range(len(blocks))]

    median = float(theta[live].abs().median())
    u, c, bounds = [], [], []
    for index in update_ids:
        uv = solver.addVar(vtype=gp.GRB.BINARY, name=f"u_{index}")
        bound = cfg.bound * max(abs(float(theta[index])), median, 1e-6)
        cv = solver.addVar(lb=-bound, ub=bound, name=f"c_{index}")
        solver.addGenConstrIndicator(uv, 0, cv == 0.)
        u.append(uv)
        c.append(cv)
        bounds.append(bound)

    savings = gp.quicksum(value * reward
                              for value, reward in zip(selected, block_savings))

    objective_scale = max(float(q.diag().max()), cfg.mu,
                          cfg.lambda_u,
                          cfg.lambda_d, 1e-12)
    variables = selected + c
    objective = gp.QuadExpr()
    add_quadratic_terms(objective, q, variables, objective_scale)
    for uv, cv in zip(u, c):
        objective += (.5 * cfg.mu / objective_scale) * cv * cv
        objective += (cfg.lambda_u / objective_scale) * uv
    objective -= (cfg.lambda_d / objective_scale) * savings
    solver.setObjective(objective, gp.GRB.MINIMIZE)
    solver.addConstr(gp.quicksum(selected) >= 1)

    for index, variable in enumerate(selected):
        variable.Start = int(index == 0)
    for variable, compensation in zip(u, c):
        variable.Start = 0
        compensation.Start = 0.
    solver.update()
    if cfg.log:
        print(f"MIQP size: {solver.NumVars} variables, {solver.NumConstrs} constraints, "
              f"{solver.NumNZs} nonzeros", flush=True)
    solver.optimize()
    if solver.SolCount == 0:
        status = solver.Status
        solver.dispose()
        raise RuntimeError(f"Gurobi found no feasible solution (status {status})")

    chosen_blocks = tuple(index for index, variable in enumerate(selected) if variable.X > .5)
    deleted = tuple(sorted(set().union(*(set(blocks[index]) for index in chosen_blocks))))
    updated = tuple(index for position, index in enumerate(update_ids) if u[position].X > .5)
    compensation = torch.tensor([c[update_ids.index(index)].X for index in updated],
                                dtype=theta.dtype, device=theta.device)
    effect = block_effects[:, list(chosen_blocks)].sum(1)
    if updated:
        effect += columns(updated) @ compensation
    damage = float(.5 * effect.square().sum())
    actual_savings = closure(deleted)
    direct_objective = (damage + .5 * cfg.mu * float(compensation.square().sum())
                        + cfg.lambda_u * len(updated)
                        - cfg.lambda_d * actual_savings)
    solver_objective = solver.ObjVal * objective_scale
    modeled_savings = round(float(savings.getValue()))
    if modeled_savings != actual_savings:
        solver.dispose()
        raise RuntimeError("Gurobi canonicalization reward disagrees with the canonicalizer: "
                           f"modeled {modeled_savings}, actual {actual_savings}, "
                           f"blocks {chosen_blocks}")
    tolerance = 1e-7 * max(1., abs(direct_objective), abs(solver_objective))
    if abs(direct_objective - solver_objective) > tolerance:
        solver.dispose()
        raise RuntimeError("Gurobi objective disagrees with direct Fisher evaluation")

    gap = float(solver.MIPGap)
    runtime = float(solver.Runtime)
    status = int(solver.Status)
    bound_active = any(abs(c[update_ids.index(index)].X) >=
                       .999 * bounds[update_ids.index(index)] for index in updated)
    solver.dispose()
    deleted_tensor = torch.tensor(deleted, device=theta.device)
    updated_tensor = torch.tensor(updated, device=theta.device)

    def apply(net):
        values = torch.cat([theta_of(linear).flatten() for linear in linears(net)])
        if updated_tensor.numel():
            values[updated_tensor] += compensation
        values[deleted_tensor] = 0
        for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            set_theta(linears(net)[layer], values[start:stop].reshape_as(matrices[layer]))

    return [{
        "kind": "discrete",
        "label": (f"discrete:B{len(chosen_blocks)}:D{len(deleted)}:"
                  f"U{len(updated)}:K{actual_savings}"),
        "apply": apply, "fisher_cost": damage,
        "fisher_cost_per_removed": damage / actual_savings,
        "search_objective": direct_objective,
        "explicit_deletions": len(deleted), "compensating_updates": len(updated),
        "predicted_savings": actual_savings, "solver_objective": solver_objective,
        "solver_status": status, "solver_gap": gap, "solver_seconds": runtime,
        "compensation_bound_active": bound_active,
        "candidate_blocks": len(blocks), "candidate_deletions": len(deletion_pool),
        "candidate_updates": len(update_ids),
    }]


class MnistDiscrete:
    uses_selection = True

    def __init__(self, options, protocol):
        self.cfg, self.protocol = options, protocol

    def propose(self, problem, target, index, progress):
        from pruning.evaluation import choose_candidate
        from pruning.models.mnist import canonicalize
        x, _ = problem.fisher_data(problem.cfg.benchmark.fisher_n, problem.cfg.benchmark.fisher_seed)
        A, G = fisher_factors(problem.model, x)
        proposals = discrete_candidates(problem.model, A, G, self.cfg)
        def apply(proposal):
            with torch.no_grad():
                proposal["apply"](problem.model)
                canonicalize(problem.model)
        return choose_candidate(problem, [{**p, "apply": lambda p=p: apply(p)}
                                          for p in proposals], self.protocol)
