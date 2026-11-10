from __future__ import annotations

from fnmatch import fnmatchcase
import math
import torch
from pruning.nnprune.network import Network
from pruning.nnprune.fisher_metric import Fisher


def logical(model, kind, terms, name):
    absorbing, neutral = (0, 1) if kind == "and" else (1, 0)
    if any(isinstance(term, int) and term == absorbing for term in terms):
        return absorbing
    variables = list({id(term): term for term in terms if not isinstance(term, int)}.values())
    if len(variables) < 2:
        return variables[0] if variables else neutral
    key = kind, tuple(sorted(id(variable) for variable in variables))
    cache = model._logic_cache
    if key in cache:
        return cache[key]
    result = model.addVar(vtype="B", name=name)
    (model.addGenConstrAnd if kind == "and" else model.addGenConstrOr)(result, variables)
    cache[key] = result
    return result


def canonical_retained(model, network, deletion, *, fixed_zeros=True):
    retained, kept = {}, {}
    live_coordinates = network.live.tolist()
    nonzero = network.theta.ne(0).tolist() if fixed_zeros else live_coordinates
    def available(index):
        if not live_coordinates[index]:
            return 0
        if index not in deletion:
            return 1
        # Every coordinate in a deletion block shares the same keep decision.
        key = id(deletion[index])
        if key not in kept:
            kept[key] = model.addVar(vtype="B", name=f"keep_{index}")
            model.addConstr(kept[key] + deletion[index] == 1)
        return kept[key]

    for chain_index, chain in enumerate(network.chains):
        matrices = [network.indices(name) for name in chain.weights]
        connections = [[[available(index) if nonzero[index] else 0 for index in row]
                        for row in matrix.tolist()] for matrix in matrices]
        forward = [[1] * matrices[0].shape[1]]
        for layer, matrix in enumerate(connections[:-1]):
            forward.append([logical(model, "or", [logical(model, "and", (edge, source), "forward_edge")
                            for edge, source in zip(row, forward[-1])], f"forward_{chain_index}_{layer}_{unit}")
                            for unit, row in enumerate(matrix)])
        backward = [[1] * matrices[-1].shape[0]]
        for layer in range(len(matrices) - 1, 0, -1):
            columns = zip(*connections[layer])
            backward.insert(0, [logical(model, "or", [logical(model, "and", (edge, target), "backward_edge")
                               for edge, target in zip(column, backward[0])], f"backward_{chain_index}_{layer}_{unit}")
                               for unit, column in enumerate(columns)])
        live = [[1] * matrices[0].shape[1]] + [
            [logical(model, "and", (a, b), f"live_{chain_index}_{layer}_{unit}")
             for unit, (a, b) in enumerate(zip(incoming, outgoing))]
            for layer, (incoming, outgoing) in enumerate(zip(forward[1:], backward[:-1]))]
        live.append([1] * matrices[-1].shape[0])
        for layer, matrix in enumerate(matrices):
            for row, indices in enumerate(matrix.tolist()):
                for column, index in enumerate(indices):
                    if live_coordinates[index]:
                        retained[index] = logical(model, "and", (available(index), live[layer][column],
                                                 live[layer + 1][row]), f"retained_{index}")
            if chain.biases[layer]:
                for row, index in enumerate(network.indices(chain.biases[layer]).tolist()):
                    if live_coordinates[index]:
                        retained[index] = logical(model, "and", (available(index), live[layer + 1][row]),
                                                 f"retained_{index}")
    return retained, {index: available(index) for index in retained}


def candidate_pool(network, diagonal, options):
    eligible = []
    protected = {name for chain in network.chains for name in chain.biases[1:] if name}
    for name, value in network.parameters.items():
        patterns = [options.parameters] if isinstance(options.parameters, str) else options.parameters
        selected = value.ndim >= 2 if patterns is None else any(fnmatchcase(name, pattern) for pattern in patterns)
        if selected:
            if name in protected:
                raise ValueError(f"{name} is a constant-folding target; keep its bias outside deletion candidates")
            eligible.append(network.indices(name).flatten())
    if not eligible:
        return []
    eligible = torch.cat(eligible)
    eligible = eligible[network.live[eligible] & network.theta[eligible].ne(0)]
    eligible_mask = torch.zeros_like(network.live)
    eligible_mask[eligible] = True
    score = .5 * network.theta.square() * diagonal
    order = eligible[torch.argsort(score[eligible])]
    groups = []
    for chain in network.chains if options.group_pool else ():
        for name in chain.weights[1:]:
            for column in network.indices(name).T:
                group = column[network.live[column] & network.theta[column].ne(0)]
                if not 0 < len(group) <= options.max_group or not bool(eligible_mask[group].all()):
                    continue
                savings = network.closure(group)
                groups.append((float(score[group].sum()) / max(1, savings), group))
    groups.sort(key=lambda item: item[0])
    structural, reserved = [], torch.zeros_like(network.live)
    for _, group in groups:
        if len(structural) >= options.group_pool:
            break
        if not bool(reserved[group].any()):
            structural.append(group)
            reserved[group] = True
    order = order[~reserved[order]]
    blocks = list(order[:options.pool * options.window].split(options.window)) + structural
    return [block for block in blocks if block.numel()]


def solve(network, fisher, options):
    import gurobipy as gp
    diagonal = fisher.diagonal()
    blocks = candidate_pool(network, diagonal, options)
    if not blocks:
        return None
    deletion_pool = torch.cat(blocks)
    correlations = fisher.correlations(deletion_pool).abs() / (diagonal + options.mu).sqrt().clamp_min(1e-15)
    correlations.masked_fill_(~network.live | network.theta.eq(0), -torch.inf)
    count = min(options.update_pool, int((network.live & network.theta.ne(0)).sum()))
    updates = correlations.topk(count).indices
    q = fisher.quadratic(blocks, updates)
    scale = max(float(q.diag().max()), options.mu, options.lambda_u, options.lambda_d, 1e-12)
    solver = gp.Model("discrete")
    solver._logic_cache = {}
    solver.Params.OutputFlag = int(options.log)
    solver.Params.TimeLimit, solver.Params.MIPGap, solver.Params.NonConvex = options.time_limit, options.gap, 0
    selected = [solver.addVar(vtype="B", name=f"delete_{index}") for index in range(len(blocks))]
    deletion = {int(index): variable for block, variable in zip(blocks, selected) for index in block.tolist()}
    retained, available = canonical_retained(solver, network, deletion)
    median = float(network.theta[network.live].abs().median())
    support, compensation, bounds = [], [], []
    for index in updates.tolist():
        indicator = solver.addVar(vtype="B", name=f"update_{index}")
        bound = options.bound * max(abs(float(network.theta[index])), median, 1e-6)
        value = solver.addVar(lb=-bound, ub=bound, name=f"change_{index}")
        solver.addGenConstrIndicator(indicator, 0, value == 0.)
        if index in deletion:
            solver.addConstr(indicator + deletion[index] <= 1)
        solver.addConstr(indicator <= retained.get(index, 1))
        support.append(indicator)
        compensation.append(value)
        bounds.append(bound)
    savings = gp.quicksum(len(block) * variable for block, variable in zip(blocks, selected))
    savings += gp.quicksum(available[index] - value for index, value in retained.items())
    variables = selected + compensation
    rows, columns = q.nonzero(as_tuple=True)
    objective = gp.QuadExpr()
    objective.addTerms((.5 * q[rows, columns] / scale).tolist(),
                       [variables[index] for index in rows.tolist()],
                       [variables[index] for index in columns.tolist()])
    objective += gp.quicksum((.5 * options.mu * value * value + options.lambda_u * indicator) / scale
                            for value, indicator in zip(compensation, support))
    solver.setObjective(objective - options.lambda_d / scale * savings, gp.GRB.MINIMIZE)
    if options.require_deletion:
        solver.addConstr(gp.quicksum(selected) >= 1)
    for index, variable in enumerate(selected):
        variable.Start = int(index == 0)
    for indicator, value in zip(support, compensation):
        indicator.Start, value.Start = 0, 0.
    solver.optimize()
    if not solver.SolCount:
        status = solver.Status
        solver.dispose()
        raise RuntimeError(f"Gurobi found no feasible solution (status {status})")
    choices = torch.tensor([variable.X > .5 for variable in selected], dtype=torch.bool, device=q.device)
    if not bool(choices.any()):
        solver.dispose()
        return None
    deleted = torch.cat([block for block, chosen in zip(blocks, choices.tolist()) if chosen])
    changed = torch.tensor([indicator.X > .5 for indicator in support], dtype=torch.bool, device=q.device)
    values = q.new_tensor([value.X for value in compensation])
    displacement = torch.cat((choices.to(q), values))
    damage = float(.5 * displacement @ q @ displacement)
    actual_savings = network.closure(deleted)
    direct = damage + .5 * options.mu * float(values.square().sum()) + options.lambda_u * int(changed.sum()) - options.lambda_d * actual_savings
    if round(savings.getValue()) != actual_savings or not math.isclose(direct, solver.ObjVal * scale, abs_tol=1e-7, rel_tol=1e-7):
        solver.dispose()
        raise RuntimeError("MIQP objective or canonical savings disagrees with applied move")
    record = {"kind": "discrete", "n_deleted": len(deleted), "n_updates": int(changed.sum()),
              "closure_savings": actual_savings, "fisher_cost": damage,
              "fisher_cost_per_removed": damage / actual_savings, "search_objective": direct,
              "solver_objective": solver.ObjVal * scale, "solver_gap": solver.MIPGap,
              "solver_seconds": solver.Runtime, "solver_status": solver.Status,
              "candidate_blocks": len(blocks), "candidate_deletions": len(deletion_pool),
              "candidate_updates": len(updates), "fisher_likelihood": fisher.likelihood,
              "solver_variables": solver.NumVars,
              "canonical_chains": len(network.chains),
              "compensation_bound_active": any(abs(value.X) >= .999 * bound for value, bound in zip(compensation, bounds))}
    solver.dispose()
    return deleted, updates[changed], values[changed], record


class Discrete:
    uses_selection = False

    def __init__(self, problem, options):
        self.options = options
        if (min(options.lambda_u, options.lambda_d, options.mu) < 0 or
                min(options.pool, options.window, options.max_group, options.fisher_batch) < 1 or
                min(options.group_pool, options.update_pool) < 0 or options.time_limit <= 0 or
                not 0 <= options.gap < 1 or options.bound <= 0 or
                options.variance <= 0 or options.likelihood not in {"auto", "bernoulli", "categorical", "gaussian"} or
                (options.fisher_n is not None and options.fisher_n < 1)):
            raise ValueError("invalid discrete optimizer settings")
        network = Network(problem, options.canonicalization)
        if hasattr(problem, "extra"):
            problem.extra["pruned_coordinates"] = network.masks
        self.chains = network.chains

    def propose(self, problem, target, index, progress):
        options = self.options
        network = Network(problem, canonicalization=False)
        network.chains = self.chains
        x, _ = problem.fisher_data(options.fisher_n, options.fisher_seed)
        fisher = Fisher(network, x, batch_size=options.fisher_batch,
                        likelihood=options.likelihood, variance=options.variance)
        result = solve(network, fisher, options)
        if result is None:
            return None
        deleted, updates, values, record = result
        network.apply(deleted, updates, values)
        record["actual_savings"] = int(network.live.sum()) - sum(int(mask.sum()) for mask in network.masks.values())
        return record
