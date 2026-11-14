"""Joint submodular: greedy cover over weights and the neuron copies that cleanup implies.

Candidates are every deletable coefficient plus, for each hidden neuron, its fan-in cut (the neuron becomes a constant
folded downstream) and its fan-out cut. A candidate's reward is the parameters it newly covers, so the savings K are a
coverage function, exact for neuron-level cleanup; its cost is the K-FAC damage. Each round greedily takes the lowest
damage per newly covered parameter: neuron cuts one at a time with their exact rank-one compensation, weights in the
order scored at the start of the round. A converged AXB solve per changed module then compensates the round exactly."""
from __future__ import annotations

import math
import time
import torch
from pruning.methods.kfac import Coefficients, validate
from pruning.nnprune.kronecker import KFAC, damage, masked_cg
from pruning.nnprune.streaming import kth


def restricted_inverse(matrix, keep):
    result = torch.zeros_like(matrix)
    index = keep.nonzero().flatten()
    if len(index):
        result[index[:, None], index[None, :]] = torch.cholesky_inverse(torch.linalg.cholesky(matrix[index][:, index]))
    return result


def downdate(inverse, index):
    """Drop one coordinate from the inverse of a restricted factor (the Schur complement)."""
    if inverse[index, index] > 0:
        inverse -= torch.outer(inverse[:, index], inverse[index, :]) / inverse[index, index]
        inverse[index, :] = inverse[:, index] = 0


class Module:
    """One affine matrix during a round: working values, restricted factor inverses and what is still uncovered."""
    def __init__(self, co, name, factors, axes):
        self.reference = co.value(name)
        self.value = self.reference.clone()
        self.a, self.b = factors
        self.free = co.live(name)
        self.p = restricted_inverse(self.b, self.free.any(1))
        self.q = restricted_inverse(self.a, self.free.any(0))
        live = co.problem.extra["live"][name].to(self.value.device)
        self.open = co.initial[name].to(self.value.device) & live
        if name in co.protected:
            self.open[:, -1] = False
        self.row_open, self.col_open = self.open.sum(1), self.open.sum(0)
        self.deleted = torch.zeros_like(self.free)
        self.fan = co.modules[name].in_features
        self.wa = self.value[:, :self.fan] @ self.a[:self.fan, :self.fan] if 0 in axes else None
        self.bw = self.b @ self.value if 1 in axes else None

    def update(self, x, y):
        self.value += torch.outer(x, y)
        if self.wa is not None:
            self.wa += torch.outer(x, y[:self.fan] @ self.a[:self.fan, :self.fan])
        if self.bw is not None:
            self.bw += torch.outer(self.b @ x, y)

    def drop_row(self, index):
        downdate(self.p, index)
        self.free[index] = False

    def drop_col(self, index):
        downdate(self.q, index)
        self.free[:, index] = False

    def cover_row(self, index):
        newly = self.open[index].clone()
        self.open[index] = False
        self.row_open[index] = 0
        self.col_open -= newly.long()
        return int(newly.sum())

    def cover_col(self, index):
        newly = self.open[:, index].clone()
        self.open[:, index] = False
        self.col_open[index] = 0
        self.row_open -= newly.long()
        return int(newly.sum())

    def zero(self, rows, cols, chunk=50_000):
        """Delete single weights (no compensation until the round's solve); returns how many were newly covered."""
        values = self.value[rows, cols].clone()
        self.value[rows, cols] = 0
        self.deleted[rows, cols] = True
        self.free[rows, cols] = False
        for start in range(0, len(rows), chunk):
            r, c, v = rows[start:start + chunk], cols[start:start + chunk], values[start:start + chunk]
            if self.wa is not None:
                inside = c < self.fan
                self.wa.index_add_(0, r[inside], -v[inside, None] * self.a[c[inside], :self.fan])
            if self.bw is not None:
                self.bw.index_add_(1, c, -(self.b[:, r] * v))
        newly = self.open[rows, cols].clone()
        self.open[rows, cols] = False
        self.row_open.index_add_(0, rows, -newly.long())
        self.col_open.index_add_(0, cols, -newly.long())
        return int(newly.sum())

    def weight_scores(self, candidates):
        cost = .5 * self.value.square() / (self.p.diag()[:, None] * self.q.diag()[None, :])
        usable = candidates & self.free & self.open[:, :self.value.shape[1]] & self.value.ne(0)
        return cost.masked_fill(~usable | ~cost.isfinite(), torch.inf)


class JointSubmodular:
    uses_selection = False

    def __init__(self, problem, branch):
        validate(branch, curvatures=("kfac",))
        if branch.replay or branch.factor_backend != "memory":
            raise ValueError("joint_submodular estimates every factor once per step: replay=false, factor_backend=memory")
        self.branch, self.options = branch, branch.options
        self.co = Coefficients(problem, branch, permanent=True)

    def groups(self, units):
        """Hidden-neuron groups: (options, hidden width), each option a (module, axis, fan-in) cut."""
        if self.branch.unit == "weight":
            return []
        return [(options, self.co.modules[options[0][0]].out_features) for _, pair in units for options in self.co.neurons(pair)]

    def neuron_scores(self, group, modules):
        options, hidden = group
        costs, size = [], 0
        for name, axis, _ in options:
            module = modules[name]
            if axis == 0:
                cost = .5 * (module.value[:, :module.fan] * module.wa).sum(1) / module.p.diag()
                size = size + module.row_open
            else:
                cost = .5 * (module.value[:, :hidden] * module.bw[:, :hidden]).sum(0) / module.q.diag()[:hidden]
                size = size + module.col_open[:hidden]
            costs.append(torch.nan_to_num(cost, nan=math.inf))
        cost, choice = torch.stack(costs).min(0)
        ratio = torch.where((size > 0) & cost.isfinite(), cost / size.clamp_min(1), torch.inf)
        return ratio, choice, cost

    def cut(self, group, index, option, modules):
        """Apply one neuron cut with its exact rank-one compensation; returns the parameters newly covered."""
        options, _ = group
        name, axis, fan = options[option]
        module = modules[name]
        compensate = self.branch.compensation == "full"
        if axis == 0:
            row = module.value[index].clone()
            row[fan:] = 0
            unit = torch.zeros_like(module.p[:, index])
            unit[index] = 1
            module.update(-(module.p[:, index] / module.p[index, index] if compensate else unit), row)
            module.deleted[index, :fan] = True
        else:
            column = module.value[:, index].clone()
            unit = torch.zeros_like(module.q[index, :])
            unit[index] = 1
            module.update(-column, module.q[index, :] / module.q[index, index] if compensate else unit)
            module.deleted[:, index] = True
        covered = 0
        for other, other_axis, _ in options:
            # The rest of the neuron is dead after the cut: cleanup removes it, so it no longer compensates either.
            target = modules[other]
            (target.drop_row if other_axis == 0 else target.drop_col)(index)
            covered += (target.cover_row if other_axis == 0 else target.cover_col)(index)
        return covered

    def select(self, units, modules, quota):
        """Greedy cover of `quota` parameters inside these units; returns the parameters covered."""
        names = [name for names, _ in units for name in names]
        groups = self.groups(units)
        touching = {name: [index for index, (options, _) in enumerate(groups) if name in {option[0] for option in options}]
                    for name in names}
        best = {index: self.neuron_scores(group, modules) for index, group in enumerate(groups)}
        pool = torch.empty(0)
        if self.branch.unit != "neuron":
            scores = {name: modules[name].weight_scores(self.co.candidates(name)) for name in names}
            finite = sum(int(score.isfinite().sum()) for score in scores.values())
            threshold = kth(lambda: ((score, None) for score in scores.values()), min(quota, finite)) if finite and quota else -math.inf
            entries = [(index, score.le(threshold).nonzero()) for index, score in enumerate(scores.values())]
            module_ids = torch.cat([torch.full((len(found),), index) for index, found in entries])
            positions = torch.cat([found.cpu() for _, found in entries])
            values = torch.cat([scores[names[index]][found[:, 0], found[:, 1]].cpu() for index, found in entries])
            order = values.argsort()
            pool, module_ids, positions = values[order], module_ids[order], positions[order]
        covered, pointer, cuts = 0, 0, 0
        while covered < quota:
            group_index, neuron, ratio = None, None, math.inf
            for index, (scores_, _, _) in best.items():
                value, position = scores_.min(0)
                if float(value) < ratio:
                    group_index, neuron, ratio = index, int(position), float(value)
            end = int(torch.searchsorted(pool, torch.tensor(ratio, dtype=pool.dtype), right=True)) if len(pool) else 0
            end = min(end, pointer + quota - covered)
            if end > pointer:
                changed, block = set(), slice(pointer, end)
                for index in module_ids[block].unique().tolist():
                    mask = module_ids[block] == index
                    rows, cols = positions[block][mask].T.to(modules[names[index]].value.device)
                    covered += modules[names[index]].zero(rows, cols)
                    changed.add(names[index])
                    if covered >= quota:
                        break
                pointer = end
            elif group_index is not None:
                _, choice, _ = best[group_index]
                covered += self.cut(groups[group_index], neuron, int(choice[neuron]), modules)
                cuts += 1
                changed = {option[0] for option in groups[group_index][0]}
            else:
                break
            for index in {index for name in changed for index in touching[name]}:
                best[index] = self.neuron_scores(groups[index], modules)
        return covered, cuts

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        co, started, history = self.co, time.perf_counter(), []
        before, units = problem.remaining(), co.units()
        names = [name for names, _ in units for name in names]
        factors = co.factors(names, self.branch.estimation)
        metric = KFAC(factors, inverses=True)
        axes = {name: set() for name in names}
        for options, _ in self.groups(units):
            for name, axis, _ in options:
                axes[name].add(axis)
        rounds, cuts = max(1, self.options.rounds), 0
        for round_index in range(rounds):
            modules = {name: Module(co, name, factors[name], axes[name]) for name in names}
            if self.branch.scope == "global":
                remaining = co.required(target)
                taken, count = self.select(units, modules, math.ceil(remaining / (rounds - round_index)))
                cuts += count
            else:
                for unit in units:
                    remaining = co.required(target, unit[0])
                    taken, count = self.select([unit], modules, math.ceil(remaining / (rounds - round_index)))
                    cuts += count
            values = {}
            for name, module in modules.items():
                if not module.deleted.any():
                    continue
                reference, start = {name: module.reference}, {name: module.value.masked_fill(module.deleted, 0.)}
                values.update(start if self.branch.compensation == "none" else
                              masked_cg(metric, reference, start, {name: module.free}, rtol=self.options.cg_rtol,
                                        iterations=self.options.cg_max_iterations, strict=True)[0])
                history.append({"round": round_index, "module": name, "deleted": int(module.deleted.sum()),
                                "predicted_fisher_damage": damage(metric, reference, {name: values[name]})})
            co.install(values)
            progress(round_index + 1, history, 0, time.perf_counter() - started)
        return {"layers": history, "new_removed": before - problem.remaining(), "neuron_cuts": cuts,
                "predicted_fisher_damage": sum(row["predicted_fisher_damage"] for row in history)}
