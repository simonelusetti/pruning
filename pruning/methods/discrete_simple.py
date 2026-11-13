"""Discrete simple: matrix-free canonical L0 on the curvature quadratic, with an L0 penalty tuned toward each target."""
from __future__ import annotations

from fnmatch import fnmatchcase
import math
import time
import torch
from pruning.methods.kfac import Coefficients, validate
from pruning.nnprune.kronecker import Scaled, lipschitz
from pruning.nnprune.l0 import minimize_l0
from pruning.nnprune.network import parameter_name
from pruning.nnprune.streaming import Store, kth


class DiscreteSimple:
    uses_selection = False

    def __init__(self, problem, branch):
        validate(branch, curvatures=("kfac", "fisher"))
        options = branch.options
        if options.solver not in {"fast", "accurate"} or options.penalty_trials < 1 or options.penalty_growth <= 1:
            raise ValueError("Invalid L0 solver settings")
        # discrete_simple_global keeps one penalty and permanent deletions over an untargeted greedy path.
        self.fixed = branch.name == "discrete_simple_global"
        self.branch, self.options, self.accurate = branch, options, options.solver == "accurate"
        self.dtype = getattr(torch, options.solver_dtype)
        self.co = Coefficients(problem, branch, permanent=self.fixed)
        patterns = [options.parameters] if isinstance(options.parameters, str) else options.parameters
        self.penalized = {}
        for name, module in self.co.modules.items():
            mask = self.co.candidates(name)
            selected = patterns is None or any(fnmatchcase(parameter_name(name, "weight"), pattern) for pattern in patterns)
            mask[:, :module.in_features] &= selected
            if mask.shape[1] > module.in_features and not options.penalize_bias:
                mask[:, -1] = False
            self.penalized[name] = mask

    def metric(self, store, names):
        if self.branch.curvature == "fisher":
            metric = store.get("fisher", self.co.problem.device)
        else:
            metric = self.co.metric(store, names, inverses=self.accurate)
            metric.factors = {name: (a.to(self.dtype), b.to(self.dtype)) for name, (a, b) in metric.factors.items()}
            if metric.inverses is not None:
                metric.inverses = {name: (a.to(self.dtype), b.to(self.dtype)) for name, (a, b) in metric.inverses.items()}
        return Scaled(metric, names) if self.options.standardize else metric

    def groups(self, units):
        """Units solved together: K-FAC separates them; the exact Fisher couples every unit in a global problem."""
        return [units] if self.branch.curvature == "fisher" and self.branch.scope == "global" else [[unit] for unit in units]

    def prepare(self, group, store, floors):
        """Names, metric, scales and scaled reference of one group; the first call fixes its step floor."""
        names = [name for names, _ in group for name in names]
        metric = self.metric(store, names)
        scale = metric.scales if isinstance(metric, Scaled) else {name: 1. for name in names}
        reference = {name: self.co.value(name, self.dtype) * scale[name] for name in names}
        if names[0] not in floors:
            floors[names[0]] = (lipschitz(metric, reference) if not self.accurate else
                                max(float(metric.diagonal(name).max()) for name in names))
        return names, metric, scale, reference

    def solve(self, group, store, penalty, floors):
        """Solve one group at `penalty`; returns unscaled values, records and the resulting removal count."""
        names, metric, scale, reference = self.prepare(group, store, floors)
        live = self.co.problem.extra.get("live")
        values, record = minimize_l0(
            reference, metric, penalty, {name: self.penalized[name] & self.co.candidates(name) for name in names}, group,
            self.options, floor=floors[names[0]], accurate=self.accurate, neurons_only=self.branch.unit == "neuron",
            fixed={name: ~self.co.live(name) for name in names} if live is not None else None)
        if not self.accurate:
            floors[names[0]] = record["lipschitz"]
        values = {name: value / scale[name] for name, value in values.items()}
        return values, {"modules": names, **record}, self.co.removed(names, values) - self.co.removed(names)

    def calibrate(self, groups, store, needed, floors):
        """Initial penalty: the needed-th smallest coordinatewise deletion cost."""
        def chunks():
            for group in groups:
                names, _, _, reference = self.prepare(group, store, floors)
                for name in names:
                    value = reference[name]
                    yield (.5 * floors[names[0]] * value.square()).masked_fill(~(self.penalized[name] & value.ne(0)), torch.inf), None
        return max(kth(chunks, max(1, needed)), torch.finfo(self.dtype).tiny)

    def search(self, groups, store, target, needed, size, floors):
        """Geometric penalty search toward `needed` removals in these groups; installs the closest trial."""
        problem = self.co.problem
        stored = problem.extra.get("l0_penalty") if self.fixed else None
        calibrate = self.options.penalty is None and stored is None
        penalty = (self.calibrate(groups, store, needed, floors) if calibrate else
                   self.options.penalty if self.options.penalty is not None else stored)
        lower, upper, best, trials = 0., None, None, []
        for trial in range(self.options.penalty_trials if calibrate and target is not None else 1):
            values, history, removed = {}, [], 0
            for group in groups:
                solved, record, count = self.solve(group, store, penalty, floors)
                values.update(solved)
                history.append(record)
                removed += count
            loss = sum(record["predicted_fisher_damage"] for record in history)
            trials.append({"penalty": penalty, "new_removed": removed, "predicted_fisher_damage": loss})
            distance = abs(removed - needed) / size
            if best is None or (distance, loss) < best[0]:
                best = (distance, loss), values, history, trials[-1]
            print(f"L0 penalty trial {trial + 1}: lambda={penalty:.6g}, new removals {removed:,}/{needed:,}", flush=True)
            if distance <= self.options.sparsity_tolerance:
                break
            if removed < needed:
                lower = penalty
                penalty = math.sqrt(lower * upper) if upper is not None else penalty * self.options.penalty_growth
            else:
                upper = penalty
                penalty = math.sqrt(lower * upper) if lower else penalty / self.options.penalty_growth
        _, values, history, summary = best
        self.co.install(values)
        if self.fixed:
            problem.extra["l0_penalty"] = summary["penalty"]
        return history, summary, trials

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        co, branch, started = self.co, self.branch, time.perf_counter()
        before, units, history, trials, floors = problem.remaining(), co.units(), [], [], {}
        if self.fixed and target is None and problem.extra.get("l0_penalty") is None:
            target = self.options.initial_sparsity
        if branch.replay:
            for stage in co.stages(units):
                store = Store()
                for name, factors in co.factors([name for names, _ in stage for name in names], branch.estimation).items():
                    store[name] = factors
                for unit in stage:
                    rows, _, tried = self.search([[unit]], store, target, co.required(target, unit[0]), co.size(unit[0]), floors)
                    history += rows
                    trials += tried
                progress(len(history), history, 0, time.perf_counter() - started)
        else:
            with co.store() as store:
                if branch.curvature == "fisher":
                    store["fisher"] = co.fisher(list(co.modules), branch.estimation, self.dtype)
                else:
                    co.snapshot(units, store, branch.estimation)
                if branch.scope == "global":
                    needed = co.required(target) if target is not None else 0
                    history, _, trials = self.search(self.groups(units), store, target, needed, problem.total, floors)
                else:
                    for unit in units:
                        rows, _, tried = self.search([[unit]], store, target, co.required(target, unit[0]), co.size(unit[0]), floors)
                        history += rows
                        trials += tried
                        progress(len(history), history, 0, time.perf_counter() - started)
        if self.fixed and problem.remaining() >= before:
            return None
        return {"layers": history, "penalty_trials": trials, "new_removed": before - problem.remaining(),
                "predicted_fisher_damage": sum(row["predicted_fisher_damage"] for row in history)}
