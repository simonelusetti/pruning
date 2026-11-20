"""Singular: rank deletions by K-FAC OBS costs, then give every surviving coefficient its full AXB update."""
from __future__ import annotations

import time
import torch
from pruning.methods.kfac import Coefficients, trusted, full, validate
from pruning.nnprune.kronecker import damage, masked_cg
from pruning.nnprune.streaming import Store, kth


class Singular:
    uses_selection = False

    def __init__(self, problem, branch):
        validate(branch, curvatures=("kfac",))
        self.branch, self.options = branch, branch.options
        self.co = Coefficients(problem, branch, permanent=True)

    def costs(self, unit, metric):
        """Singleton costs per removed coefficient: weights (size 1) and whole neurons (cheapest cut over its size)."""
        names, pair = unit
        co, chunks = self.co, []
        # Without compensation a deletion costs its OBD diagonal term, not its compensated OBS cost; with target=labels
        # it also costs g'delta, delta being the deletion plus, under compensation, its quadratic response.
        compensate = self.branch.compensation == "full"
        if self.branch.unit != "neuron":
            for name in names:
                value = co.value(name)
                cost = .5 * value.square() * (1 / metric.inverse_diagonal(name) if compensate else metric.diagonal(name))
                if name in co.linear:
                    inverse_a, inverse_b = metric.inverses[name]
                    g = co.linear[name]
                    cost = cost - value * (inverse_b @ g @ inverse_a / metric.inverse_diagonal(name) if compensate else g)
                chunks.append({"options": ((name, None, None),), "ratio": cost.masked_fill(~(co.candidates(name) & value.ne(0)), torch.inf),
                               "size": None, "choice": None})
        if self.branch.unit != "weight":
            for options in co.neurons(pair):
                costs, size = [], 0
                hidden = self.co.modules[options[0][0]].out_features
                for name, axis, fan_in in options:
                    value, (a, b), (inverse_a, inverse_b) = co.value(name), metric.factors[name], metric.inverses[name]
                    g = co.linear.get(name, torch.zeros_like(value))
                    alive = (co.initial[name] & full(co.modules[name]).ne(0).to(co.device)).to(value.device)
                    if axis == 0:
                        rows = value[:, :fan_in]
                        costs.append(.5 * (rows * (rows @ a[:fan_in, :fan_in])).sum(1) *
                                     (1 / inverse_b.diag() if compensate else b.diag()) -
                                     (rows * (inverse_b @ g[:, :fan_in] / inverse_b.diag()[:, None] if compensate else
                                              g[:, :fan_in])).sum(1))
                        size = size + alive.sum(1)
                    else:
                        columns = value[:, :hidden]
                        costs.append(.5 * (columns * (b @ columns)).sum(0) *
                                     (1 / inverse_a.diag()[:hidden] if compensate else a.diag()[:hidden]) -
                                     (columns * ((g @ inverse_a)[:, :hidden] / inverse_a.diag()[:hidden] if compensate else
                                                 g[:, :hidden])).sum(0))
                        size = size + alive[:, :hidden].sum(0)
                cost, choice = torch.stack(costs).min(0)
                chunks.append({"options": options, "ratio": torch.where(size > 0, cost / size.clamp_min(1), torch.inf),
                               "size": size.double(), "choice": choice})
        return chunks

    def deletions(self, names, chunks, threshold):
        """Deleted coordinates, and the frozen rest of every cut neuron (cleanup removes it, so it does not compensate)."""
        deleted = {name: torch.zeros_like(self.co.candidates(name)) for name in names}
        frozen = {name: torch.zeros_like(mask) for name, mask in deleted.items()}
        for chunk in chunks:
            chosen = chunk["ratio"] <= threshold
            for index, (name, axis, fan_in) in enumerate(chunk["options"]):
                if axis is None:
                    deleted[name] |= chosen
                    continue
                picked = chosen & chunk["choice"].eq(index)
                if axis == 0:
                    deleted[name][picked, :fan_in] = True
                else:
                    deleted[name][:, :len(chosen)][:, picked] = True
            for name, axis, _ in chunk["options"] if chunk["options"][0][1] is not None else ():
                if axis == 0:
                    frozen[name][chosen] = True
                else:
                    frozen[name][:, :len(chosen)][:, chosen] = True
        return {name: (mask & self.co.candidates(name), frozen[name]) for name, mask in deleted.items()}

    def prune(self, unit, store, target, threshold, history):
        names, co = unit[0], self.co
        metric = co.metric(store, names, inverses=True)
        chunks = store.get(("cost", names), co.problem.device) if threshold is not None else self.costs(unit, metric)
        if threshold is None:
            required = co.required(target, names)
            threshold = kth(lambda: ((chunk["ratio"], chunk["size"]) for chunk in chunks), required) if required else -torch.inf
        values = {}
        for name, (mask, frozen) in self.deletions(names, chunks, threshold).items():
            if not mask.any():
                continue
            reference = {name: co.value(name)}
            if self.branch.compensation == "none":
                values[name] = reference[name].masked_fill(mask, 0.)
                history.append({"module": name, "deleted": int(mask.sum()),
                                "predicted_fisher_damage": damage(metric, reference, {name: values[name]})})
                continue
            values.update(masked_cg(metric, reference, {name: reference[name].masked_fill(mask, 0.)},
                                    {name: co.live(name) & ~mask & ~frozen}, rtol=self.options.cg_rtol,
                                    iterations=self.options.cg_max_iterations, strict=True)[0])
            history.append({"module": name, "deleted": int(mask.sum()),
                            "predicted_fisher_damage": damage(metric, reference, {name: values[name]})})
        co.install(values)

    @trusted
    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        co, branch, started, history = self.co, self.branch, time.perf_counter(), []
        before, units = problem.remaining(), co.units()
        if branch.replay:
            for stage in co.stages(units):
                store = Store()
                for name, factors in co.factors([name for names, _ in stage for name in names], branch.estimation).items():
                    store[name] = factors
                for unit in stage:
                    self.prune(unit, store, target, None, history)
                progress(len(history), history, 0, time.perf_counter() - started)
        else:
            with co.store() as store:
                co.snapshot(units, store, branch.estimation)
                threshold = None
                if branch.scope == "global":
                    for unit in units:
                        store[("cost", unit[0])] = self.costs(unit, co.metric(store, unit[0], inverses=True))
                    required = co.required(target)
                    threshold = kth(lambda: ((chunk["ratio"], chunk["size"]) for unit in units
                                             for chunk in store.get(("cost", unit[0]), problem.device)), required) if required else -torch.inf
                for unit in units:
                    self.prune(unit, store, target, threshold, history)
                    progress(len(history), history, 0, time.perf_counter() - started)
        return {"layers": history, "new_removed": before - problem.remaining(),
                "predicted_fisher_damage": sum(row["predicted_fisher_damage"] for row in history)}
