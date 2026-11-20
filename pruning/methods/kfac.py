"""What our curvature methods share: the pruned coefficients, their neuron structure, budgets, cleanup and curvature.

Shared settings (conf/settings/default.yaml) mean the same thing in every method:
  scope   local: each solve unit meets the target on its own; global: one model-wide budget and problem.
  replay  local only: units are pruned block by block, each measured on the already pruned model.
  unit    weight: single coefficients; neuron: whole hidden units; both: either, with neuron savings credited.
  affine  biases are deletion candidates and compensation variables, or frozen.
  target  model: costs are the quadratic KL to the dense model; labels: deletion costs also carry g'delta, g the
          true-label loss gradient on the held-out selection split, with delta the deletion and, under compensation,
          its quadratic response. Compensation never follows g, so nothing moves unless something is deleted.
A solve unit is one affine module, or a whole canonical group (MLP chain, gated MLP) when neurons are candidates."""
from __future__ import annotations

import contextlib
import copy
import math
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pruning.nnprune.kronecker import KFAC, RowsFisher, example_kl, kronecker_factors
from pruning.nnprune.fisher_metric import Fisher
from pruning.nnprune.network import Network, linear_chains, parameter_name
from pruning.nnprune.canonicalize.mlp import canonicalize_mlp_tensors
from pruning.nnprune.canonicalize.gated import GatedGroup, canonicalize_gated
from pruning.nnprune.streaming import Store
from pruning.paths import SCRATCH


def validate(branch, *, curvatures, scopes=("local", "global"), units=("weight", "neuron", "both")):
    if branch.scope not in scopes or branch.unit not in units or branch.curvature not in curvatures:
        raise ValueError(f"{branch.name} supports scope {scopes}, unit {units}, curvature {curvatures}")
    if branch.replay and (branch.scope != "local" or branch.curvature != "kfac"):
        raise ValueError("replay prunes block by block, so it needs scope=local and K-FAC curvature")
    if branch.compensation not in {"full", "none"} or (branch.compensation == "none" and branch.name not in {"singular", "joint_submodular"}):
        raise ValueError("compensation is full, or none with singular or joint_submodular")
    if branch.objective not in {"kl", "spread"} or (branch.objective == "spread" and branch.curvature != "kfac"):
        raise ValueError("objective is kl, or spread with K-FAC curvature")
    if branch.factor_backend not in {"memory", "streaming"} or (branch.factor_backend == "streaming" and branch.curvature != "kfac"):
        raise ValueError("factor_backend is memory, or streaming with K-FAC curvature")
    if branch.target not in {"model", "labels"} or (branch.target == "labels" and branch.name not in {"singular", "joint_submodular"}):
        raise ValueError("target is model, or labels with singular or joint_submodular")


def trusted(propose):
    """With target=labels a step is tried with the factor damping scaled by each of `branch.trust`, keeping the
    lowest selection-split loss: a grid Levenberg-Marquardt trust region on how far the linear term is believed."""
    def search(self, problem, target, index, progress):
        if self.branch.target != "labels":
            return propose(self, problem, target, index, progress)
        estimation = self.branch.estimation
        base, start, best = (estimation.a_damping, estimation.b_damping), problem.snapshot(), None
        try:
            for scale in self.branch.trust:
                problem.restore(start)
                estimation.a_damping, estimation.b_damping = base[0] * scale, base[1] * scale
                result = propose(self, problem, target, index, progress)
                loss = float(problem.losses("selection").mean())
                if best is None or loss < best[0]:
                    best = loss, scale, problem.snapshot(), result
        finally:
            estimation.a_damping, estimation.b_damping = base
        problem.restore(best[2])
        return best[3] | {"trust_scale": best[1], "trust_selection_loss": best[0]}
    return search


def full(module):
    return (torch.cat((module.weight.detach(), module.bias.detach()[:, None]), 1)
            if module.bias is not None else module.weight.detach().clone())


@torch.no_grad()
def write(module, values):
    module.weight.copy_(values[:, :module.in_features].to(module.weight))
    if module.bias is not None:
        module.bias.copy_(values[:, -1].to(module.bias))


class Coefficients:
    def __init__(self, problem, branch, *, permanent):
        self.problem, self.branch = problem, branch
        model, extra = problem.model, problem.extra
        model.eval()
        self.modules = (problem.affine_modules() if hasattr(problem, "affine_modules") else
                        {name: module for name, module in model.named_modules() if type(module) is nn.Linear})
        self.device = problem.cfg.runtime.get("state_device", problem.device)
        self.initial = {name: full(module).ne(0).to(self.device) for name, module in self.modules.items()}
        if permanent:
            extra["live"] = {name: torch.ones_like(mask) for name, mask in self.initial.items()}
        extra["pruned_coordinate_convention"] = "live"
        chains = problem.linear_chains() if hasattr(problem, "linear_chains") else linear_chains(model)
        self.chains = [(names, chain.activations) for chain in chains
                       if set(names := tuple(weight.rpartition(".")[0] for weight in chain.weights)) <= self.modules.keys()]
        groups = problem.gated_groups() if hasattr(problem, "gated_groups") else []
        self.gated = [group for group in groups if set(group.names) <= self.modules.keys()]
        # Downstream biases absorb folded constants, so they are never deletion candidates.
        self.protected = {name for names, _ in self.chains for name in names[1:] if self.modules[name].bias is not None}
        if (branch.objective == "spread" or branch.target == "labels") and hasattr(problem, "b_sequences"):
            raise ValueError("objective=spread and target=labels currently need a supervised benchmark")
        self.linear = {}
        self.dense = copy.deepcopy(model).eval() if branch.objective == "spread" else None
        self.account()

    def width(self, name):
        module = self.modules[name]
        return module.in_features + int(self.branch.affine and module.bias is not None)

    def value(self, name, dtype=torch.float64):
        return full(self.modules[name])[:, :self.width(name)].to(dtype)

    def live(self, name):
        """Free coordinates: everything, or for permanent deletions everything not yet deleted."""
        live = self.problem.extra.get("live")
        mask = live[name] if live is not None else torch.ones_like(self.initial[name])
        return mask[:, :self.width(name)].to(self.problem.device)

    def candidates(self, name):
        """Deletable coordinates: initially nonzero, still live, not a protected folding bias."""
        mask = self.initial[name] & self.problem.extra.get("live", self.initial)[name]
        if name in self.protected:
            mask[:, -1] = False
        return mask[:, :self.width(name)].to(self.problem.device)

    def units(self):
        if self.branch.unit == "weight":
            return [((name,), None) for name in self.modules]
        units, used = [], set()
        for names, _ in self.chains:
            edges = tuple((left, right, self.modules[left].in_features) for left, right in zip(names, names[1:]))
            units.append((names, edges[0] if len(edges) == 1 else edges))
            used.update(names)
        for group in self.gated:
            units.append((group.names, group))
            used.update(group.names)
        return units + ([((name,), None) for name in self.modules if name not in used] if self.branch.unit == "both" else [])

    def stages(self, units):
        """Units in forward order, grouped by the problem's blocks (decoder layers) or one unit per stage."""
        if not hasattr(self.problem, "stage"):
            return [[unit] for unit in units]
        groups = {}
        for unit in units:
            groups.setdefault(self.problem.stage(unit[0][0]), []).append(unit)
        return [groups[key] for key in sorted(groups)]

    def neurons(self, pair):
        """Hidden units of a canonical group: (incoming rows, outgoing column) per neuron, as module/axis options."""
        if isinstance(pair, GatedGroup):
            return [((pair.gate, 0, self.modules[pair.gate].in_features), (pair.up, 0, self.modules[pair.up].in_features),
                     (pair.down, 1, None))]
        edges = [pair] if pair is not None and not isinstance(pair[0], tuple) else list(pair or ())
        return [((left, 0, fan_in), (right, 1, None)) for left, right, fan_in in edges]

    def expand(self, name, value):
        module = self.modules[name]
        if value.shape[1] < module.in_features + int(module.bias is not None):
            value = torch.cat((value, module.bias.detach()[:, None].to(value)), 1)
        return value

    def canonicalize(self, values):
        """Function-preserving cleanup of every canonical group whose matrices are all in `values` (in place)."""
        for names, activations in self.chains:
            if set(names) <= values.keys():
                canonicalize_mlp_tensors([values[name][:, :self.modules[name].in_features] for name in names],
                                         [values[name][:, -1] if self.modules[name].bias is not None else None for name in names],
                                         activations)
        for group in self.gated:
            if set(group.names) <= values.keys():
                canonicalize_gated(*(values[name] for name in group.names))

    def removed(self, names, values=None):
        """Initially nonzero coordinates that are zero now, or would be after installing `values` and cleanup."""
        current = {name: full(self.modules[name]).double() for name in names}
        if values is not None:
            current.update({name: self.expand(name, value).double() for name, value in values.items()})
            self.canonicalize(current)
        return sum(int((self.initial[name] & current[name].eq(0).to(self.device)).sum()) for name in names)

    def size(self, names):
        return sum(self.initial[name].numel() for name in names)

    @torch.no_grad()
    def install(self, values):
        full_values = {name: self.expand(name, value).double() for name, value in values.items()}
        touched = set(full_values)
        for names in [names for names, _ in self.chains] + [group.names for group in self.gated]:
            if touched & set(names):
                full_values.update({name: full(self.modules[name]).double() for name in names if name not in full_values})
        self.canonicalize(full_values)
        for name, value in full_values.items():
            write(self.modules[name], value)
            if "live" in self.problem.extra:
                self.problem.extra["live"][name] &= ~(value.eq(0).to(self.device) & self.initial[name])
        self.account(full_values)

    def account(self, names=None):
        masks = self.problem.extra.setdefault("pruned_coordinates", {})
        for name in self.modules if names is None else names:
            module = self.modules[name]
            retained = full(module).ne(0).to(self.device) | ~self.initial[name]
            masks[parameter_name(name, "weight")] = retained[:, :module.in_features]
            if module.bias is not None:
                masks[parameter_name(name, "bias")] = retained[:, -1]

    def required(self, target, names=None):
        """New removals needed to reach `target`: model-wide, or within the given unit's modules."""
        problem = self.problem
        if names is None:
            return max(0, math.ceil(target * problem.total) - (problem.total - problem.remaining()))
        return max(0, math.ceil(target * self.size(names)) - self.removed(names))

    def factors(self, names, estimation):
        """Damped K-FAC factors. With objective=spread, each example is weighted by how much more than average the
        current model already distorts its prediction (exact KL from the dense model, standardized), which is the
        reweighted-Fisher step for the bound mean_i KL_i + beta * std_i KL_i, beta = sqrt(2 log(1/eta) / n).
        With target=labels the same call refreshes `linear`, the held-out gradient at the current weights."""
        problem, sampling = self.problem, self.problem.cfg.sampling
        language = hasattr(problem, "b_sequences")
        x, _ = problem.fisher_data(estimation.n, estimation.seed)
        weights = self.spread_weights(x, estimation) if self.dense is not None else None
        factors = kronecker_factors(
            problem.model, x, estimation, {name: self.modules[name] for name in names},
            b_inputs=torch.cat(problem.b_sequences).to(x.device) if language else None,
            max_b_tokens=(sampling.b_tokens or None) if language else None,
            a_batch_size=sampling.activation_batch_size if language else None,
            b_batch_size=sampling.b_batch_size if language else None, weights=weights)
        if self.branch.target == "labels":
            self.linear.update(self.label_gradient(names, estimation))
        return {name: (a[:self.width(name), :self.width(name)], b) for name, (a, b) in factors.items()}

    @torch.no_grad()
    def spread_weights(self, x, estimation):
        def outputs(model):
            rows = [model(batch) for (batch,) in DataLoader(TensorDataset(x), batch_size=estimation.batch)]
            return torch.cat([row.logits if hasattr(row, "logits") else row for row in rows]).double()
        kl = example_kl(outputs(self.dense), outputs(self.problem.model), estimation)
        spread = kl.std(unbiased=False)
        z = (kl - kl.mean()) / spread if spread > 0 else torch.zeros_like(kl)
        beta = math.sqrt(2 * math.log(1 / estimation.confidence) / len(kl))
        return (1 + beta * z).clamp_min(0)

    def label_gradient(self, names, estimation):
        """Mean gradient of the true-label loss on the selection split, zero on coordinates no longer live."""
        data = self.problem.splits["selection"]
        batches = data if isinstance(data, DataLoader) else DataLoader(TensorDataset(*data), batch_size=estimation.batch)
        parameters = [parameter for name in names for parameter in (self.modules[name].weight, self.modules[name].bias)
                      if parameter is not None]
        required = [parameter.requires_grad for parameter in parameters]
        totals, count = [torch.zeros_like(parameter, dtype=torch.float64) for parameter in parameters], 0
        try:
            for parameter in parameters:
                parameter.requires_grad_(True)
            with torch.enable_grad():
                for x, y in batches:
                    output = self.problem.model(x.to(self.problem.device))
                    output, y = output.logits if hasattr(output, "logits") else output, y.to(self.problem.device)
                    loss = (F.cross_entropy(output, y, reduction="sum") if output.shape[-1] > 1 else
                            F.binary_cross_entropy_with_logits(output.flatten(), y.to(output).flatten(), reduction="sum"))
                    for total, gradient in zip(totals, torch.autograd.grad(loss, parameters)):
                        total += gradient.double()
                    count += len(y)
        finally:
            for parameter, flag in zip(parameters, required):
                parameter.requires_grad_(flag)
        totals, result = iter(totals), {}
        for name in names:
            gradient = next(totals) / count
            if self.modules[name].bias is not None:
                gradient = torch.cat((gradient, next(totals)[:, None] / count), 1)
            result[name] = gradient[:, :self.width(name)].masked_fill(~self.live(name), 0.)
        return result

    def fisher(self, names, estimation, dtype):
        network = Network(self.problem, canonicalization=False)
        network.masks = {name: torch.ones_like(value, dtype=torch.bool) for name, value in network.parameters.items()}
        x, _ = self.problem.fisher_data(estimation.n, estimation.seed)
        columns, slices, start = [], {}, 0
        for name in names:
            index = network.indices(parameter_name(name, "weight"))
            if self.width(name) > self.modules[name].in_features:
                index = torch.cat((index, network.indices(parameter_name(name, "bias"))[:, None]), 1)
            slices[name] = start, start + index.numel(), index.shape
            columns.append(index.flatten())
            start += index.numel()
        columns = torch.cat(columns)
        fisher = Fisher(network, x, batch_size=estimation.batch, likelihood=estimation.likelihood, variance=estimation.variance)
        if len(x) * fisher_outputs(network, x) * len(columns) > 2e9:
            raise ValueError("Exact Fisher rows exceed memory; use curvature=kfac on this model")
        return RowsFisher(torch.cat([rows[:, columns] for rows in fisher.batches()]), slices, dtype)

    @contextlib.contextmanager
    def store(self):
        """Factors and scores in memory, or in a scratch directory with factor_backend=streaming."""
        if self.branch.factor_backend == "memory":
            yield Store()
            return
        root = Path(self.problem.cfg.runtime.scratch or SCRATCH)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="factors-", dir=root) as directory:
            yield Store(Path(directory))

    def snapshot(self, units, store, estimation):
        """K-FAC factors on the unchanged model: one estimate, or one per stage when streaming."""
        groups = self.stages(units) if self.branch.factor_backend == "streaming" else [units]
        for group in groups:
            names = [name for names, _ in group for name in names]
            for name, factors in self.factors(names, estimation).items():
                store[name] = factors

    def metric(self, store, names, *, inverses):
        return KFAC({name: store.get(name, self.problem.device) for name in names}, inverses=inverses)


def fisher_outputs(network, x):
    with torch.no_grad():
        return network.logits(network.parameters, x[:1]).numel()
