from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def kronecker_factors(model, inputs, settings, modules, *, b_inputs=None, max_b_tokens=None,
                      a_batch_size=None, b_batch_size=None, weights=None):
    """Damped affine K-FAC blocks (A over inputs with a bias column, B over sampled output scores).

    Optional nonnegative per-example `weights` (one per input row, with B estimated on the same inputs) reweight
    every example's contribution to both factors."""
    statistics, handles, batch_shape, phase, scale = {}, [], None, None, None
    if weights is not None and b_inputs is not None:
        raise ValueError("example weights need A and B estimated on the same inputs")
    b_inputs = inputs if b_inputs is None else b_inputs
    weights = torch.ones(len(inputs), device=inputs.device) if weights is None else weights

    def weighted(values):
        return values * scale.reshape(-1, *([1] * (values.ndim - 1))).to(values)
    for name, module in modules.items():
        width = module.in_features + int(module.bias is not None)
        statistics[name] = [module.weight.new_zeros((width, width), dtype=torch.float64),
                            module.weight.new_zeros((module.out_features,) * 2, dtype=torch.float64), 0, 0]

        def accumulate(gradient, key=name):
            gradient = weighted(gradient)
            if settings.causal:
                gradient = gradient.reshape(*batch_shape, -1)[:, :-1]
                if max_b_tokens is not None:
                    quota = max(1, math.ceil(max_b_tokens / len(b_inputs)))
                    positions = torch.linspace(0, gradient.shape[1] - 1, min(quota, gradient.shape[1]),
                                               device=gradient.device).long()
                    gradient = gradient.index_select(1, positions)
            gradient = gradient.detach().reshape(-1, gradient.shape[-1]).double()
            if max_b_tokens is not None:
                gradient = gradient[:max(0, max_b_tokens - statistics[key][3])]
            statistics[key][1].addmm_(gradient.T, gradient)
            statistics[key][3] += len(gradient)

        def capture(_module, args, output, key=name, bias=module.bias is not None):
            if phase == "a":
                values = weighted(args[0].detach().double()).reshape(-1, args[0].shape[-1])
                if bias:
                    values = torch.cat((values, values.new_ones((len(values), 1))), 1)
                statistics[key][0].addmm_(values.T, values)
                statistics[key][2] += len(values)
            else:
                output.requires_grad_(True)
                output.register_hook(lambda gradient: accumulate(gradient, key))
        handles.append(module.register_forward_hook(capture))
    parameters = list(model.parameters())
    required = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    generator = torch.Generator(device=inputs.device).manual_seed(settings.seed)
    try:
        phase = "a"
        with torch.no_grad():
            for x, scale in DataLoader(TensorDataset(inputs, weights.sqrt()), batch_size=a_batch_size or settings.batch):
                model(x)
        phase = "b"
        with torch.enable_grad():
            for x, scale in DataLoader(TensorDataset(b_inputs, weights.sqrt()), batch_size=b_batch_size or settings.batch):
                if max_b_tokens is not None and min(values[3] for values in statistics.values()) >= max_b_tokens:
                    break
                x = x.detach().requires_grad_(x.is_floating_point())
                batch_shape = x.shape[:2]
                output = model(x)
                logits = output.logits if hasattr(output, "logits") else output
                logits = logits[:, :-1] if settings.causal else logits
                likelihood = settings.likelihood
                if likelihood == "auto":
                    likelihood = "bernoulli" if logits[:1].numel() == 1 else "categorical"
                if likelihood == "categorical":
                    probability = logits.detach().float().softmax(-1)
                    labels = torch.multinomial(probability.reshape(-1, probability.shape[-1]), 1,
                                               generator=generator).view(*probability.shape[:-1], 1)
                    gradient = -probability
                    gradient.scatter_add_(-1, labels, torch.ones_like(labels, dtype=gradient.dtype))
                elif likelihood == "bernoulli":
                    probability = logits.detach().sigmoid()
                    gradient = torch.bernoulli(probability, generator=generator) - probability
                else:
                    gradient = torch.randn(logits.shape, device=logits.device, generator=generator) / math.sqrt(settings.variance)
                if logits.requires_grad:
                    logits.backward(gradient)
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in zip(parameters, required):
            parameter.requires_grad_(flag)
    factors = {}
    for name, (a, b, count_a, count_b) in statistics.items():
        if not count_a or not count_b:
            raise ValueError(f"No curvature observations for {name}")
        a.div_(count_a)
        b.div_(count_b)
        for factor, damping in ((a, settings.a_damping), (b, settings.b_damping)):
            factor.diagonal().add_(damping * factor.diag().mean().clamp_min(torch.finfo(factor.dtype).eps))
        factors[name] = a, b
    return factors


def example_kl(reference, output, settings):
    """Per-example KL(p_reference || p_output) between two models' outputs under the configured likelihood."""
    if settings.likelihood == "gaussian":
        return .5 * (output - reference).square().reshape(len(output), -1).sum(1) / settings.variance
    if output[:1].numel() == 1 or settings.likelihood == "bernoulli":
        p, q = reference.sigmoid().clamp(1e-12, 1 - 1e-12), output.sigmoid().clamp(1e-12, 1 - 1e-12)
        return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).reshape(len(output), -1).sum(1)
    log_p, log_q = reference.log_softmax(-1), output.log_softmax(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).reshape(len(output), -1).sum(1)


def inverse(matrix):
    return torch.cholesky_inverse(torch.linalg.cholesky(matrix))


def dot(first, second):
    return float(sum((first[name] * second[name]).sum(dtype=torch.float64) for name in first))


class KFAC:
    """Block-diagonal metric: module m contributes B_m V A_m; no cross-module terms."""
    def __init__(self, factors, *, inverses=False):
        self.factors = factors
        self.inverses = {name: (inverse(a), inverse(b)) for name, (a, b) in factors.items()} if inverses else None

    def product(self, values):
        return {name: self.factors[name][1] @ value @ self.factors[name][0] for name, value in values.items()}

    def diagonal(self, name):
        a, b = self.factors[name]
        return b.diag()[:, None] * a.diag()[None, :]

    def inverse_diagonal(self, name):
        a, b = self.inverses[name]
        return b.diag()[:, None] * a.diag()[None, :]

    def precondition(self, values):
        if self.inverses is None:
            return {name: value / self.diagonal(name).clamp_min(torch.finfo(value.dtype).tiny)
                    for name, value in values.items()}
        return {name: self.inverses[name][1] @ value @ self.inverses[name][0] for name, value in values.items()}


class RowsFisher:
    """Exact model Fisher on the coefficient matrices, from square-root rows held in memory (small models only)."""
    def __init__(self, rows, slices, dtype):
        self.rows, self.slices = rows.to(dtype), slices
        self.diagonals = self.rows.square().sum(0)

    def flat(self, values):
        result = self.rows.new_zeros(self.rows.shape[1])
        for name, value in values.items():
            start, stop, _ = self.slices[name]
            result[start:stop] = value.flatten()
        return result

    def unflat(self, flat, names):
        return {name: flat[self.slices[name][0]:self.slices[name][1]].view(self.slices[name][2]) for name in names}

    def product(self, values):
        return self.unflat(self.rows.T @ (self.rows @ self.flat(values)), values)

    def diagonal(self, name):
        return self.unflat(self.diagonals, [name])[name]

    def precondition(self, values):
        return {name: value / self.diagonal(name).clamp_min(torch.finfo(value.dtype).tiny)
                for name, value in values.items()}


class Scaled:
    """Standardised coordinates s*V with s = sqrt(diag F), as unit-norm columns in L0 regression."""
    def __init__(self, metric, names):
        self.metric = metric
        self.scales = {}
        for name in names:
            diagonal = metric.diagonal(name)
            self.scales[name] = torch.where(diagonal > 0, diagonal.clamp_min(0).sqrt(), torch.ones_like(diagonal))

    def product(self, values):
        result = self.metric.product({name: value / self.scales[name] for name, value in values.items()})
        return {name: value / self.scales[name] for name, value in result.items()}

    def diagonal(self, name):
        return self.metric.diagonal(name) / self.scales[name].square()

    def precondition(self, values):
        result = self.metric.precondition({name: value * self.scales[name] for name, value in values.items()})
        return {name: value * self.scales[name] for name, value in result.items()}


def lipschitz(metric, shapes, iterations=12):
    """Power-iteration estimate of the largest metric eigenvalue, never below the largest diagonal."""
    vector = {name: value.new_ones(value.shape) for name, value in shapes.items()}
    rayleigh = 0.
    for _ in range(iterations):
        norm = math.sqrt(max(dot(vector, vector), torch.finfo(torch.float64).tiny))
        vector = {name: value / norm for name, value in vector.items()}
        product = metric.product(vector)
        rayleigh, vector = dot(vector, product), product
    diagonal = max(float(metric.diagonal(name).max()) for name in shapes)
    return max(1.01 * rayleigh, diagonal, torch.finfo(torch.float64).tiny)


@torch.no_grad()
def masked_cg(metric, reference, start, free, *, rtol, iterations, strict):
    """Minimise 0.5 (V-W)' F (V-W) over the free coordinates; the rest stay at `start`.

    Residuals are relative to the gradient at W on the free coordinates, the scale of the fixed deletions."""
    def masked(values):
        return {name: tensor.masked_fill(~free[name], 0.) for name, tensor in values.items()}

    def residual():
        return masked({name: -tensor for name, tensor in
                       metric.product({name: value[name] - reference[name] for name in value}).items()})

    value = {name: torch.where(free[name], reference[name], tensor) for name, tensor in start.items()}
    current = residual()
    initial = math.sqrt(dot(current, current))
    value = {name: tensor.clone() for name, tensor in start.items()}
    current = residual()
    count = 0
    if initial > 0:
        z = masked(metric.precondition(current))
        direction, rz = z, dot(current, z)
        for count in range(1, iterations + 1):
            product = masked(metric.product(direction))
            curvature = dot(direction, product)
            if not math.isfinite(curvature) or curvature <= 0:
                if strict:
                    raise RuntimeError("masked CG encountered nonpositive curvature")
                break
            alpha = rz / curvature
            for name in value:
                value[name].add_(direction[name], alpha=alpha)
                current[name].sub_(product[name], alpha=alpha)
            if math.sqrt(dot(current, current)) <= rtol * initial:
                current = residual()  # A fresh residual guards against recursive drift.
                if math.sqrt(dot(current, current)) <= rtol * initial:
                    break
                z = masked(metric.precondition(current))
                direction, rz = z, dot(current, z)
                continue
            z = masked(metric.precondition(current))
            next_rz = dot(current, z)
            direction = {name: z[name] + (next_rz / rz) * direction[name] for name in z}
            rz = next_rz
        current = residual()
    relative = math.sqrt(dot(current, current)) / initial if initial > 0 else 0.
    if strict and relative > rtol:
        raise RuntimeError(f"masked CG did not converge in {iterations} iterations; relative residual={relative:g}")
    return value, {"iterations": count, "relative_residual": relative}


def damage(metric, reference, value):
    delta = {name: value[name] - reference[name] for name in value}
    return .5 * dot(delta, metric.product(delta))
