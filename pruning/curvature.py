from __future__ import annotations
from dataclasses import dataclass
import math
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from pruning.models.opt import linear_layers, forward_layer, capture_first_inputs, capture_first_inputs as capture_first_layer_inputs


@dataclass
class ActivationStats:
    mlp_sample: torch.Tensor
    attn_sample: torch.Tensor
    mlp_mean: torch.Tensor
    mlp_second: torch.Tensor


class SecondMoment:
    def __init__(self, width, include_bias=False, device="cpu"):
        self.matrix = torch.zeros((width, width), device=device)
        self.count = 0
        self.include_bias = include_bias

    def add(self, values):
        values = values.detach().reshape(-1, values.shape[-1]).to(self.matrix)
        if self.include_bias:
            values = torch.cat((values, values.new_ones((len(values), 1))), 1)
        self.matrix.addmm_(values.T, values)
        self.count += len(values)

    def value(self):
        return self.matrix / self.count


def collect_a(layer, hidden, kwargs, names, batch_size=1, *, return_hidden=False):
    """Collect OPT input moments, sharing the identical Q/K/V factor."""
    modules = linear_layers(layer)
    owners, groups = {}, {}
    for name in names:
        prefix, _, projection = name.rpartition(".")
        group = prefix if projection in {"q_proj", "k_proj", "v_proj"} else name
        owners[name] = groups.setdefault(group, name)
    stats = {name: SecondMoment(modules[name].in_features, device=modules[name].weight.device)
             for name in dict.fromkeys(owners.values())}
    handles = [modules[name].register_forward_pre_hook(
        lambda _module, inputs, key=name: stats[key].add(inputs[0])) for name in stats]
    try:
        output = forward_layer(layer, hidden, kwargs, batch_size)
    finally:
        for handle in handles:
            handle.remove()
    factors = {name: value.value() for name, value in stats.items()}
    factors = {name: factors[owner] for name, owner in owners.items()}
    return (factors, output) if return_hidden else factors


def estimate_b_covariance(model, sequences, names, max_tokens, seed,
                          layers_count, layer_index=None, *, layer_indices=None,
                          on_sequence=None, batch_size=1):
    """Accumulate model-score covariances without backpropagating before the first target."""
    modules = {(index, name): linear_layers(layer)[name]
               for index, layer in enumerate(model.model.decoder.layers[:layers_count])
               for name in names if (layer_index is None or index == layer_index)
               and (layer_indices is None or index in layer_indices)}
    device = next(model.parameters()).device
    covariance = {key: torch.zeros((module.out_features, module.out_features),
                                    dtype=torch.float64, device=device)
                  for key, module in modules.items()}
    counts = {key: 0 for key in modules}
    quota = max(1, math.ceil(max_tokens / len(sequences)))
    batch_shape = None

    def accumulate(key, gradient):
        gradient = gradient.detach().reshape(*batch_shape, -1)[:, :-1].float()
        if quota < gradient.shape[1]:
            positions = torch.linspace(0, gradient.shape[1] - 1, quota, device=device).long()
            gradient = gradient.index_select(1, positions)
        selected = gradient.reshape(-1, gradient.shape[-1])[:max_tokens - counts[key]].double()
        covariance[key].addmm_(selected.T, selected)
        counts[key] += len(selected)

    handles = []
    for key, module in modules.items():
        def capture(_module, _inputs, output, key=key):
            output.register_hook(lambda gradient: accumulate(key, gradient))
        handles.append(module.register_forward_hook(capture))
    required = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def start_gradient(_module, args, kwargs):
        if args:
            return (args[0].detach().requires_grad_(True), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": kwargs["hidden_states"].detach().requires_grad_(True)}

    first_layer = min(index for index, _ in modules)
    handles.append(model.model.decoder.layers[first_layer].register_forward_pre_hook(
        start_gradient, with_kwargs=True))
    generator = torch.Generator(device=device).manual_seed(seed)
    batches = DataLoader(TensorDataset(torch.cat(sequences)), batch_size=batch_size)
    started, completed = time.perf_counter(), 0
    try:
        with torch.enable_grad():
            for (ids,) in batches:
                if min(counts.values()) >= max_tokens:
                    break
                ids = ids.to(device)
                batch_shape = ids.shape
                logits = model(ids, use_cache=False).logits[:, :-1].float()
                probabilities = torch.softmax(logits.detach(), -1)
                labels = torch.multinomial(
                    probabilities.reshape(-1, probabilities.shape[-1]), 1,
                    generator=generator).reshape(logits.shape[:-1])
                # The sampled log-probability gradient is one_hot(label) - probability.
                probabilities.neg_().scatter_add_(
                    -1, labels.unsqueeze(-1), probabilities.new_ones((*labels.shape, 1)))
                logits.backward(probabilities)
                completed += len(ids)
                if on_sequence is not None:
                    on_sequence(completed, covariance, counts, time.perf_counter() - started)
    finally:
        for handle in handles:
            handle.remove()
        for parameter, value in zip(model.parameters(), required):
            parameter.requires_grad_(value)
    return covariance, counts, time.perf_counter() - started


def inverse_factors(covariance, counts, damping_ratio):
    factors = {}
    for key in tuple(covariance):
        factor = covariance.pop(key)
        factor.div_(counts[key])
        damping = damping_ratio * max(
            float(factor.diag().mean()), torch.finfo(factor.dtype).eps)
        factor.diagonal().add_(damping)
        cholesky, info = torch.linalg.cholesky_ex(factor)
        if int(info):
            raise RuntimeError(f"B factor is not positive definite: {key}")
        factors[key] = (factor, torch.cholesky_inverse(cholesky))
    return factors


@torch.inference_mode()
def activation_samples(model, sequences, max_tokens: int, batch_size: int):
    """Summarise every calibration sequence and retain a distributed sample.

    MLP means and second moments use every token.  Correlation-heavy merge and
    attention proposals use at most ``max_tokens`` positions, sampled evenly
    from every calibration document instead of taking only the first documents.
    """
    layers = model.model.decoder.layers
    mlp, attn = [[] for _ in layers], [[] for _ in layers]
    mlp_sum = [None for _ in layers]
    mlp_sumsq = [None for _ in layers]
    mlp_count = [0 for _ in layers]
    quota = max(1, math.ceil(max_tokens / len(sequences)))
    handles = []

    def distributed_sample(value, have):
        batch = value.shape[0] if value.ndim > 2 else 1
        value = value.detach().reshape(batch, -1, value.shape[-1]).float().cpu()
        take = min(quota, value.shape[1])
        positions = torch.linspace(0, value.shape[1] - 1, take).long()
        return value[:, positions].reshape(-1, value.shape[-1])[:max_tokens - have]

    for li, layer in enumerate(layers):
        def capture_mlp(_module, args, index=li):
            nonlocal mlp_sum, mlp_sumsq, mlp_count
            x_all = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu()
            total = x_all.sum(0)
            total_sq = x_all.square().sum(0)
            mlp_sum[index] = total if mlp_sum[index] is None else mlp_sum[index] + total
            mlp_sumsq[index] = (total_sq if mlp_sumsq[index] is None
                                else mlp_sumsq[index] + total_sq)
            mlp_count[index] += x_all.shape[0]
            have = sum(x.shape[0] for x in mlp[index])
            if have < max_tokens:
                mlp[index].append(distributed_sample(args[0], have))

        def capture_attn(_module, args, index=li):
            have = sum(x.shape[0] for x in attn[index])
            if have < max_tokens:
                attn[index].append(distributed_sample(args[0], have))

        handles.append(layer.fc2.register_forward_pre_hook(capture_mlp))
        handles.append(layer.self_attn.out_proj.register_forward_pre_hook(capture_attn))
    try:
        device = next(model.parameters()).device
        dataset = TensorDataset(torch.cat(sequences, 0))
        for (ids,) in DataLoader(dataset, batch_size=batch_size):
            model.model.decoder(ids.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [ActivationStats(
        torch.cat(mlp[i], 0), torch.cat(attn[i], 0),
        mlp_sum[i] / mlp_count[i], mlp_sumsq[i] / mlp_count[i])
        for i in range(len(layers))]


@torch.no_grad()
def neuron_factor_snapshot(model, calibration, b_calibration, layers_count, cfg, on_sequence=None):
    names = ("fc1", "fc2")
    device = next(model.parameters()).device
    covariance, counts, b_seconds = estimate_b_covariance(
        model, b_calibration, names, cfg.b_tokens, cfg.seed, layers_count,
        batch_size=cfg.b_batch_size, on_sequence=on_sequence)
    print(f"Hidden-neuron B snapshot: {b_seconds:.1f}s, "
          f"{min(counts.values())} observations", flush=True)
    hidden, kwargs = capture_first_inputs(model, calibration)
    snapshots = []
    for index, layer in enumerate(model.model.decoder.layers[:layers_count]):
        started = time.perf_counter()
        if (layer.fc1.bias is None or layer.fc2.bias is None or
                layer.fc1.out_features != layer.fc2.in_features or
                bool((layer.activation_fn(layer.fc1.weight.new_zeros(1)) != 0).any())):
            raise ValueError("hidden-neuron cleanup requires an OPT MLP with biases and activation(0)=0")
        stats = {name: SecondMoment(getattr(layer, name).in_features + 1,
                                    include_bias=True, device=device) for name in names}
        probe = []

        def capture_probe(_module, inputs):
            if not probe:
                probe.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:32].clone())

        handles = [getattr(layer, name).register_forward_pre_hook(
            lambda _module, inputs, key=name: stats[key].add(inputs[0])) for name in names]
        handles.append(layer.fc1.register_forward_pre_hook(capture_probe))
        try:
            hidden = forward_layer(layer, hidden, kwargs, cfg.activation_batch_size)
        finally:
            for handle in handles:
                handle.remove()
        a, b = {}, {}
        for name in names:
            factor = stats[name].value().double()
            factor = (factor + factor.T) / 2
            factor.diagonal().add_(cfg.percdamp * factor.diag().mean())
            a[name] = factor
            factor = covariance.pop((index, name)).to(device) / counts[index, name]
            factor = (factor + factor.T) / 2
            factor.diagonal().add_(cfg.b_damp_ratio * max(
                float(factor.diag().mean()), torch.finfo(factor.dtype).eps))
            b[name] = factor
        snapshots.append({"a": a, "b": b, "probe": probe[0],
                          "a_estimation_seconds": time.perf_counter() - started})
        print(f"Hidden-neuron A snapshot: block {index + 1}/{layers_count}", flush=True)
    return snapshots, counts, b_seconds
