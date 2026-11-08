from __future__ import annotations
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruning.evaluation import language_losses
from pruning.models.base import TorchProblem
from pruning.data import calibration_tokens, selection_sequences, wikitext_test_tokens, official_test_tokens


def linear_layers(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: child for name, child in module.named_modules()
            if isinstance(child, nn.Linear)}


class StopCapture(Exception):
    pass


def selected_names(layer, selection):
    names = tuple(linear_layers(layer))
    if selection == "all":
        return names
    requested = tuple(value.strip() for value in selection.split(",") if value.strip())
    missing = set(requested) - set(names)
    if missing:
        raise ValueError(f"unknown linear modules: {sorted(missing)}")
    return requested


@torch.inference_mode()
def capture_first_inputs(model, sequences):
    layers = model.model.decoder.layers
    original = layers[0]
    hidden, kwargs = [], []
    device = next(model.parameters()).device

    class Catcher(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, values, **arguments):
            hidden.append(values.detach().cpu())
            kwargs.append({key: (value.detach().cpu() if torch.is_tensor(value) else value)
                           for key, value in arguments.items()})
            raise StopCapture

    layers[0] = Catcher(original)
    try:
        for sequence in sequences:
            try:
                model(sequence.to(device), use_cache=False)
            except StopCapture:
                pass
    finally:
        layers[0] = original
    return torch.cat(hidden), kwargs


def forward_layer(layer, hidden, kwargs, batch_size=1):
    """Replay OPT windows with shared head masks and cache positions."""
    device = next(layer.parameters()).device
    output = torch.empty_like(hidden)
    with torch.inference_mode():
        batches = DataLoader(TensorDataset(hidden, torch.arange(len(hidden))), batch_size=batch_size)
        for values, indices in batches:
            group = [kwargs[index] for index in indices.tolist()]
            arguments = {}
            for key, value in group[0].items():
                if torch.is_tensor(value):
                    if key in {"attention_mask", "position_ids"}:
                        value = torch.cat([item[key] for item in group])
                    value = value.to(device)
                arguments[key] = value
            result = layer(values.to(device), **arguments)[0]
            output[indices] = result.cpu()
    return output


def parameter_count(model, layers_count, names):
    return sum(module.weight.numel() +
               (module.bias.numel() if module.bias is not None else 0)
               for layer in model.model.decoder.layers[:layers_count]
               for name, module in linear_layers(layer).items() if name in names)


def nonzero_count(model, layers_count, names):
    return sum(int(torch.count_nonzero(module.weight)) +
               (int(torch.count_nonzero(module.bias)) if module.bias is not None else 0)
               for layer in model.model.decoder.layers[:layers_count]
               for name, module in linear_layers(layer).items() if name in names)


def neuron_size(layer):
    return layer.fc1.in_features + int(layer.fc1.bias is not None) + layer.fc2.out_features


def neuron_capacity(model, layers_count):
    return sum(layer.fc1.out_features * neuron_size(layer)
               for layer in model.model.decoder.layers[:layers_count])


def neuron_removed_count(model, deleted):
    return sum(int(deleted[index].sum()) * neuron_size(layer)
               for index, layer in enumerate(model.model.decoder.layers[:len(deleted)]))


def augmented_linear(module):
    return torch.cat((module.weight.detach(), module.bias.detach()[:, None]), 1)


@torch.no_grad()
def copy_augmented(module, values):
    module.weight.copy_(values[:, :-1].to(module.weight))
    module.bias.copy_(values[:, -1].to(module.bias))


class OptProblem(TorchProblem):
    def __init__(self, cfg, options):
        self.cfg, self.sampling, self.options = cfg, cfg.sampling, options
        self.device = torch.device(cfg.runtime.device)
        if cfg.benchmark.name == "opt_cpu" and self.device.type != "cpu":
            raise ValueError("opt_cpu retains the CPU baseline protocol")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.benchmark.model, use_fast=cfg.benchmark.get("tokenizer_fast", False))
        super().__init__(AutoModelForCausalLM.from_pretrained(
            cfg.benchmark.model, dtype=cfg.benchmark.dtype).to(self.device).eval())
        self.model.config.use_cache = False
        self.layers_count = min(cfg.benchmark.max_layers, len(self.model.model.decoder.layers))
        self.names = selected_names(self.model.model.decoder.layers[0], options.get("modules", "all"))
        self.accounting_names = selected_names(self.model.model.decoder.layers[0],
                                              options.get("accounting_modules", options.get("modules", "all")))
        if not set(self.names) <= set(self.accounting_names):
            raise ValueError("accounting_modules must include every pruned module")
        self.total = parameter_count(self.model, self.layers_count, self.accounting_names)
        self.initial_nonzero = nonzero_count(self.model, self.layers_count, self.names)
        self.neurons = cfg.branch.name == "continuous" and options.deletion_unit == "hidden_neuron"
        if self.neurons:
            self.extra["deleted_neurons"] = {
                index: torch.zeros(layer.fc1.out_features, dtype=torch.bool, device=self.device)
                for index, layer in enumerate(self.model.model.decoder.layers[:self.layers_count])}
        sampling = self.sampling
        for context in (sampling.context, sampling.b_context, sampling.selection_context, sampling.heldout_context):
            if not 2 <= context <= self.model.config.max_position_embeddings:
                raise ValueError("context length is outside the model's position limits")
        if sampling.heldout_sequences < 2 or (cfg.protocol.selection_check and sampling.selection_sequences < 2):
            raise ValueError("evaluation and enabled selection checks require at least two sequences")
        self.splits, self.metadata = {}, {}
        candidate_data = cfg.benchmark.get("selection_dataset", sampling.dataset)
        for name, count, context, seed, dataset, offset in (
            ("selection", sampling.candidate_sequences, sampling.selection_context,
             sampling.candidate_seed, candidate_data, cfg.benchmark.get("candidate_offset", 0)),
            ("validation", sampling.selection_sequences if cfg.protocol.selection_check else 0,
             sampling.selection_context, sampling.selection_seed, candidate_data,
             cfg.benchmark.get("selection_offset", sampling.candidate_sequences if cfg.branch.name == "classic" else 0)),
            ("reporting", sampling.heldout_sequences, sampling.heldout_context,
             sampling.heldout_seed, cfg.benchmark.get("reporting_dataset", sampling.dataset), 0)):
            if name == "selection" and cfg.branch.name != "classic":
                count = 0
            if count and dataset.endswith("_test"):
                corpus = dataset.removesuffix("_test")
                ids = (wikitext_test_tokens(self.tokenizer, cfg.benchmark.get("evaluation_tokens", 32768))
                       if corpus == "wikitext2" else official_test_tokens(self.tokenizer, corpus, context))
                count = ids.shape[1] // context
                self.splits[name] = list(ids[:, :count * context].reshape(count, context).unsqueeze(1))
            else:
                self.splits[name] = (selection_sequences(self.tokenizer, dataset, context, offset, count, seed)
                                     if count else None)
            self.metadata[name] = {"dataset": dataset, "seed": seed, "count": count,
                                   "seqlen": context, "offset": offset}
        self.calibration = self.b_sequences = None
        self.settings = OmegaConf.merge(OmegaConf.to_container(sampling, resolve=True),
                                        OmegaConf.to_container(options, resolve=True))
        self.settings.seed = cfg.benchmark.seed
        self.settings.b_tokens = (sampling.b_tokens or sampling.b_sequences * (sampling.b_context - 1))

    @property
    def deleted_neurons(self):
        return self.extra["deleted_neurons"]

    def remaining(self):
        if self.neurons:
            return self.total - neuron_removed_count(self.model, self.deleted_neurons)
        return self.total - (self.initial_nonzero - nonzero_count(self.model, self.layers_count, self.names))

    def losses(self, split):
        return language_losses(self.model, self.splits[split], self.sampling.evaluation_batch_size)

    def split_metadata(self, split):
        return dict(self.metadata[split])

    def prepare(self, index):
        sampling = self.sampling
        seed = sampling.seed + index if sampling.refresh == "step" else sampling.seed
        started = time.perf_counter()
        redraw = self.calibration is None or sampling.refresh == "step"
        if redraw:
            needs_b = self.cfg.branch.name == "continuous"
            count = max(sampling.calibration_sequences, sampling.b_sequences) if needs_b else sampling.calibration_sequences
            context = max(sampling.context, sampling.b_context) if needs_b else sampling.context
            pool = calibration_tokens(self.tokenizer, sampling.dataset, count, context, seed)
            self.calibration = [value[:, :sampling.context] for value in pool[:sampling.calibration_sequences]]
            self.b_sequences = [value[:, :sampling.b_context] for value in pool[:sampling.b_sequences]]
        return {"calibration_seed": seed, "calibration_sampling_seconds": time.perf_counter() - started}
