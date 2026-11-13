from __future__ import annotations
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruning.evaluation import language_losses
from pruning.models.base import TorchProblem, shrink_linear, silent
from pruning.paths import MODEL_CACHE
from pruning.data import calibration_tokens, selection_sequences, wikitext_test_tokens, official_test_tokens


def linear_layers(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: child for name, child in module.named_modules()
            if isinstance(child, nn.Linear)}


def decoder_layers(model):
    return model.model.decoder.layers if model.config.model_type == "opt" else model.model.layers


def layer_prefix(model, index):
    return f"model.decoder.layers.{index}" if model.config.model_type == "opt" else f"model.layers.{index}"


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
    layers = decoder_layers(model)
    original = layers[0]
    hidden, kwargs = [], []
    device = next(model.parameters()).device

    class Catcher(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, values, **arguments):
            hidden.append(values.detach().cpu())
            kwargs.append({key: (tuple(item.detach().cpu() for item in value) if isinstance(value, tuple) else
                                 value.detach().cpu() if torch.is_tensor(value) else value)
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
                elif isinstance(value, tuple):
                    value = tuple(torch.cat([item[key][index] for item in group]).to(device) for index in range(len(value)))
                arguments[key] = value
            result = layer(values.to(device), **arguments)
            result = result[0] if isinstance(result, tuple) else result
            output[indices] = result.cpu()
    return output


def parameter_count(model, layers_count, names):
    return sum(module.weight.numel() +
               (module.bias.numel() if module.bias is not None else 0)
               for layer in decoder_layers(model)[:layers_count]
               for name, module in linear_layers(layer).items() if name in names)


def nonzero_count(model, layers_count, names):
    return sum(int(torch.count_nonzero(module.weight)) +
               (int(torch.count_nonzero(module.bias)) if module.bias is not None else 0)
               for layer in decoder_layers(model)[:layers_count]
               for name, module in linear_layers(layer).items() if name in names)


def language_splits(cfg, tokenizer, max_positions):
    """Selection, validation, and reporting token windows shared by every method on a language benchmark."""
    sampling = cfg.sampling
    for context in (sampling.context, sampling.b_context, sampling.selection_context, sampling.heldout_context):
        if not 2 <= context <= max_positions:
            raise ValueError("context length is outside the model's position limits")
    if sampling.heldout_sequences < 2 or (cfg.protocol.selection_check and sampling.selection_sequences < 2):
        raise ValueError("evaluation and enabled selection checks require at least two sequences")
    splits, metadata = {}, {}
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
            ids = (wikitext_test_tokens(tokenizer, cfg.benchmark.get("evaluation_tokens", 32768))
                   if corpus == "wikitext2" else official_test_tokens(tokenizer, corpus, context))
            count = ids.shape[1] // context
            splits[name] = list(ids[:, :count * context].reshape(count, context).unsqueeze(1))
        else:
            splits[name] = selection_sequences(tokenizer, dataset, context, offset, count, seed) if count else None
        metadata[name] = {"dataset": dataset, "seed": seed, "count": count, "seqlen": context, "offset": offset}
    return splits, metadata


class OptProblem(TorchProblem):
    def __init__(self, cfg, options):
        self.cfg, self.sampling, self.options = cfg, cfg.sampling, options
        self.device = torch.device(cfg.runtime.device)
        if cfg.benchmark.name == "opt_cpu" and self.device.type != "cpu":
            raise ValueError("opt_cpu retains the CPU baseline protocol")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.benchmark.model, cache_dir=MODEL_CACHE,
                                                       use_fast=cfg.benchmark.get("tokenizer_fast", False))
        super().__init__(AutoModelForCausalLM.from_pretrained(
            cfg.benchmark.model, cache_dir=MODEL_CACHE, dtype=cfg.benchmark.dtype).to(self.device).eval())
        self.model.config.use_cache = False
        if self.model.config.model_type not in {"opt", "llama"}:
            raise ValueError("Language pruning supports OPT and LLaMA models")
        self.snapshot_device = cfg.runtime.get("snapshot_device")
        layers = decoder_layers(self.model)
        self.layers_count = min(cfg.benchmark.max_layers, len(layers))
        self.names = selected_names(layers[0], options.get("modules", "all"))
        self.accounting_names = selected_names(layers[0],
                                              options.get("accounting_modules", options.get("modules", "all")))
        if not set(self.names) <= set(self.accounting_names):
            raise ValueError("accounting_modules must include every pruned module")
        self.total = parameter_count(self.model, self.layers_count, self.accounting_names)
        self.initial_nonzero = nonzero_count(self.model, self.layers_count, self.names)
        self.splits, self.metadata = language_splits(cfg, self.tokenizer, self.model.config.max_position_embeddings)
        sampling = self.sampling
        self.calibration = self.b_sequences = None
        self.settings = OmegaConf.merge(OmegaConf.to_container(sampling, resolve=True),
                                        OmegaConf.to_container(options, resolve=True))
        self.settings.seed = cfg.benchmark.seed
        self.settings.b_tokens = (sampling.b_tokens or sampling.b_sequences * (sampling.b_context - 1))

    def linear_chains(self):
        from pruning.nnprune.network import LinearChain
        if self.model.config.model_type == "llama":
            return []
        return [LinearChain((f"model.decoder.layers.{index}.fc1.weight", f"model.decoder.layers.{index}.fc2.weight"),
                            (f"model.decoder.layers.{index}.fc1.bias", f"model.decoder.layers.{index}.fc2.bias"),
                            (layer.activation_fn,))
                for index, layer in enumerate(self.model.model.decoder.layers[:self.layers_count])]

    def gated_groups(self):
        from pruning.nnprune.canonicalize.gated import GatedGroup
        if self.model.config.model_type != "llama":
            return []
        for layer in decoder_layers(self.model)[:self.layers_count]:
            if not bool(layer.mlp.act_fn(layer.mlp.gate_proj.weight.new_zeros(1)).eq(0).all()):
                raise ValueError("Gated canonicalization requires activation(0)=0")
        return [GatedGroup(*(f"model.layers.{index}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")))
                for index in range(self.layers_count)]

    def affine_modules(self):
        return {f"{layer_prefix(self.model, index)}.{name}": module
                for index, layer in enumerate(decoder_layers(self.model)[:self.layers_count])
                for name, module in linear_layers(layer).items() if name in self.names}

    def stage(self, name):
        """Decoder block of a module, the unit of replay and of streamed factor estimation."""
        return int(name.removeprefix(layer_prefix(self.model, 0)[:-1]).split(".")[0])

    def fisher_data(self, count, seed):
        sequences = self.calibration if count is None else self.calibration[:count]
        return torch.cat(sequences).to(self.device), None

    def remaining(self):
        if "pruned_coordinates" in self.extra:
            if self.extra.get("pruned_coordinate_convention") == "live":
                return self.masked_remaining()
            return self.total - sum(int(mask.sum()) for mask in self.extra["pruned_coordinates"].values())
        return self.total - (self.initial_nonzero - nonzero_count(self.model, self.layers_count, self.names))

    @torch.no_grad()
    def compact(self):
        """Drop MLP units whose output is identically zero, leaving smaller dense layers."""
        for layer in decoder_layers(self.model):
            block = layer if self.model.config.model_type == "opt" else layer.mlp
            names = ("fc1",) if block is layer else ("gate_proj", "up_proj")
            consumer = block.fc2 if block is layer else block.down_proj
            activation = block.activation_fn if block is layer else block.act_fn
            dead = consumer.weight.eq(0).all(0) | silent(getattr(block, names[-1]))
            if bool(activation(consumer.weight.new_zeros(1)).eq(0).all()):
                dead |= silent(getattr(block, names[0]))
            if not dead.any():
                continue
            keep = ~dead
            for name in names:
                setattr(block, name, shrink_linear(getattr(block, name), keep, 0))
            setattr(block, "fc2" if block is layer else "down_proj", shrink_linear(consumer, keep, 1))

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
            needs_b = self.cfg.branch.get("curvature") == "kfac"
            count = max(sampling.calibration_sequences, sampling.b_sequences) if needs_b else sampling.calibration_sequences
            context = max(sampling.context, sampling.b_context) if needs_b else sampling.context
            pool = calibration_tokens(self.tokenizer, sampling.dataset, count, context, seed)
            self.calibration = [value[:, :sampling.context] for value in pool[:sampling.calibration_sequences]]
            self.b_sequences = [value[:, :sampling.b_context] for value in pool[:sampling.b_sequences]]
        return {"calibration_seed": seed, "calibration_sampling_seconds": time.perf_counter() - started}
