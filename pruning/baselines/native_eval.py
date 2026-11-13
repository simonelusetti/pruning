"""Run inside a native program's own environment: load its output and score our token windows.

Kept compatible with Python 3.9, torch 1.10, and transformers 4.28, the oldest pinned environment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def saved_state(path):
    state = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        from safetensors.torch import load_file
        state.update(load_file(str(file)))
    for file in [] if state else sorted(Path(path).glob("pytorch_model*.bin")):
        state.update(torch.load(file, map_location="cpu"))
    return state


def resized_model(path, dtype):
    """Rebuild a saved model whose linear layers may be smaller or carry new biases."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights
    state = saved_state(path)
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), torch_dtype=dtype)
    for name, module in list(model.named_modules()):
        weight = state.get(name + ".weight")
        has_bias = name + ".bias" in state
        if isinstance(module, torch.nn.Linear) and weight is not None and (
                weight.shape != module.weight.shape or has_bias != (module.bias is not None)):
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child,
                    torch.nn.Linear(weight.shape[1], weight.shape[0], bias=has_bias, dtype=dtype))
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "head_dim"):
            module.num_heads = module.q_proj.out_features // module.head_dim
            module.num_key_value_heads = module.k_proj.out_features // module.head_dim
            module.hidden_size = module.q_proj.out_features
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if "rotary_emb" not in key and key != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"saved model does not match its rebuilt shapes: {missing[:5]} {unexpected[:5]}")
    model.tie_weights()
    return model


def load(args, dtype):
    from transformers import AutoModelForCausalLM
    if args.kind == "dense":
        return AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache, torch_dtype=dtype)
    if args.kind == "llm_pruner":
        return torch.load(next(Path(args.path).rglob("pytorch_model.bin")), map_location="cpu")["model"]
    if args.kind == "slicegpt":
        from slicegpt import hf_utils
        adapter, _ = hf_utils.load_sliced_model(args.model, args.path, sparsity=args.target, round_interval=8)
        return adapter.model
    return resized_model(args.path, dtype)


@torch.no_grad()
def losses(model, ids, batch_size, device):
    values = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size].to(device)
        logits = model(batch).logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="none")
        values.append(loss.reshape(len(batch), -1).mean(1).cpu())
    return torch.cat(values).double().tolist()


def counts(model):
    """Nonzero coefficients inside decoder layers (norms excluded) and in the whole model; ties counted once."""
    seen, pool, total, size, numel = set(), 0, 0, 0, 0
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        nonzero = int(torch.count_nonzero(parameter))
        total += nonzero
        numel += parameter.numel()
        size += parameter.numel() * parameter.element_size()
        if ".layers." in name and "norm" not in name:
            pool += nonzero
    return {"pool_nonzero": pool, "model_nonzero": total, "model_parameters": numel, "parameter_bytes": size}


def main():
    parser = argparse.ArgumentParser()
    for name in ("kind", "model", "path", "tokens", "cache", "dtype", "output"):
        parser.add_argument("--" + name)
    parser.add_argument("--target", type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    dtype, device = getattr(torch, args.dtype), torch.device("cuda:0")
    model = load(args, dtype).to(device=device, dtype=dtype).eval()
    model.config.use_cache = False
    tokens = torch.load(args.tokens)
    result = {"losses": {}, **counts(model)}
    for split in ("validation", "reporting"):
        if split in tokens:
            started = time.perf_counter()
            result["losses"][split] = losses(model, tokens[split], args.batch_size, device)
            result["seconds"] = time.perf_counter() - started
    Path(args.output).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
