from __future__ import annotations
import torch
from pruning.baselines.sparsegpt import sparsegpt_matrix


def calibration_inputs(adapter, state, count):
    X = adapter.data["train"][0]
    generator = torch.Generator(device=X.device).manual_seed(7)
    X = X[torch.randperm(len(X), generator=generator, device=X.device)[:count]]
    parameters = adapter.effective(state)
    _, cache = adapter.forward_full(parameters, state.struct, X)
    d, heads, head_width = adapter.d, adapter.H, adapter.dh
    return {
        "W_Q": cache["x0"].reshape(-1, d),
        "W_K": cache["x0"].reshape(-1, d),
        "W_V": cache["x0"].reshape(-1, d),
        "W_O": cache["z"].permute(0, 2, 1, 3).reshape(-1, heads * head_width),
        "W_in": cache["x1"].reshape(-1, d),
        "W_out": cache["act"].reshape(-1, adapter.dm),
    }


def matrix_view(adapter, state, name):
    weight = adapter.effective(state)[name]
    if name in ("W_Q", "W_K", "W_V"):
        return weight.permute(0, 2, 1).reshape(adapter.H * adapter.dh, adapter.d)
    if name == "W_O":
        return weight.reshape(adapter.H * adapter.dh, adapter.d).T
    return weight


def write_matrix(adapter, state, name, weight, pruned):
    live = ~pruned
    if name in ("W_Q", "W_K", "W_V"):
        shape = (adapter.H, adapter.dh, adapter.d)
        state.params[name] = weight.reshape(shape).permute(0, 2, 1).contiguous()
        state.masks[name] = live.reshape(shape).permute(0, 2, 1).contiguous()
    elif name == "W_O":
        state.params[name] = weight.T.reshape(adapter.H, adapter.dh, adapter.d).contiguous()
        state.masks[name] = live.T.reshape(adapter.H, adapter.dh, adapter.d).contiguous()
    else:
        state.params[name] = weight.contiguous()
        state.masks[name] = live.contiguous()


@torch.no_grad()
def sparsegpt_state(adapter, initial, inputs, sparsity, percdamp=.01, blocksize=128):
    state = initial.clone()
    for name in ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"):
        weight = matrix_view(adapter, state, name)
        reconstructed, pruned = sparsegpt_matrix(
            weight, inputs[name].float().T @ inputs[name].float(), sparsity,
            percdamp=percdamp, blocksize=blocksize)
        write_matrix(adapter, state, name, reconstructed, pruned)
    adapter.canonicalize(state)
    return state


class ModaddSparseGPT:
    uses_selection = False

    def __init__(self, options):
        self.cfg, self.inputs = options, None

    def propose(self, problem, target, index, progress):
        if self.inputs is None:
            self.inputs = calibration_inputs(problem.adapter, problem.state,
                                             problem.cfg.benchmark.calibration_examples)
        problem.state = sparsegpt_state(problem.adapter, problem.state, self.inputs,
                                        target, self.cfg.percdamp, self.cfg.blocksize)
        return {"kind": "sparsegpt"}
