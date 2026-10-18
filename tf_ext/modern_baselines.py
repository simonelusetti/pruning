"""Faithful toy-GPT adaptations of SparseGPT, Wanda, and FLAP.

Reference implementations inspected when this adapter was written:

* SparseGPT: https://github.com/IST-DASLab/sparsegpt, commit 147d2159
* Wanda: https://github.com/locuslab/wanda, commit 8e8fc87b
* FLAP: https://github.com/CASIA-LMC-Lab/FLAP, commit 3bb57db3

The reference repositories operate on ``nn.Linear`` modules in Hugging Face
models.  This project stores each attention projection as one tensor per head,
so ``_linear_views`` reshapes those tensors into the same [output, input]
matrices before applying the reference algorithms.  Only the six transformer
block maps are eligible: Q, K, V, O, MLP-in, and MLP-out.  Embeddings and the LM
head are excluded, matching the official pruning drivers.

All methods start from the exact Stage-1 GPT state.  Already-deleted coordinates
are removed from each compact matrix before pruning; this applies each baseline
to the quotient network rather than charging it for Stage-1 zeros.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nnprune.core import (ACTIVE, Move, _anchored_columns, _row,
                          _within_tolerance)


@dataclass
class LinearView:
    """A GPT tensor exposed as the [output, input] matrix seen by a pruner."""

    name: str
    layer: int
    weight: torch.Tensor
    mask: torch.Tensor
    inputs: torch.Tensor
    write: object


def _calibration_cache(adapter, state, X):
    P = adapter.effective(state)
    _, cache = adapter._forward(P, state.struct, X, cache=True)
    return P, cache


def _linear_views(adapter, state, layer, X) -> list[LinearView]:
    """Return the six block linear maps and their current calibration inputs.

    The cache is recomputed after every completed transformer block by Wanda and
    SparseGPT, as in the official layerwise drivers.  Within one block all six
    maps use activations captured before that block is pruned.
    """
    P, c = _calibration_cache(adapter, state, X)
    l, H, dh, d = layer, adapter.H, adapter.dh, adapter.d
    n1 = (c["x"][2 * l] / c["s"][2 * l]).reshape(-1, d)
    n2 = (c["x"][2 * l + 1] / c["s"][2 * l + 1]).reshape(-1, d)

    vg = state.struct["v_group"][l]
    v = torch.einsum("btd,hde->bhte", c["x"][2 * l] / c["s"][2 * l], P["W_V"][l][vg])
    v = v + P["pV"][l][vg][None, :, None] / c["s"][2 * l][:, None]
    attended = torch.einsum("bhts,bhse->bhte", c["A"][l], v)
    o_in = attended.permute(0, 2, 1, 3).reshape(-1, H * dh)
    a = c["act"][l].reshape(-1, adapter.dm)

    def qkv(name):
        W = P[name][l].permute(0, 2, 1).reshape(H * dh, d)
        M = state.masks[name][l].permute(0, 2, 1).reshape(H * dh, d)

        def write(s, value, mask):
            s.params[name][l] = value.reshape(H, dh, d).permute(0, 2, 1).contiguous()
            s.masks[name][l] = mask.reshape(H, dh, d).permute(0, 2, 1).contiguous()

        return LinearView(name, l, W, M, n1, write)

    def out_proj():
        W = P["W_O"][l].reshape(H * dh, d).T
        M = state.masks["W_O"][l].reshape(H * dh, d).T

        def write(s, value, mask):
            s.params["W_O"][l] = value.T.reshape(H, dh, d).contiguous()
            s.masks["W_O"][l] = mask.T.reshape(H, dh, d).contiguous()

        return LinearView("W_O", l, W, M, o_in, write)

    def direct(name, inputs):
        W, M = P[name][l], state.masks[name][l]

        def write(s, value, mask):
            s.params[name][l] = value.contiguous()
            s.masks[name][l] = mask.contiguous()

        return LinearView(name, l, W, M, inputs, write)

    return [qkv("W_Q"), qkv("W_K"), qkv("W_V"), out_proj(),
            direct("W_1", n2), direct("W_2", a)]


def _compact(view: LinearView):
    """Extract the dense live rectangle of a Stage-1 linear map."""
    rows = view.mask.any(1).nonzero().flatten()
    cols = view.mask.any(0).nonzero().flatten()
    live = view.mask[rows[:, None], cols[None, :]]
    if not bool(live.all()):
        raise ValueError(f"{view.name}[{view.layer}] no longer has a dense live rectangle")
    return (view.weight[rows[:, None], cols[None, :]].clone(),
            view.inputs[:, cols].clone(), rows, cols)


@torch.no_grad()
def wanda_matrix(weight: torch.Tensor, inputs: torch.Tensor, sparsity: float):
    """Official unstructured Wanda rule: prune a fraction within every row."""
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    n_prune = int(weight.shape[1] * sparsity)
    mask = torch.zeros_like(weight, dtype=torch.bool)
    if n_prune:
        scaler = inputs.float().pow(2).sum(0)
        metric = weight.abs().float() * torch.sqrt(scaler)[None, :]
        order = torch.argsort(metric, dim=1, stable=True)
        mask.scatter_(1, order[:, :n_prune], True)
    out = weight.clone()
    out[mask] = 0
    return out, mask


@torch.no_grad()
def sparsegpt_matrix(weight: torch.Tensor, inputs: torch.Tensor, sparsity: float,
                     percdamp: float = 0.01, blocksize: int = 128):
    """SparseGPT's official unstructured reconstruction kernel.

    This is the no-quantization, unstructured branch of ``SparseGPT.fasterprune``.
    The Hessian's overall scale is immaterial to the reconstruction update, so
    it is formed directly as X^T X before applying the reference 1% damping.
    """
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if sparsity == 0:
        return weight.clone(), torch.zeros_like(weight, dtype=torch.bool)

    original_dtype = weight.dtype
    # The reference kernel explicitly promotes weights and calibration inputs
    # to float32, even when the model itself is stored in a lower precision.
    W = weight.float().clone()
    X = inputs.float()
    H = X.T @ X
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    diag = torch.arange(H.shape[0], device=H.device)
    damp = percdamp * torch.diag(H).mean()
    H[diag, diag] += damp
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)

    pruned = torch.zeros_like(W, dtype=torch.bool)
    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1 + blocksize, W.shape[1])
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        metric = W1.pow(2) / torch.diag(Hinv1).reshape(1, -1).pow(2)
        rank = min(int(metric.numel() * sparsity), metric.numel() - 1)
        threshold = torch.sort(metric.flatten()).values[rank]
        mask1 = metric <= threshold

        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err[:, None] @ Hinv1[i, i:][None, :]
            Err1[:, i] = err

        W[:, i1:i2] = Q1
        pruned[:, i1:i2] = mask1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    pruned[:, dead] = True
    W[pruned] = 0
    return W.to(original_dtype), pruned


def _write_compact(state, view, compact_weight, compact_pruned, rows, cols):
    W, M = view.weight.clone(), view.mask.clone()
    W[rows[:, None], cols[None, :]] = compact_weight
    submask = M[rows[:, None], cols[None, :]]
    submask[compact_pruned] = False
    M[rows[:, None], cols[None, :]] = submask
    W[~M] = 0
    view.write(state, W, M)


@torch.no_grad()
def weight_prune_state(adapter, state0, method: str, sparsity: float, X,
                       percdamp: float = 0.01, blocksize: int = 128):
    """Run official Wanda or SparseGPT layerwise on a fresh Stage-1 state."""
    if method not in ("wanda", "sparsegpt"):
        raise ValueError(method)
    state = state0.clone()
    for l in range(adapter.L):
        views = _linear_views(adapter, state, l, X)
        for view in views:
            W, inp, rows, cols = _compact(view)
            if method == "wanda":
                Q, mask = wanda_matrix(W, inp, sparsity)
            else:
                Q, mask = sparsegpt_matrix(W, inp, sparsity, percdamp, blocksize)
            _write_compact(state, view, Q, mask, rows, cols)
        adapter.canonicalize(state)
    return state


@torch.no_grad()
def flap_statistics(adapter, state, X):
    """FLAP WIFV/AL-AM scores and compensation means on the unpruned model."""
    P, c = _calibration_cache(adapter, state, X)
    heads, units = [], []
    for l in range(adapter.L):
        n1 = c["x"][2 * l] / c["s"][2 * l]
        vg = state.struct["v_group"][l]
        v = torch.einsum("btd,hde->bhte", n1, P["W_V"][l][vg])
        v = v + P["pV"][l][vg][None, :, None] / c["s"][2 * l][:, None]
        attended = torch.einsum("bhts,bhse->bhte", c["A"][l], v)

        # Official WIFV: input fluctuation times squared column norm.  For
        # attention FLAP squares feature scores, standardizes all features in a
        # layer, then averages the features belonging to each head.
        av_seq = attended.permute(0, 2, 1, 3).reshape(len(X), -1, adapter.H * adapter.dh).double()
        av_mean, av_fluc = _flap_input_stats(av_seq)
        Wo = P["W_O"][l].reshape(adapter.H * adapter.dh, adapter.d).T.double()
        feature = (av_fluc * Wo.pow(2).sum(0)).pow(2)
        feature = (feature - feature.mean()) / feature.std(unbiased=True).clamp_min(1e-30)
        hscore = feature.reshape(adapter.H, adapter.dh).mean(1)

        act_seq = c["act"][l].reshape(len(X), -1, adapter.dm).double()
        act_mean, act_fluc = _flap_input_stats(act_seq)
        W2 = P["W_2"][l].double()
        uscore = act_fluc * W2.pow(2).sum(0)
        uscore = (uscore - uscore.mean()) / uscore.std(unbiased=True).clamp_min(1e-30)

        for h in range(adapter.H):
            if bool(state.struct["head_alive"][l, h]):
                cost = _head_cost(state, l, h)
                heads.append({"kind": "head", "l": l, "i": h,
                              "score": float(hscore[h]), "cost": cost,
                              "mean": (av_mean.reshape(adapter.H, adapter.dh)[h]
                                       @ P["W_O"][l, h].double()).to(P["W_O"].dtype)})
        for i in range(adapter.dm):
            if int(state.struct["unit_type"][l, i]) == ACTIVE:
                cost = _unit_cost(state, l, i)
                units.append({"kind": "unit", "l": l, "i": i,
                              "score": float(uscore[i]), "cost": cost,
                              "mean": float(act_mean[i])})
    return heads + units


@torch.no_grad()
def _flap_input_stats(inputs: torch.Tensor):
    """Reproduce FLAP ``BiasGPT.add_batch`` for [sample, token, feature].

    The official driver feeds one sequence at a time.  Keeping its recurrence,
    including its precise normalization, matters because FLAP ranks fluctuation
    rather than using an ordinary variance estimator.
    """
    mean = torch.zeros(inputs.shape[-1], dtype=torch.double, device=inputs.device)
    fluc = torch.zeros_like(mean)
    nsamples = 0
    for sample in inputs.double():
        nsamples_new = nsamples + 1
        mean *= nsamples / nsamples_new
        mean += sample.mean(0) / nsamples_new
        if nsamples == 0:
            fluc.zero_()
        else:
            fluc *= (nsamples - 1) / nsamples
            # In BiasGPT, old_baseline_inp aliases the in-place updated tensor,
            # so both factors use the new running mean.
            fluc += (sample - mean).pow(2).sum(0) / nsamples_new
        nsamples = nsamples_new
    return mean, fluc


def _head_cost(state, l, h):
    return sum(int(state.masks[n][l, h].sum())
               for n in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "pQ", "pK", "pV"))


def _unit_cost(state, l, i):
    return (int(state.masks["W_1"][l, i].sum()) + int(state.masks["b_1"][l, i])
            + int(state.masks["W_2"][l, :, i].sum()))


def flap_selection(items, sparsity: float):
    """FLAP AL-AM global threshold with architecture-specific parameter costs."""
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if sparsity == 0 or not items:
        return []
    order = sorted(range(len(items)), key=lambda j: items[j]["score"], reverse=True)
    costs = torch.tensor([items[j]["cost"] for j in order], dtype=torch.double)
    target_keep = (1 - sparsity) * float(costs.sum())
    boundary = int(torch.argmin((costs.cumsum(0) - target_keep).abs()))
    threshold = items[order[boundary]]["score"]
    # The official implementation retains score > threshold.
    return [item for item in items if item["score"] <= threshold]


@torch.no_grad()
def flap_state(adapter, state0, sparsity: float, items):
    """Apply FLAP's selected head/unit removals and mean-bias compensation."""
    state = state0.clone()
    adapter._cur = None
    for item in flap_selection(items, sparsity):
        if item["kind"] == "head":
            move = Move("head_mean", f"L{item['l']}h{item['i']}", item["l"],
                        {"l": item["l"], "h": item["i"], "mean": item["mean"]})
        else:
            move = Move("unit_mean", f"mlp{item['l']}", item["l"],
                        {"l": item["l"], "i": item["i"], "mean": item["mean"]})
        if not adapter.apply_structural(state, move):
            raise RuntimeError(f"FLAP move became invalid: {move.describe()}")
    adapter.canonicalize(state)
    return state


def modern_baseline_path(adapter, state0, method: str, cfg, fracs=None,
                         calib_n=None, percdamp=0.01, blocksize=128):
    """Evaluate SparseGPT, Wanda, or FLAP under the repository's common rule."""
    if method not in ("wanda", "sparsegpt", "flap"):
        raise ValueError(method)
    if fracs is None:
        fracs = np.round(np.concatenate([np.arange(0, 0.5, 0.02),
                                         np.arange(0.5, 0.96, 0.05)]), 4)
    default_n = 2048 if method == "flap" else 128
    n = min(calib_n or default_n, len(adapter.data["train"][0]))
    X = adapter.data["train"][0][:n]
    items = flap_statistics(adapter, state0, X) if method == "flap" else None
    ref = {c: adapter.per_sample(state0, cfg.stop, c) for c in ("loss", "error")}
    k0 = adapter.k_eff(state0)
    rows = []
    for f in fracs:
        if method == "flap":
            state = flap_state(adapter, state0, float(f), items)
        else:
            state = weight_prune_state(adapter, state0, method, float(f), X,
                                       percdamp=percdamp, blocksize=blocksize)
        rows.append(_row(adapter, state, k0, method=method, target_sparsity=float(f),
                         calib_n=n, percdamp=(percdamp if method == "sparsegpt" else np.nan),
                         blocksize=(blocksize if method == "sparsegpt" else np.nan),
                         **_anchored_columns(adapter, state, ref, cfg)))
    return _within_tolerance(pd.DataFrame(rows))
