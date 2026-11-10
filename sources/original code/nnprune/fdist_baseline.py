"""
Berman, Fu, Hirst, Obirai (2026) Fisher-distance pruning as a baseline.

Paper: "Optimal Pruning for Neural Architectures using Fisher Information
Distances", arXiv:2609.16129.  Code: https://github.com/edhirst/Fdist_Pruning.
The Fisher routines are their code, vendored verbatim in third_party/fdist_fim.py.
The score formulas and the pruning protocol below mirror their pruners
(src/pruning/f_dist_*.py) line by line; tests/test_fdist.py checks this
against their own pruner classes when their repository is available.

Schemes (their names):
  fdist_one_shot   s_k = sqrt(F_kk(theta*)) |theta_k|; every sparsity level is
                   re-pruned from the dense model (metric fixed at theta*).
  fdist_iterative  s_k = sqrt(F_kk(theta)) |theta_k|; one running model, the
                   Fisher is recomputed on the current pruned model each step.
  fdist_global     s_k = |theta_k| mean_alpha sqrt(F_kk(alpha theta)),
                   alpha = linspace(1, 1/K, K); running model.
  fdist_exact      s_k = sqrt(mean_alpha F_kk(theta | theta_k -> alpha theta_k)) |theta_k|,
                   alpha = linspace(1, 0, K), via their forward-mode kernel; running model.

F is their MODEL Fisher (`calculate_fim_backprop`: expectation over the model's own
predictive distribution), on a 500-point training subset drawn with seed 42 and
batched by 128 (their defaults: fim_subset_size, fim_subset_seed, batch_size).
K = 3 (their f_dist_avg_points).

Their protocol: global, unstructured, no retraining; eligible = every parameter
for an MLP, every parameter except the positional embedding for a transformer
(their `prunable_exclude` defaults); a parameter is active iff it is nonzero.

What we add is only the EVALUATION, identical to every other baseline here: the
anchored paired 2 sigma tests on the validation split and k_eff after the
structural cascades.  Cascades are applied to a COPY used for evaluation; the
running model is never canonicalised, so their procedure is untouched.

A Bernoulli model (the moons MLP, one logit z) is presented to their code as
the two-class logits (0, z).  Its softmax is (1 - sigmoid(z), sigmoid(z)), so the
model Fisher is exactly p(1-p)(dz/dtheta)^2.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .core import Adapter, PruneConfig, State, _anchored_columns, _row, _within_tolerance, zero_entries
from .third_party.fdist_fim import calculate_fim_backprop, fisher_entries_forward

SCHEMES = ("fdist_one_shot", "fdist_iterative", "fdist_global", "fdist_exact")


class AdapterModule(nn.Module):
    """nn.Module view of an adapter state, so their functional_call-based Fisher
    code runs unchanged.  Every parameter of the state is a module parameter
    except fold targets that are not part of the trained model (B_attn)."""

    def __init__(self, adapter: Adapter, state: State, exclude=("B_attn",)):
        super().__init__()
        self._adapter = adapter
        self._struct = state.struct
        eff = adapter.effective(state)
        self._fixed = {k: v.detach() for k, v in eff.items() if k in exclude}
        self._names = [k for k in eff if k not in exclude]
        for k in self._names:
            self.register_parameter(k, nn.Parameter(eff[k].detach().clone()))

    def forward(self, x):
        P = dict(self._fixed)
        for k in self._names:
            P[k] = getattr(self, k)
        z = self._adapter.logits(P, self._struct, x)
        if self._adapter.loss_type == "bernoulli":
            z = torch.stack([torch.zeros_like(z), z], -1)
        return z


def fisher_loader(adapter: Adapter, subset_size=500, seed=42, batch_size=128):
    X, y = adapter.data["train"]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:subset_size]
    Xs, ys = X[idx], y[idx]
    return [(Xs[i:i + batch_size], ys[i:i + batch_size]) for i in range(0, Xs.shape[0], batch_size)]


def _fisher_dict(module: AdapterModule, loader) -> dict:
    flat = calculate_fim_backprop(module, loader)
    out, off = {}, 0
    for name, p in module.named_parameters():
        n = p.numel()
        out[name] = flat[off:off + n].view_as(p).detach()
        off += n
    return out


def _module(adapter, state):
    return AdapterModule(adapter, state)


def _sqrtF_at(adapter, state, loader) -> dict:
    F = _fisher_dict(_module(adapter, state), loader)
    return {k: torch.sqrt(torch.clamp(v, min=0.0)) for k, v in F.items()}


def _sqrtF_global(adapter, state, loader, K) -> dict:
    """mean_alpha sqrt(F(alpha theta)), alpha = linspace(1, 1/K, K), ALL parameters scaled."""
    acc = None
    for a in torch.linspace(1.0, 1.0 / K, steps=K).tolist():
        s = state.clone()
        if abs(a - 1.0) > 1e-12:
            for k in s.params:
                s.params[k] = s.params[k] * a
        sq = _sqrtF_at(adapter, s, loader)
        acc = sq if acc is None else {k: acc[k] + sq[k] for k in acc}
    return {k: v / float(K) for k, v in acc.items()}


def _sqrtF_exact(adapter, state, loader, K, probe_batch=32) -> dict:
    """sqrt of the per-coordinate path average, alpha = linspace(1, 0, K); the
    alpha = 1 term from one reverse pass, the others from their JVP kernel."""
    mod = _module(adapter, state)
    F_base = _fisher_dict(mod, loader)
    alphas = [a for a in torch.linspace(1.0, 0.0, steps=K).tolist() if abs(a - 1.0) >= 1e-12]
    out = {}
    for name, p in mod.named_parameters():
        Fa = F_base[name].reshape(-1).clone()
        nz = (p.detach().reshape(-1) != 0).nonzero().view(-1)
        if nz.numel():
            acc = Fa[nz].clone()
            for a in alphas:
                acc += fisher_entries_forward(mod, loader, name, nz,
                                              torch.full((nz.numel(),), float(a)),
                                              probe_batch=probe_batch)
            Fa[nz] = acc / float(K)
        out[name] = torch.sqrt(torch.clamp(Fa, min=0.0)).view_as(p)
    return out


def _eligible(adapter, state, exclude) -> list:
    return [k for k in state.params if k not in exclude and k != "B_attn"]


def _prune_lowest(state: State, scores: dict, names: list, k: int) -> int:
    """Zero the k lowest-scored ACTIVE eligible entries (their topk-smallest rule)."""
    flat_s, flat_n, flat_i = [], [], []
    for j, name in enumerate(names):
        act = (state.params[name] != 0).reshape(-1) & state.masks[name].reshape(-1)
        idx = act.nonzero().view(-1)
        flat_s.append(scores[name].reshape(-1)[idx])
        flat_n.append(torch.full_like(idx, j))
        flat_i.append(idx)
    s = torch.cat(flat_s)
    k = min(int(k), s.numel())
    if k <= 0:
        return 0
    sel = torch.topk(s, k=k, largest=False).indices
    fn, fi = torch.cat(flat_n)[sel], torch.cat(flat_i)[sel]
    for j, name in enumerate(names):
        idx = fi[fn == j]
        if idx.numel():
            zero_entries(state, name, idx)
    return k


def fdist_path(adapter: Adapter, state0: State, scheme: str, cfg: PruneConfig, fracs=None,
               exclude=(), K: int = 3, subset_size=500, subset_seed=42,
               probe_batch=32, verbose=False) -> pd.DataFrame:
    """Their scheme on a sparsity grid, evaluated like every other baseline.
    `fracs` are fractions of ALL trained parameters (their ratio semantics)."""
    if scheme not in SCHEMES:
        raise ValueError(scheme)
    if fracs is None:
        fracs = np.round(np.concatenate([np.arange(0, 0.5, 0.02), np.arange(0.5, 0.99, 0.01)]), 4)
    loader = fisher_loader(adapter, subset_size, subset_seed)
    names = _eligible(adapter, state0, exclude)
    total = sum(int(state0.params[k].numel()) for k in state0.params if k != "B_attn")
    ref = {c: adapter.per_sample(state0, cfg.stop, c) for c in ("loss", "error")}
    k0 = adapter.k_eff(state0)
    rows, t0 = [], time.time()

    if scheme == "fdist_one_shot":
        sq = _sqrtF_at(adapter, state0, loader)
        scores = {k: sq[k] * state0.params[k].abs() for k in names}
    running, pruned = state0.clone(), 0

    for f in fracs:
        target = int(round(float(f) * total))
        if scheme == "fdist_one_shot":
            raw = state0.clone()
            _prune_lowest(raw, scores, names, target)
        else:
            delta = target - pruned
            if delta > 0:
                if scheme == "fdist_iterative":
                    sq = _sqrtF_at(adapter, running, loader)
                elif scheme == "fdist_global":
                    sq = _sqrtF_global(adapter, running, loader, K)
                else:
                    sq = _sqrtF_exact(adapter, running, loader, K, probe_batch)
                sc = {k: sq[k] * running.params[k].abs() for k in names}
                pruned += _prune_lowest(running, sc, names, delta)
            raw = running
        ev = raw.clone()
        adapter.canonicalize(ev)
        rows.append(_row(adapter, ev, k0, method=scheme, weight_frac=float(f),
                         n_zeroed=int(sum(int((~raw.masks[k]).sum()) for k in names)),
                         seconds=time.time() - t0, **_anchored_columns(adapter, ev, ref, cfg)))
        if verbose:
            r = rows[-1]
            print(f"[{scheme}] ratio {f:.2f}  k_eff removed {100 * r['frac_removed']:.1f}%  "
                  f"test acc {r['test_acc']:.3f}  {r['seconds']:.0f}s")
    return _within_tolerance(pd.DataFrame(rows))
