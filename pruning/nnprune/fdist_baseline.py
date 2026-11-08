"""Berman model-Fisher diagonal and score primitives; scheduling lives in engine.py."""
from __future__ import annotations


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .core import Adapter, State, zero_entries
from .third_party.fdist_fim import calculate_fim_backprop


class AdapterModule(nn.Module):
    """nn.Module view of an adapter state, so their functional_call-based Fisher
    code runs unchanged.  Every parameter of the state is a module parameter
    except fold targets that are not part of the trained model (B_attn)."""

    def __init__(self, adapter: Adapter, state: State, exclude=("B_attn",)):
        super().__init__()
        self._adapter = adapter
        self._struct = state.struct
        self._sequence = bool(getattr(adapter, "fdist_sequence", False))
        eff = adapter.effective(state)
        self._fixed = {k: v.detach() for k, v in eff.items() if k in exclude}
        self._names = [k for k in eff if k not in exclude]
        for k in self._names:
            self.register_parameter(k, nn.Parameter(eff[k].detach().clone()))

    def forward(self, x):
        if self._sequence:
            # The external code expects one categorical prediction per input.
            # Sequence adapters append the selected position to each window.
            pos, x = x[:, -1].long(), x[:, :-1]
        P = dict(self._fixed)
        for k in self._names:
            P[k] = getattr(self, k)
        z = self._adapter.logits(P, self._struct, x)
        if self._sequence:
            z = z[torch.arange(len(z), device=z.device), pos]
        if self._adapter.loss_type == "bernoulli":
            z = torch.stack([torch.zeros_like(z), z], -1)
        return z


def fisher_loader(adapter: Adapter, subset_size=500, seed=42, batch_size=128):
    X, y = adapter.data["train"]
    g = torch.Generator(device=X.device).manual_seed(seed)
    if bool(getattr(adapter, "fdist_sequence", False)):
        # Sample token predictions, then encode the selected position in an
        # extra input column decoded by AdapterModule.forward.
        T = X.shape[1]
        idx = torch.randperm(X.shape[0] * T, generator=g, device=X.device)[:subset_size]
        wi, pos = torch.div(idx, T, rounding_mode="floor"), idx % T
        Xs = torch.cat([X[wi], pos[:, None].to(X.dtype)], 1)
        ys = y[wi, pos]
    else:
        idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:subset_size]
        Xs, ys = X[idx], y[idx]
    return DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=False)


def _fisher_dict(module: AdapterModule, loader) -> dict:
    device = next(module.parameters()).device
    flat = calculate_fim_backprop(module, loader, device=str(device)).to(device)
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
