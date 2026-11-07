"""Sigmoid MLP adapter and its classical structural moves."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .canonicalize.mlp import HiddenLayer, canonicalize_mlp
from .core import ACTIVE, REMOVED, STEP, Adapter, Move, State, zero_entries

CLASSIC_MOVES = ("unit_sat", "unit_mean", "unit_step", "unit_merge")
MOVE_GROUPS = {"classic": CLASSIC_MOVES,
               "fisher": ("fisher",), "fisher_general": ("fisher",)}

def make_moons_data(n=2500, noise=0.25, seed=123, val_frac=0.2, test_frac=0.2):
    """Train / validation / test.  The scaler is fit on TRAIN only."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_frac,
                                                random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    t = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
    l = lambda a: torch.tensor(a, dtype=torch.float32)
    data = {"train": (t(X_tr), l(y_tr)), "val": (t(X_va), l(y_va)), "test": (t(X_te), l(y_te))}
    return data, sc


def moons_fresh_sample(scaler, n=8000, noise=0.25, seed=999):
    """Fresh draw from the same distribution (for expected-information diagnostics)."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return (torch.tensor(scaler.transform(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class MLPAdapter(Adapter):
    move_groups = MOVE_GROUPS

    def __init__(self, data, widths=(2, 50, 50, 50, 1), step_layers=(0,)):
        super().__init__(data)
        self.widths = list(widths)
        self.L = len(widths) - 1
        self.loss_type = "bernoulli" if widths[-1] == 1 else "categorical"
        self.prunable = [f"W{l}" for l in range(self.L)]
        self.site_layer = {f"W{l}": l for l in range(self.L)}
        # layers whose INPUTS change sign (standardised features): only there is
        # a single weight -> +-inf a Heaviside of one input rather than a constant
        self.step_layers = tuple(step_layers)

    # ---- construction -------------------------------------------------------
    def init_params(self, seed=123) -> dict:
        g = torch.Generator().manual_seed(seed)
        P = {}
        for l in range(self.L):
            fan_in, fan_out = self.widths[l], self.widths[l + 1]
            bound = 1.0 / np.sqrt(fan_in)
            P[f"W{l}"] = (torch.rand(fan_out, fan_in, generator=g) * 2 - 1) * bound
            P[f"b{l}"] = (torch.rand(fan_out, generator=g) * 2 - 1) * bound
        return P

    def make_state(self, params: dict) -> State:
        params = {k: v.detach().clone() for k, v in params.items()}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
        struct = {}
        for l in range(self.L - 1):
            w = self.widths[l + 1]
            struct[f"type{l}"] = torch.full((w,), ACTIVE, dtype=torch.int8)
            struct[f"step_j{l}"] = torch.zeros(w, dtype=torch.long)
            struct[f"step_s{l}"] = torch.ones(w)
        return State(params, masks, struct)

    # ---- forward --------------------------------------------------------------
    def logits(self, P, S, X, return_acts=False):
        h = X
        acts = []
        out = None
        for l in range(self.L):
            z = h @ P[f"W{l}"].T + P[f"b{l}"]
            if l < self.L - 1:
                a = torch.sigmoid(z)
                t = S[f"type{l}"]
                if l in self.step_layers:
                    stepv = (h[..., S[f"step_j{l}"]] * S[f"step_s{l}"] > 0).to(a.dtype)
                    a = torch.where(t == STEP, stepv, a)
                a = a * (t != REMOVED).to(a.dtype)
                acts.append(a)
                h = a
            else:
                out = z.squeeze(-1) if self.widths[-1] == 1 else z
        return (out, acts) if return_acts else out

    # ---- structural bookkeeping ----------------------------------------------
    def _kill(self, s: State, l: int, i: int) -> None:
        device = s.params[f"W{l}"].device
        zero_entries(s, f"W{l}", (torch.arange(self.widths[l], device=device) + i * self.widths[l]))
        zero_entries(s, f"b{l}", torch.tensor([i], device=device))
        Wn = f"W{l + 1}"
        col = torch.zeros_like(s.masks[Wn])
        col[:, i] = True
        zero_entries(s, Wn, col)
        s.struct[f"type{l}"][i] = REMOVED

    def _fold_const(self, s: State, l: int, i: int, c: float) -> None:
        Wn, bn = f"W{l + 1}", f"b{l + 1}"
        s.params[bn] += s.params[Wn][:, i] * s.masks[Wn][:, i].to(s.params[Wn].dtype) * c
        self._kill(s, l, i)

    def canonicalize(self, s: State) -> None:
        layers = []
        for l in range(self.L - 1):
            def remove(ids, l=l):
                for i in ids.tolist():
                    self._kill(s, l, i)

            def fold(ids, l=l):
                for i in ids.tolist():
                    b = (s.params[f"b{l}"][i] if bool(s.masks[f"b{l}"][i])
                         else s.params[f"b{l}"].new_zeros(()))
                    self._fold_const(s, l, i, float(torch.sigmoid(b)))

            layers.append(HiddenLayer(
                live=lambda l=l: s.struct[f"type{l}"] != REMOVED,
                constant=lambda l=l: s.struct[f"type{l}"] == ACTIVE,
                fan_in=lambda l=l: s.masks[f"W{l}"].any(1),
                fan_out=lambda l=l: s.masks[f"W{l + 1}"].any(0),
                remove=remove,
                fold=fold,
            ))
        canonicalize_mlp(layers)

    @torch.no_grad()
    def refresh_extra(self, s: State, cfg) -> dict:
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        enabled = self.resolve_struct_kinds(cfg.struct_kinds)
        ctx = {"acts": acts, "act_mean": [a.mean(0) for a in acts],
               "merge_pairs": [[] for _ in acts], "fisher_moves": []}

        if "unit_merge" in enabled and cfg.n_merge:
            for layer, activations in enumerate(acts):
                ids = (s.struct[f"type{layer}"] == ACTIVE).nonzero().flatten()
                if ids.numel() < 2:
                    continue
                values = activations[:, ids]
                centered = values - values.mean(0)
                scale = centered.square().mean(0).sqrt().clamp_min(1e-12)
                correlation = (centered.T @ centered) / len(values)
                correlation /= scale[:, None] * scale[None, :]
                pairs = torch.triu_indices(len(ids), len(ids), 1, device=ids.device)
                scores = correlation[pairs[0], pairs[1]].abs()
                for index in torch.argsort(scores, descending=True)[:cfg.n_merge].tolist():
                    left, right = int(pairs[0, index]), int(pairs[1, index])
                    for source, target in ((left, right), (right, left)):
                        x, y = centered[:, source], centered[:, target]
                        alpha = float((x * y).mean() / y.square().mean().clamp_min(1e-12))
                        beta = float(values[:, source].mean() - alpha * values[:, target].mean())
                        ctx["merge_pairs"][layer].append(
                            (int(ids[source]), int(ids[target]), alpha, beta,
                             float(scores[index])))

        if "fisher" in enabled:
            from .fisher_miqp import miqp_moves
            ctx["fisher_moves"] = miqp_moves(self, s, cfg)
        return ctx

    def structural_moves(self, s: State, ctx: dict, cfg) -> list:
        moves = list(ctx["fisher_moves"])
        for layer in range(self.L - 1):
            status = s.struct[f"type{layer}"]
            mean = ctx["act_mean"][layer]
            for unit in (status != REMOVED).nonzero().flatten().tolist():
                moves.extend(Move("unit_sat", f"h{layer}", layer,
                                  {"l": layer, "i": unit, "c": value})
                             for value in (0.0, 1.0))
                moves.append(Move("unit_mean", f"h{layer}", layer,
                                  {"l": layer, "i": unit, "c": float(mean[unit])}))
            if layer in self.step_layers:
                inputs = self.data["train"][0] if layer == 0 else ctx["acts"][layer - 1]
                features = ((inputs.amin(0) < 0) & (inputs.amax(0) > 0)).nonzero().flatten()
                for unit in (status == ACTIVE).nonzero().flatten().tolist():
                    selected = features
                    if cfg.step_per_unit is not None and len(features) > cfg.step_per_unit:
                        weights = self.effective(s)[f"W{layer}"][unit, features].abs()
                        selected = features[torch.topk(weights, cfg.step_per_unit).indices]
                    for feature in selected.tolist():
                        moves.extend(Move("unit_step", f"h{layer}", layer,
                                          {"l": layer, "i": unit, "j": feature, "s": sign})
                                     for sign in (1.0, -1.0))
            moves.extend(Move("unit_merge", f"h{layer}", layer,
                              {"l": layer, "i": source, "i2": target,
                               "alpha": alpha, "beta": beta, "corr": correlation})
                         for source, target, alpha, beta, correlation
                         in ctx["merge_pairs"][layer])
        return moves

    def apply_structural(self, s: State, move: Move) -> bool:
        payload = move.payload
        if move.kind == "fisher":
            for name, (indices, delta) in payload["updates"].items():
                indices = indices.to(s.params[name].device)
                s.params[name].view(-1)[indices] += delta.to(s.params[name])
            for name, indices in payload["deleted"].items():
                zero_entries(s, name, indices.to(s.params[name].device))
            return True

        layer, unit = payload["l"], payload["i"]
        status = s.struct[f"type{layer}"]
        if move.kind in ("unit_sat", "unit_mean"):
            if int(status[unit]) == REMOVED:
                return False
            self._fold_const(s, layer, unit, payload["c"])
            return True
        if move.kind == "unit_step":
            if int(status[unit]) != ACTIVE or layer not in self.step_layers:
                return False
            device = s.params[f"W{layer}"].device
            indices = torch.arange(self.widths[layer], device=device) + unit * self.widths[layer]
            zero_entries(s, f"W{layer}", indices)
            zero_entries(s, f"b{layer}", torch.tensor([unit], device=device))
            status[unit] = STEP
            s.struct[f"step_j{layer}"][unit] = payload["j"]
            s.struct[f"step_s{layer}"][unit] = payload["s"]
            return True
        if move.kind == "unit_merge":
            target = payload["i2"]
            if int(status[unit]) != ACTIVE or int(status[target]) != ACTIVE:
                return False
            weight_name, bias_name = f"W{layer + 1}", f"b{layer + 1}"
            weights, mask = s.params[weight_name], s.masks[weight_name]
            source = weights[:, unit] * mask[:, unit].to(weights.dtype)
            weights[:, target] = (weights[:, target] * mask[:, target].to(weights.dtype)
                                    + payload["alpha"] * source)
            mask[:, target] |= mask[:, unit]
            s.params[bias_name] += payload["beta"] * source
            self._kill(s, layer, unit)
            return True
        raise ValueError(move.kind)

    # ---- classical structured baseline ------------------------------------------
    @torch.no_grad()
    def classical_unit_order(self, s: State) -> list:
        """Mean-ablation order: Var(a_i) * ||W_{l+1}[:, i]||^2, smallest first."""
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        items = []
        for l, a in enumerate(acts):
            Wn = s.params[f"W{l + 1}"] * s.masks[f"W{l + 1}"]
            sc = a.var(0) * Wn.pow(2).sum(0)
            t = s.struct[f"type{l}"]
            items += [(float(sc[i]), (l, i)) for i in range(a.shape[1]) if int(t[i]) != REMOVED]
        return [u for _, u in sorted(items, key=lambda t: t[0])]

    @torch.no_grad()
    def mean_move(self, s: State, unit) -> Move:
        l, i = unit
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        return Move("unit_mean", f"h{l}", l, {"l": l, "i": i, "c": float(acts[l][:, i].mean())})

    # ---- Wanda input norms --------------------------------------------------------
    @torch.no_grad()
    def input_norms(self, s: State) -> dict:
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        ins = [X] + acts
        return {f"W{l}": ins[l].pow(2).sum(0).sqrt()[None, :] for l in range(self.L)}


# --------------------------------------------------------------------------
# Training (same recipe as pruningNN.ipynb: Adam 1e-3, full batch)
# --------------------------------------------------------------------------

def train_mlp(adapter: MLPAdapter, epochs=2500, lr=1e-3, seed=123, verbose=True,
              batch_size=None):
    torch.manual_seed(seed)
    X, y = adapter.data["train"]
    P = {k: v.to(X.device).requires_grad_(True) for k, v in adapter.init_params(seed).items()}
    s = adapter.make_state(P).to(X.device)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size or len(X),
                        shuffle=batch_size is not None)
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    hist = []
    for ep in range(epochs):
        total = torch.zeros((), device=X.device)
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = adapter.loss_from_logits(adapter.logits(P, s.struct, xb), yb).mean()
            loss.backward()
            opt.step()
            total += loss.detach() * len(xb)
        train_loss = float(total / len(X))
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                st = adapter.make_state(P).to(X.device)
                hist.append({"epoch": ep, "train_loss": train_loss,
                             "val_loss": adapter.metrics(st, "val")["loss"]})
            if verbose and ep % 500 == 0:
                print(f"epoch {ep:5d}  train {hist[-1]['train_loss']:.4f}  val {hist[-1]['val_loss']:.4f}")
    return adapter.make_state(P).to(X.device), hist
