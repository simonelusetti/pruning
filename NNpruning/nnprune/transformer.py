"""
One-layer attention + ReLU-MLP transformer trained from scratch on a + b (mod p).

Sequence (a, b, '='), causal attention, logits read at the last position, no
LayerNorm (so no LayerNorm-gain gauge), no attention biases (so no key-bias null).

Boundary moves
--------------
Attention head h (layer index 1)
* head_off      head output -> 0 (e.g. all mass on a sink with v = 0).  Frees Q,K,V,O.
* head_mean     head output -> its training mean per position, folded into the
                per-position attention-output bias B_attn (mean-ablation).
* attn_uniform  s -> 0 limit of the softmax (W_QK -> 0): uniform causal pattern.
                Frees W_Q, W_K of the head.
* attn_fixed    pattern -> its training mean (constant in the input); covers the
                hard-max limit whenever the argmax does not depend on the input.
                Frees W_Q, W_K; the fixed pattern costs T(T-1)/2 numbers.
* qk_rank / ov_rank   rank of W_Q W_K^T (resp. W_V W_O) reduced by one with the
                activation-weighted SVD  min || C_l^{1/2} (M - M') C_r^{1/2} ||_F
                (closed form; the same idea as activation-aware SVD compression).
MLP unit i (layer index 2)
* unit_sat      b_i -> -inf: ReLU unit off (ReLU has no finite +inf saturation).
* unit_mean     fan-in -> 0: unit = its mean activation, folded into b_out.
* unit_merge    a_i ~= alpha a_i2 + beta, folded into unit i2 and b_out.
* unit_fisher / head_fisher  delete one component after jointly refitting every
                surviving outgoing map and the applicable bias in the downstream
                Fisher metric.

`query_positions` lists the positions whose outputs reach the loss.  With one
layer and logits read at the last token only that position matters, so the
activation statistics used by the rank / mean moves are taken there.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .core import ACTIVE, REMOVED, Adapter, Move, State, zero_entries
from .mlp import (grouped_fisher_compensation, grouped_fisher_pair_compensations,
                  grouped_fisher_singleton_compensations,
                  quantization_metadata, quantize_grouped_fisher_compensation,
                  quantize_grouped_fisher_singleton_compensations,
                  quantize_shortlisted_fisher_moves)

CLASSIC_MOVES = ("head_off", "head_mean", "attn_uniform", "attn_fixed",
                 "qk_rank", "ov_rank", "unit_sat", "unit_mean", "unit_merge")
FISHER_MOVES = ("unit_fisher", "unit_fisher_pair", "head_fisher",
                "head_fisher_pair", "qk_rank", "ov_rank")
FISHER_QUANTIZED_MOVES = ("unit_fisher_quantized", "head_fisher_quantized",
                          "qk_rank", "ov_rank")
FISHER_QUANTIZED_POST_MOVES = ("unit_fisher_quantized_post",
                               "head_fisher_quantized_post", "qk_rank", "ov_rank")
EXTENSION_MOVES = ("unit_fisher_quantized", "head_fisher_quantized",
                   "unit_fisher_quantized_post", "head_fisher_quantized_post")
MOVE_GROUPS = {"classic": CLASSIC_MOVES, "outdated": (),
               "extension": EXTENSION_MOVES, "fisher": FISHER_MOVES}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def make_modadd_data(p=59, train_frac=0.5, val_frac=0.25, seed=0):
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    X = torch.stack([a, b, torch.full_like(a, p)], 1)
    y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_tr = int(round(train_frac * p * p))
    n_va = int(round(val_frac * p * p))
    sp = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    return {k: (X[v], y[v]) for k, v in sp.items()}


def _psd_sqrt(C, floor, inv=False):
    w, V = torch.linalg.eigh((C + C.T) / 2)
    w = w.clamp_min(floor)
    return (V * (w.rsqrt() if inv else w.sqrt())) @ V.T


def weighted_lowrank(M, C_left, C_right, r, eps=1e-8):
    """Factors A, B (d x r) minimising || C_l^{1/2} (M - A B^T) C_r^{1/2} ||_F.

    Eigenvalues of each C are floored at eps * lambda_max: directions carrying no
    data weight are then left essentially as in M instead of being amplified."""
    M = M.double()
    Cl, Cr = C_left.double(), C_right.double()
    fl = eps * float(torch.linalg.eigvalsh(Cl).max().clamp_min(1e-300))
    fr = eps * float(torch.linalg.eigvalsh(Cr).max().clamp_min(1e-300))
    Sl, Sr = _psd_sqrt(Cl, fl), _psd_sqrt(Cr, fr)
    Sli, Sri = _psd_sqrt(Cl, fl, inv=True), _psd_sqrt(Cr, fr, inv=True)
    U, S, Vh = torch.linalg.svd(Sl @ M @ Sr)
    sq = S[:r].sqrt()
    A = Sli @ (U[:, :r] * sq)
    B = Sri @ (Vh[:r].T * sq)
    return A.float(), B.float()


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class TransformerAdapter(Adapter):
    loss_type = "categorical"
    move_groups = MOVE_GROUPS
    prunable = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]
    site_layer = {"W_E": 0, "W_pos": 0, "W_Q": 1, "W_K": 1, "W_V": 1, "W_O": 1,
                  "W_in": 2, "W_out": 2, "W_U": 3}

    def __init__(self, data, p, d=64, n_heads=4, d_head=16, d_mlp=256, T=3,
                 query_positions=(2,)):
        super().__init__(data)
        self.p, self.d, self.H, self.dh, self.dm, self.T = p, d, n_heads, d_head, d_mlp, T
        self.qpos = list(query_positions)
        self.causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

    # ---- construction -------------------------------------------------------
    def init_params(self, seed=0) -> dict:
        g = torch.Generator().manual_seed(seed)
        r = lambda *s, std: torch.randn(*s, generator=g) * std
        d, H, dh, dm, p, T = self.d, self.H, self.dh, self.dm, self.p, self.T
        return {
            "W_E": r(p + 1, d, std=1 / math.sqrt(d)),
            "W_pos": r(T, d, std=1 / math.sqrt(d)),
            "W_Q": r(H, d, dh, std=1 / math.sqrt(d)),
            "W_K": r(H, d, dh, std=1 / math.sqrt(d)),
            "W_V": r(H, d, dh, std=1 / math.sqrt(d)),
            "W_O": r(H, dh, d, std=1 / math.sqrt(H * dh)),
            "B_attn": torch.zeros(T, d),
            "W_in": r(dm, d, std=1 / math.sqrt(d)),
            "b_in": torch.zeros(dm),
            "W_out": r(d, dm, std=1 / math.sqrt(dm)),
            "b_out": torch.zeros(d),
            "W_U": r(d, p, std=1 / math.sqrt(d)),
            "b_U": torch.zeros(p),
        }

    def make_state(self, params: dict) -> State:
        params = {k: v.detach().clone() for k, v in params.items()}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
        masks["B_attn"][:] = False           # fold target, live only once used
        struct = {
            "head_alive": torch.ones(self.H, dtype=torch.bool),
            "head_fixed": torch.zeros(self.H, dtype=torch.bool),
            "A_fixed": torch.zeros(self.H, self.T, self.T),
            "unit_type": torch.full((self.dm,), ACTIVE, dtype=torch.int8),
        }
        return State(params, masks, struct)

    # ---- forward --------------------------------------------------------------
    def forward_full(self, P, S, X):
        x0 = P["W_E"][X] + P["W_pos"]
        q = torch.einsum("btd,hde->bhte", x0, P["W_Q"])
        k = torch.einsum("btd,hde->bhte", x0, P["W_K"])
        v = torch.einsum("btd,hde->bhte", x0, P["W_V"])
        sc = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(self.dh)
        sc = sc.masked_fill(~self.causal, float("-inf"))
        attn = torch.softmax(sc, -1)
        attn = torch.where(S["head_fixed"][:, None, None], S["A_fixed"], attn)
        z = torch.einsum("bhts,bhse->bhte", attn, v)
        ho = torch.einsum("bhte,hed->bhtd", z, P["W_O"])
        ho = ho * S["head_alive"].to(ho.dtype)[:, None, None]
        x1 = x0 + ho.sum(-3) + P["B_attn"]
        act = torch.relu(x1 @ P["W_in"].T + P["b_in"])
        act = act * (S["unit_type"] == ACTIVE).to(act.dtype)
        x2 = x1 + act @ P["W_out"].T + P["b_out"]
        logits = x2[..., -1, :] @ P["W_U"] + P["b_U"]
        return logits, {"x0": x0, "attn": attn, "z": z, "ho": ho, "x1": x1, "act": act, "x2": x2}

    def logits(self, P, S, X):
        return self.forward_full(P, S, X)[0]

    # ---- structural bookkeeping ----------------------------------------------
    def _kill_head(self, s: State, h: int) -> None:
        for name in ("W_Q", "W_K", "W_V", "W_O"):
            m = torch.zeros_like(s.masks[name])
            m[h] = True
            zero_entries(s, name, m)
        s.struct["head_alive"][h] = False
        s.struct["head_fixed"][h] = False

    def _zero_qk(self, s: State, h: int) -> None:
        for name in ("W_Q", "W_K"):
            m = torch.zeros_like(s.masks[name])
            m[h] = True
            zero_entries(s, name, m)

    def _kill_unit(self, s: State, i: int) -> None:
        device = s.params["W_in"].device
        zero_entries(s, "W_in", torch.arange(self.d, device=device) + i * self.d)
        zero_entries(s, "b_in", torch.tensor([i], device=device))
        col = torch.zeros_like(s.masks["W_out"])
        col[:, i] = True
        zero_entries(s, "W_out", col)
        s.struct["unit_type"][i] = REMOVED

    def _fold_unit(self, s: State, i: int, c: float) -> None:
        W, M = s.params["W_out"], s.masks["W_out"]
        s.params["b_out"] += W[:, i] * M[:, i].to(W.dtype) * c
        self._kill_unit(s, i)

    def canonicalize(self, s: State) -> None:
        st = s.struct
        for h in range(self.H):
            if not bool(st["head_alive"][h]):
                if any(bool(s.masks[n][h].any()) for n in ("W_Q", "W_K", "W_V", "W_O")):
                    self._kill_head(s, h)
                continue
            # head-dimension columns: a QK (OV) column is null if its partner is dead
            qk = s.masks["W_Q"][h].any(0) & s.masks["W_K"][h].any(0)
            ov = s.masks["W_V"][h].any(0) & s.masks["W_O"][h].any(1)
            for e in (~qk).nonzero().flatten().tolist():
                for n in ("W_Q", "W_K"):
                    m = torch.zeros_like(s.masks[n]); m[h, :, e] = True
                    zero_entries(s, n, m)
            for e in (~ov).nonzero().flatten().tolist():
                m = torch.zeros_like(s.masks["W_V"]); m[h, :, e] = True
                zero_entries(s, "W_V", m)
                m = torch.zeros_like(s.masks["W_O"]); m[h, e, :] = True
                zero_entries(s, "W_O", m)
            if not bool(ov.any()):
                self._kill_head(s, h)
            elif bool(st["head_fixed"][h]) and bool(qk.any()):
                self._zero_qk(s, h)
        # MLP units
        changed = True
        while changed:
            changed = False
            t = st["unit_type"]
            fan_out = s.masks["W_out"].any(0)
            fan_in = s.masks["W_in"].any(1)
            for i in ((t == ACTIVE) & ~fan_out).nonzero().flatten().tolist():
                self._kill_unit(s, i); changed = True
            t = st["unit_type"]
            for i in ((t == ACTIVE) & ~fan_in).nonzero().flatten().tolist():
                b = s.params["b_in"][i] if bool(s.masks["b_in"][i]) else s.params["b_in"].new_zeros(())
                self._fold_unit(s, i, float(torch.relu(b))); changed = True

    def extra_k(self, s: State) -> int:
        return int(s.struct["head_fixed"].sum()) * (self.T * (self.T - 1) // 2)

    def qk_rank(self, s, h) -> int:
        return int((s.masks["W_Q"][h].any(0) & s.masks["W_K"][h].any(0)).sum())

    def ov_rank(self, s, h) -> int:
        return int((s.masks["W_V"][h].any(0) & s.masks["W_O"][h].any(1)).sum())

    # ---- statistics for structural moves ----------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s: State, cfg) -> dict:
        X, _ = self.data["train"]
        P = self.effective(s)
        logits, c = self.forward_full(P, s.struct, X)
        enabled = self.resolve_struct_kinds(cfg.struct_kinds)
        qp = self.qpos
        n = X.shape[0]
        xq = c["x0"][:, qp].reshape(-1, self.d).double()
        xk = c["x0"].reshape(-1, self.d).double()
        ctx = {
            "head_mean": c["ho"].mean(0),                                   # (H,T,d)
            "attn_mean": c["attn"].mean(0),                                 # (H,T,T)
            "Cq": xq.T @ xq / xq.shape[0],
            "Ck": xk.T @ xk / xk.shape[0],
        }
        Cy = []
        for h in range(self.H):
            y = torch.einsum("bts,bsd->btd", c["attn"][:, h, qp], c["x0"]).reshape(-1, self.d).double()
            Cy.append(y.T @ y / y.shape[0])
        ctx["Cy"] = torch.stack(Cy)
        A = c["act"][:, qp].reshape(-1, self.dm)
        ctx["act_mean"] = A.mean(0)
        ctx["act"] = A
        # merge pairs among active units
        pairs = []
        ids = (s.struct["unit_type"] == ACTIVE).nonzero().flatten()
        if "unit_merge" in enabled and ids.numel() >= 2 and cfg.n_merge > 0:
            Aa = A[:, ids]
            Ac = Aa - Aa.mean(0)
            sd = Ac.pow(2).mean(0).sqrt()
            ok = sd > 1e-8
            C = (Ac.T @ Ac) / Aa.shape[0] / (sd[:, None] * sd[None, :]).clamp_min(1e-16)
            C = C * (ok[:, None] & ok[None, :])
            iu = torch.triu_indices(len(ids), len(ids), 1, device=ids.device)
            vals = C[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                a_, b_ = int(iu[0, t_]), int(iu[1, t_])
                for i, i2 in ((a_, b_), (b_, a_)):
                    alpha = float((Ac[:, i] * Ac[:, i2]).mean() / Ac[:, i2].pow(2).mean().clamp_min(1e-16))
                    beta = float(Aa[:, i].mean() - alpha * Aa[:, i2].mean())
                    pairs.append((int(ids[i]), int(ids[i2]), alpha, beta, float(vals[t_])))
        ctx["merge_pairs"] = pairs
        ctx["fisher_moves"] = (self.fisher_deletion_moves(s, P, logits, c, cfg)
                               if any(k in enabled for k in (
                                   "unit_fisher", "unit_fisher_pair",
                                   "head_fisher", "head_fisher_pair",
                                   "unit_fisher_quantized", "head_fisher_quantized",
                                   "unit_fisher_quantized_post",
                                   "head_fisher_quantized_post"))
                               else [])
        return ctx

    @torch.no_grad()
    def fisher_deletion_moves(self, s: State, P, logits, cache, cfg) -> list[Move]:
        n = logits.shape[0]
        if cfg.screen_n is not None and cfg.screen_n < n:
            generator = torch.Generator(device=logits.device).manual_seed(cfg.seed + 7)
            sample_idx = torch.randperm(n, generator=generator, device=logits.device)[:cfg.screen_n]
            logits = logits[sample_idx]
        else:
            sample_idx = torch.arange(n, device=logits.device)

        probabilities = torch.softmax(logits, -1)
        generator = torch.Generator(device=logits.device).manual_seed(cfg.seed + 41)
        labels = torch.multinomial(probabilities, cfg.n_mc, replacement=True,
                                   generator=generator)
        score = (torch.nn.functional.one_hot(labels, self.p).to(logits.dtype)
                 - probabilities[:, None, :]).reshape(-1, self.p)
        repeat = cfg.n_mc
        g2 = score @ P["W_U"].T
        weights = torch.ones(g2.shape[0], dtype=g2.dtype, device=g2.device)
        act = cache["act"][sample_idx, -1].repeat_interleave(repeat, 0)
        z = cache["z"][sample_idx, :, -1].repeat_interleave(repeat, 0)
        moves = []
        enabled = self.resolve_struct_kinds(cfg.struct_kinds)

        if any(k in enabled for k in (
                "unit_fisher", "unit_fisher_pair", "unit_fisher_quantized",
                "unit_fisher_quantized_post")):
            live = ((s.struct["unit_type"] == ACTIVE)
                    & s.masks["W_out"].any(0)).nonzero().flatten()
            if live.numel():
                A = act[:, live]
                outgoing = P["W_out"][:, live]
                adjustable = s.masks["W_out"][:, live]
                if "unit_fisher" in enabled:
                    fits = grouped_fisher_singleton_compensations(
                        A, weights, outgoing, downstream_jacobian=g2,
                        adjustable_mask=adjustable)
                    for global_i_tensor, fit in zip(live, fits):
                        moves.append(Move(
                            "unit_fisher", "mlp", 2,
                            {"i": int(global_i_tensor),
                             "survivor_ids": live[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "fisher_cost": fit.predicted_cost,
                             "fisher_rank": fit.rank,
                             "fisher_rcond": fit.rcond,
                             "screen_cost": "fisher"},
                            pred_cost=fit.predicted_cost))
                if "unit_fisher_pair" in enabled:
                    pair_fits = grouped_fisher_pair_compensations(
                        A, weights, outgoing, downstream_jacobian=g2,
                        adjustable_mask=adjustable,
                        limit=cfg.n_struct * cfg.expand_on_fail)
                    for pair, fit in pair_fits:
                        global_pair = live[pair]
                        moves.append(Move(
                            "unit_fisher_pair", "mlp", 2,
                            {"i": int(global_pair[0]), "j": int(global_pair[1]),
                             "survivor_ids": live[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "fisher_cost": fit.predicted_cost,
                             "fisher_rank": fit.rank,
                             "fisher_rcond": fit.rcond,
                             "screen_cost": "fisher"},
                            pred_cost=fit.predicted_cost))
                if "unit_fisher_quantized" in enabled:
                    fits = grouped_fisher_singleton_compensations(
                        A, weights, outgoing, downstream_jacobian=g2,
                        adjustable_mask=adjustable)
                    quantized_fits = quantize_grouped_fisher_singleton_compensations(
                        fits, A, weights, outgoing, cfg.fisher_quant_gamma,
                        downstream_jacobian=g2)
                    for global_i_tensor, quantized in zip(live, quantized_fits):
                        moves.append(Move(
                            "unit_fisher_quantized", "mlp", 2,
                            {"i": int(global_i_tensor),
                             "survivor_ids": live[quantized.survivor_indices],
                             "delta_out": quantized.delta_weights,
                             "delta_bias": quantized.delta_bias,
                             "gamma": cfg.fisher_quant_gamma,
                             "fisher_cost": quantized.predicted_cost,
                             "fisher_rank": quantized.rank,
                             "fisher_rcond": quantized.rcond,
                             "screen_cost": "fisher",
                             **quantization_metadata(quantized)},
                            pred_cost=quantized.predicted_cost))
                if "unit_fisher_quantized_post" in enabled:
                    fits = grouped_fisher_singleton_compensations(
                        A, weights, outgoing, downstream_jacobian=g2,
                        adjustable_mask=adjustable)
                    context = {"activations": A, "fisher_weights": weights,
                               "outgoing": outgoing, "jacobian": g2}
                    for local_i, (global_i_tensor, fit) in enumerate(zip(live, fits)):
                        moves.append(Move(
                            "unit_fisher_quantized_post", "mlp", 2,
                            {"i": int(global_i_tensor),
                             "survivor_ids": live[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "fisher_cost": fit.predicted_cost,
                             "fisher_rank": fit.rank,
                             "fisher_rcond": fit.rcond,
                             "screen_cost": "fisher",
                             "_quant_context": context,
                             "_quant_fit": fit,
                             "_quant_remove": local_i,
                             "_quant_singleton": True},
                            pred_cost=fit.predicted_cost))

        if any(k in enabled for k in (
                "head_fisher", "head_fisher_pair", "head_fisher_quantized",
                "head_fisher_quantized_post")):
            local = (act > 0).to(act.dtype) * (s.struct["unit_type"] == ACTIVE)
            g1 = g2 + ((g2 @ P["W_out"]) * local) @ P["W_in"]
            live_dims = ((s.struct["head_alive"][:, None]
                         & s.masks["W_V"].any(1)
                         & s.masks["W_O"].any(2)).reshape(-1).nonzero().flatten())
            if live_dims.numel():
                A = z.reshape(z.shape[0], -1)[:, live_dims]
                outgoing = P["W_O"].reshape(-1, self.d).T[:, live_dims]
                adjustable = s.masks["W_O"].reshape(-1, self.d).T[:, live_dims]
                heads = s.struct["head_alive"].nonzero().flatten()
                if "head_fisher" in enabled:
                    for h in heads.tolist():
                        remove = ((live_dims // self.dh) == h).nonzero().flatten()
                        if remove.numel() == 0:
                            continue
                        fit = grouped_fisher_compensation(
                            A, weights, outgoing, remove, downstream_jacobian=g1,
                            adjustable_mask=adjustable)
                        moves.append(Move(
                            "head_fisher", f"head{h}", 1,
                            {"h": h, "survivor_dims": live_dims[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "fisher_cost": fit.predicted_cost,
                             "fisher_rank": fit.rank,
                             "fisher_rcond": fit.rcond,
                             "screen_cost": "fisher"},
                            pred_cost=fit.predicted_cost))
                if "head_fisher_pair" in enabled and heads.numel() >= 2:
                    for pair in torch.combinations(heads, 2):
                        remove = (((live_dims // self.dh) == pair[0])
                                  | ((live_dims // self.dh) == pair[1])).nonzero().flatten()
                        fit = grouped_fisher_compensation(
                            A, weights, outgoing, remove, downstream_jacobian=g1,
                            adjustable_mask=adjustable)
                        moves.append(Move(
                            "head_fisher_pair", f"heads{int(pair[0])}+{int(pair[1])}", 1,
                            {"h": int(pair[0]), "j": int(pair[1]),
                             "survivor_dims": live_dims[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "fisher_cost": fit.predicted_cost,
                             "fisher_rank": fit.rank,
                             "fisher_rcond": fit.rcond,
                             "screen_cost": "fisher"},
                            pred_cost=fit.predicted_cost))
                if "head_fisher_quantized" in enabled:
                    group_ids = live_dims // self.dh
                    for h in heads.tolist():
                        remove = (group_ids == h).nonzero().flatten()
                        if remove.numel() == 0:
                            continue
                        fit = grouped_fisher_compensation(
                            A, weights, outgoing, remove, downstream_jacobian=g1,
                            adjustable_mask=adjustable)
                        quantized = quantize_grouped_fisher_compensation(
                            fit, A, weights, outgoing, remove,
                            cfg.fisher_quant_gamma, downstream_jacobian=g1,
                            group_ids=group_ids)
                        moves.append(Move(
                            "head_fisher_quantized", f"head{h}", 1,
                            {"h": h,
                             "survivor_dims": live_dims[quantized.survivor_indices],
                             "delta_out": quantized.delta_weights,
                             "delta_bias": quantized.delta_bias,
                             "gamma": cfg.fisher_quant_gamma,
                             "fisher_cost": quantized.predicted_cost,
                             "fisher_rank": quantized.rank,
                             "fisher_rcond": quantized.rcond,
                             "screen_cost": "fisher",
                             **quantization_metadata(quantized)},
                            pred_cost=quantized.predicted_cost))
                if "head_fisher_quantized_post" in enabled:
                    group_ids = live_dims // self.dh
                    context = {"activations": A, "fisher_weights": weights,
                               "outgoing": outgoing, "jacobian": g1}
                    for h in heads.tolist():
                        remove = (group_ids == h).nonzero().flatten()
                        if remove.numel() == 0:
                            continue
                        fit = grouped_fisher_compensation(
                            A, weights, outgoing, remove, downstream_jacobian=g1,
                            adjustable_mask=adjustable)
                        moves.append(Move(
                            "head_fisher_quantized_post", f"head{h}", 1,
                            {"h": h,
                             "survivor_dims": live_dims[fit.survivor_indices],
                             "delta_out": fit.delta_weights,
                             "delta_bias": fit.delta_bias,
                             "fisher_cost": fit.predicted_cost,
                             "fisher_rank": fit.rank,
                             "fisher_rcond": fit.rcond,
                             "screen_cost": "fisher",
                             "_quant_context": context,
                             "_quant_fit": fit,
                             "_quant_remove": remove,
                             "_quant_group_ids": group_ids},
                            pred_cost=fit.predicted_cost))
        return moves

    def prepare_structural_candidates(self, s: State, moves: list, cfg) -> list:
        return quantize_shortlisted_fisher_moves(moves, cfg.fisher_quant_gamma)

    def structural_moves(self, s: State, ctx: dict, cfg) -> list:
        st = s.struct
        mv = list(ctx.get("fisher_moves", []))
        for h in range(self.H):
            if not bool(st["head_alive"][h]):
                continue
            site = f"head{h}"
            mv.append(Move("head_off", site, 1, {"h": h}))
            mv.append(Move("head_mean", site, 1, {"h": h, "mean": ctx["head_mean"][h]}))
            if not bool(st["head_fixed"][h]) and self.qk_rank(s, h) > 0:
                mv.append(Move("attn_uniform", site, 1, {"h": h}))
                mv.append(Move("attn_fixed", site, 1, {"h": h, "A": ctx["attn_mean"][h]}))
                if self.qk_rank(s, h) > 1:
                    mv.append(Move("qk_rank", site, 1, {"h": h, "Cq": ctx["Cq"], "Ck": ctx["Ck"]}))
            if self.ov_rank(s, h) > 1:
                mv.append(Move("ov_rank", site, 1, {"h": h, "Cy": ctx["Cy"][h]}))
        for i in (st["unit_type"] == ACTIVE).nonzero().flatten().tolist():
            mv.append(Move("unit_sat", "mlp", 2, {"i": i, "c": 0.0}))
            mv.append(Move("unit_mean", "mlp", 2, {"i": i, "c": float(ctx["act_mean"][i])}))
        for (i, i2, alpha, beta, corr) in ctx["merge_pairs"]:
            mv.append(Move("unit_merge", "mlp", 2, {"i": i, "i2": i2, "alpha": alpha, "beta": beta, "corr": corr}))
        return mv

    def apply_structural(self, s: State, m: Move) -> bool:
        st, p = s.struct, m.payload
        if m.kind.startswith("unit_"):
            t = st["unit_type"]
            i = p["i"]
            if int(t[i]) != ACTIVE:
                return False
            if m.kind in ("unit_fisher", "unit_fisher_pair", "unit_fisher_quantized",
                          "unit_fisher_quantized_post"):
                removed = [i, p["j"]] if m.kind == "unit_fisher_pair" else [i]
                if any(int(t[j]) != ACTIVE for j in removed):
                    return False
                survivor_ids = p["survivor_ids"]
                if survivor_ids.numel() and not bool((t[survivor_ids] == ACTIVE).all()):
                    return False
                s.params["W_out"][:, survivor_ids] += p["delta_out"].to(s.params["W_out"].dtype)
                s.params["b_out"] += p["delta_bias"].to(s.params["b_out"].dtype)
                for j in removed:
                    self._kill_unit(s, j)
                return True
            if m.kind in ("unit_sat", "unit_mean"):
                self._fold_unit(s, i, p["c"])
                return True
            i2 = p["i2"]
            if int(t[i2]) != ACTIVE:
                return False
            W, M = s.params["W_out"], s.masks["W_out"]
            col = W[:, i] * M[:, i].to(W.dtype)
            W[:, i2] = W[:, i2] * M[:, i2].to(W.dtype) + p["alpha"] * col
            M[:, i2] = M[:, i2] | M[:, i]
            s.params["b_out"] += p["beta"] * col
            self._kill_unit(s, i)
            return True

        h = p["h"]
        if not bool(st["head_alive"][h]):
            return False
        if m.kind in ("head_fisher", "head_fisher_pair", "head_fisher_quantized",
                      "head_fisher_quantized_post"):
            removed = [h, p["j"]] if m.kind == "head_fisher_pair" else [h]
            if any(not bool(st["head_alive"][j]) for j in removed):
                return False
            survivor_dims = p["survivor_dims"]
            flat = s.params["W_O"].view(-1, self.d)
            flat[survivor_dims] += p["delta_out"].to(flat.dtype).T
            s.params["B_attn"][-1] += p["delta_bias"].to(s.params["B_attn"].dtype)
            s.masks["B_attn"][-1] = True
            for j in removed:
                self._kill_head(s, j)
            return True
        if m.kind == "head_off":
            self._kill_head(s, h)
            return True
        if m.kind == "head_mean":
            s.params["B_attn"] += p["mean"]
            s.masks["B_attn"][:] = True
            self._kill_head(s, h)
            return True
        if m.kind in ("attn_uniform", "attn_fixed", "qk_rank"):
            if bool(st["head_fixed"][h]) or self.qk_rank(s, h) == 0:
                return False
            if m.kind == "attn_uniform":
                self._zero_qk(s, h)
                return True
            if m.kind == "attn_fixed":
                st["A_fixed"][h] = p["A"]
                st["head_fixed"][h] = True
                self._zero_qk(s, h)
                return True
            r = self.qk_rank(s, h) - 1
            if r < 1:
                return False
            Wq = s.params["W_Q"][h] * s.masks["W_Q"][h]
            Wk = s.params["W_K"][h] * s.masks["W_K"][h]
            A, B = weighted_lowrank(Wq @ Wk.T, p["Cq"], p["Ck"], r)
            self._set_factors(s, "W_Q", "W_K", h, A, B, r, transpose_second=False)
            return True
        if m.kind == "ov_rank":
            r = self.ov_rank(s, h) - 1
            if r < 1:
                return False
            Wv = s.params["W_V"][h] * s.masks["W_V"][h]
            Wo = s.params["W_O"][h] * s.masks["W_O"][h]
            A, B = weighted_lowrank(Wv @ Wo, p["Cy"], torch.eye(self.d, device=Wv.device), r)
            self._set_factors(s, "W_V", "W_O", h, A, B, r, transpose_second=True)
            return True
        raise ValueError(m.kind)

    def _set_factors(self, s, n1, n2, h, A, B, r, transpose_second):
        s.params[n1][h].zero_(); s.masks[n1][h] = False
        s.params[n1][h][:, :r] = A; s.masks[n1][h][:, :r] = True
        s.params[n2][h].zero_(); s.masks[n2][h] = False
        if transpose_second:          # W_O[h] is (d_head, d): rows are head dims
            s.params[n2][h][:r, :] = B.T; s.masks[n2][h][:r, :] = True
        else:
            s.params[n2][h][:, :r] = B; s.masks[n2][h][:, :r] = True

    # ---- classical structured baseline ------------------------------------------
    @torch.no_grad()
    def classical_unit_order(self, s: State) -> list:
        """Mean-ablation order: variance of each unit's / head's contribution to the
        residual stream at the query positions (smallest first)."""
        X, _ = self.data["train"]
        _, c = self.forward_full(self.effective(s), s.struct, X)
        qp = self.qpos
        A = c["act"][:, qp].reshape(-1, self.dm)
        Wout = s.params["W_out"] * s.masks["W_out"]
        sc_u = A.var(0) * Wout.pow(2).sum(0)
        sc_h = c["ho"][:, :, qp].var(0).sum((-1, -2))          # (H,)
        items = [(float(sc_u[i]), ("unit", i)) for i in range(self.dm)
                 if int(s.struct["unit_type"][i]) == ACTIVE]
        items += [(float(sc_h[h]), ("head", h)) for h in range(self.H) if bool(s.struct["head_alive"][h])]
        return [u for _, u in sorted(items, key=lambda t: t[0])]

    @torch.no_grad()
    def mean_move(self, s: State, unit) -> Move:
        X, _ = self.data["train"]
        _, c = self.forward_full(self.effective(s), s.struct, X)
        kind, j = unit
        if kind == "unit":
            mean = float(c["act"][:, self.qpos, j].mean())
            return Move("unit_mean", "mlp", 2, {"i": j, "c": mean})
        return Move("head_mean", f"head{j}", 1, {"h": j, "mean": c["ho"][:, j].mean(0)})

    # ---- Wanda input norms --------------------------------------------------------
    @torch.no_grad()
    def input_norms(self, s: State) -> dict:
        X, _ = self.data["train"]
        _, c = self.forward_full(self.effective(s), s.struct, X)
        n = X.shape[0]
        counts = torch.bincount(X[:, :].reshape(-1), minlength=self.p + 1).float()
        x0n = c["x0"].reshape(-1, self.d).pow(2).sum(0).sqrt()
        zn = c["z"].pow(2).sum((0, 2)).sqrt()                      # (H, dh)
        x1n = c["x1"].reshape(-1, self.d).pow(2).sum(0).sqrt()
        an = c["act"].reshape(-1, self.dm).pow(2).sum(0).sqrt()
        x2n = c["x2"][:, -1].pow(2).sum(0).sqrt()
        return {
            "W_E": counts.sqrt()[:, None],
            "W_pos": torch.full((self.T, 1), math.sqrt(n), device=X.device),
            "W_Q": x0n[None, :, None], "W_K": x0n[None, :, None], "W_V": x0n[None, :, None],
            "W_O": zn[:, :, None],
            "W_in": x1n[None, :], "W_out": an[None, :], "W_U": x2n[:, None],
        }


# --------------------------------------------------------------------------
# Training (AdamW with strong weight decay, full batch: the grokking recipe)
# --------------------------------------------------------------------------

def train_transformer(adapter: TransformerAdapter, steps=8000, lr=1e-3, wd=1.0, seed=0,
                      log_every=500, stop_val_acc=None, verbose=True, batch_size=None):
    torch.manual_seed(seed)
    X, y = adapter.data["train"]
    P = {k: v.to(X.device).requires_grad_(True) for k, v in adapter.init_params(seed).items()}
    trainable = [v for k, v in P.items() if k != "B_attn"]
    s = adapter.make_state(P).to(X.device)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size or len(X),
                        shuffle=batch_size is not None)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    hist = []
    eff_mask = {k: s.masks[k].float() for k in P}
    for it in range(steps):
        total = torch.zeros((), device=X.device)
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            Pe = {k: P[k] * eff_mask[k] for k in P}
            loss = adapter.loss_from_logits(adapter.logits(Pe, s.struct, xb), yb).mean()
            loss.backward()
            opt.step()
            total += loss.detach() * len(xb)
        train_loss = float(total / len(X))
        if it % log_every == 0 or it == steps - 1:
            st = adapter.make_state(P).to(X.device)
            va = adapter.metrics(st, "val")
            hist.append({"step": it, "train_loss": train_loss, "val_loss": va["loss"], "val_acc": va["acc"]})
            if verbose:
                print(f"step {it:6d}  train {train_loss:.4f}  val {va['loss']:.4f}  val_acc {va['acc']:.3f}")
            if stop_val_acc is not None and va["acc"] >= stop_val_acc and it > 0:
                break
    return adapter.make_state(P).to(X.device), hist
