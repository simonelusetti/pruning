"""
Gap-based identifiability diagnostics for network blocks.

`classify_spectrum` is a verbatim port of criteria_bench.spectrum.classify_spectrum
(tolerance-free plateau test, outermost admissible gap; see that module for the
argument).  It is vendored so this package does not import sympy and was checked
against the original when criteria_bench was importable.

What is measured
----------------
The expected (model) Fisher information of a parameter BLOCK, i.e. the GGN,
evaluated over a population of inputs:
  Bernoulli    exact:  sum_n p_n (1 - p_n) j_n j_n^T,   j_n = dz_n / dtheta
  Categorical  Monte Carlo: labels drawn from the model, R draws per input.
Every MC gradient lies in the row space of the Jacobian, so structural null
directions come out exactly null; the MC only adds noise to the retained part.

Analytic expectations used in the notebook (1-layer modular-addition model)
--------------------------------------------------------------------------
* QK block of a head (2 d d_h numbers).  Only the last position reaches the loss
  and its residual x_2 is the same for every input ('=' at position 2), so the
  scores depend on (W_Q, W_K) only through u = W_K W_Q^T x_2 in R^d:
  null = 2 d d_h - d.
* OV block (2 d d_h).  Only the product W_V W_O (rank d_h) is identified:
  null = d_h^2 (the GL(d_h) gauge).
* ReLU unit block (W_in[i], b_in[i], W_out[:, i]; 2d + 1 numbers): positive
  rescaling gauge, null = 1 if the unit is active on part of the data.
* Sigmoid MLP (moons): no continuous symmetry, expected null = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
from torch.func import grad, vmap

REGULAR = "regular"
STRUCTURAL = "structural"
UNDER_DETERMINED = "under_determined"


@dataclass
class SpectrumVerdict:
    regime: str
    null_dim: int
    plateau_decades: float
    condition_number: float
    p: int
    eigenvalues: np.ndarray
    null_participation: np.ndarray
    repairable: bool
    reason: str = ""
    candidates: list = field(default_factory=list)

    def __repr__(self):
        return (f"<{self.regime} null_dim={self.null_dim}/{self.p} "
                f"plateau={self.plateau_decades:.1f}dec repairable={self.repairable}>")


def noise_floor_of(eigenvalues) -> float:
    """Relative floating-point noise floor of a computed PSD spectrum, estimated
    from the data: the magnitude of the most negative computed eigenvalue (a PSD
    matrix has none, so their size measures the rounding noise), never below
    16 eps."""
    ev = np.asarray(eigenvalues, dtype=float).ravel()
    lam = float(ev.max()) if ev.size else 0.0
    if lam <= 0:
        return 0.0
    return max(16 * np.finfo(float).eps, max(0.0, -float(ev.min())) / lam)


def classify_spectrum(eigenvalues: Sequence[float], min_decades: float = 3.0,
                      null_max: float = 1e-6,
                      eigenvectors: Optional[np.ndarray] = None,
                      noise_floor=None) -> SpectrumVerdict:
    """Port of criteria_bench.spectrum.classify_spectrum (same rule, same output)
    when noise_floor is None.

    noise_floor ('auto' or a float, relative to lambda_max): eigenvalues below it
    are numerically indistinguishable from zero and are raised to it before the
    gap test.  Without it, rounding noise of both signs (+1e-20 and -1e-20, the
    latter clipped to 0) creates an artificial ~290-decade "plateau" inside the
    noise, and the null dimension returned is the count of negative rounding
    errors.  With a sloppy spectrum that decays into the noise this is the only
    "gap" there is, so the floor is what lets the rule refuse."""
    ev = np.asarray(eigenvalues, dtype=float).ravel()
    p = ev.size
    if p == 0:
        return SpectrumVerdict(REGULAR, 0, np.inf, 1.0, 0, ev, np.zeros(0), False, "no parameters")
    order = np.argsort(ev)[::-1]
    ev_sorted = ev[order]
    lam_max = float(ev_sorted[0])
    if not np.isfinite(lam_max) or lam_max <= 0:
        return SpectrumVerdict(UNDER_DETERMINED, p, 0.0, np.inf, p, ev_sorted, np.ones(p) / p,
                               False, "expected information is zero or non-finite")
    mu = np.clip(ev_sorted / lam_max, 0.0, None)
    if noise_floor is not None:
        fl = noise_floor_of(ev) if noise_floor == "auto" else float(noise_floor)
        mu = np.maximum(mu, fl)
    tiny = np.finfo(float).tiny
    cands = []
    for d in range(1, p):
        lo = mu[p - d]
        hi = mu[p - d - 1]
        width = float(np.log10(max(hi, tiny)) - np.log10(max(lo, tiny)))
        cands.append({"null_dim": d, "plateau_decades": width, "mu_null_max": float(lo),
                      "admissible": bool(width >= min_decades and lo <= null_max)})
    admissible = [c for c in cands if c["admissible"]]

    def _participation(d):
        if eigenvectors is None or d == 0:
            return np.zeros(p)
        V = np.asarray(eigenvectors, dtype=float)[:, order]
        return np.sum(V[:, p - d:] ** 2, axis=1)

    def _cond(d):
        kept = ev_sorted[: p - d]
        kept = kept[kept > 0]
        return float(kept[0] / kept[-1]) if kept.size else np.inf

    if admissible:
        best = max(admissible, key=lambda c: c["null_dim"])
        d = best["null_dim"]
        return SpectrumVerdict(STRUCTURAL, d, best["plateau_decades"], _cond(d), p, ev_sorted,
                               _participation(d), True,
                               (f"gap of {best['plateau_decades']:.1f} decades below dimension "
                                f"{p - d}; cut-off directions at {best['mu_null_max']:.1e} of lambda_max"),
                               cands)
    if float(mu[-1]) > null_max:
        return SpectrumVerdict(REGULAR, 0, np.inf, _cond(0), p, ev_sorted, np.zeros(p), False,
                               f"smallest eigenvalue is {float(mu[-1]):.1e} of lambda_max; identifiable",
                               cands)
    widest = max(cands, key=lambda c: c["plateau_decades"]) if cands else None
    return SpectrumVerdict(UNDER_DETERMINED, 0, (widest["plateau_decades"] if widest else 0.0),
                           _cond(0), p, ev_sorted, np.zeros(p), False,
                           (f"small eigenvalues present (min {float(mu[-1]):.1e} of lambda_max) but "
                            f"the widest gap is only {(widest['plateau_decades'] if widest else 0.0):.1f} "
                            f"decades < min_decades={min_decades}"), cands)


def min_decades_sweep(eigenvalues, grid=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0), null_max=1e-6,
                      noise_floor="auto") -> list:
    """The one tunable, always reported."""
    return [{"min_decades": md, **{k: getattr(v, k) for k in ("regime", "null_dim", "plateau_decades")}}
            for md in grid for v in [classify_spectrum(eigenvalues, md, null_max, noise_floor=noise_floor)]]


# --------------------------------------------------------------------------
# Block Fisher
# --------------------------------------------------------------------------

def block_fisher(adapter, state, block: dict, X: torch.Tensor, R: int = 2, seed: int = 0,
                 batch: int = 256) -> np.ndarray:
    """Expected Fisher (GGN) of the parameter entries selected by `block`
    ({name: bool mask}) over the inputs X, in float64."""
    eff = {k: v.double() for k, v in adapter.effective(state).items()}
    struct = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in state.struct.items()}
    names = list(block)
    sel = {k: block[k].reshape(-1).nonzero().squeeze(1) for k in names}
    sizes = [int(sel[k].numel()) for k in names]
    theta0 = torch.cat([eff[k].reshape(-1)[sel[k]] for k in names])
    P = theta0.numel()

    def build(theta):
        Pd = dict(eff)
        off = 0
        for k, n in zip(names, sizes):
            flat = eff[k].reshape(-1).index_put((sel[k],), theta[off:off + n])
            Pd[k] = flat.reshape(eff[k].shape)
            off += n
        return Pd

    def out_single(theta, x):
        return adapter.logits(build(theta), struct, x.unsqueeze(0)).squeeze(0)

    F = torch.zeros(P, P, dtype=torch.float64, device=theta0.device)
    gen = torch.Generator(device=X.device).manual_seed(seed)
    for start in range(0, X.shape[0], batch):
        xb = X[start:start + batch]
        if xb.is_floating_point():
            xb = xb.double()
        with torch.no_grad():
            zb = adapter.logits(eff, struct, xb)
        if adapter.loss_type == "bernoulli":
            J = vmap(grad(out_single), in_dims=(None, 0))(theta0, xb)
            pb = torch.sigmoid(zb)
            F += J.T @ (J * (pb * (1 - pb))[:, None])
        else:
            probs = torch.softmax(zb, -1)
            for _ in range(R):
                yh = torch.multinomial(probs.float(), 1, generator=gen).squeeze(-1)
                G = vmap(grad(lambda th, x, t: torch.log_softmax(out_single(th, x), -1)
                              .gather(0, t.unsqueeze(0)).squeeze(0)),
                         in_dims=(None, 0, 0))(theta0, xb, yh)
                F += G.T @ G / R
    F = (F + F.T) / 2
    return F.cpu().numpy()


def classify_block(adapter, state, block, X, expected_null=None, R=2, min_decades=3.0,
                   null_max=1e-6, seed=0, noise_floor="auto") -> dict:
    F = block_fisher(adapter, state, block, X, R=R, seed=seed)
    ev = np.linalg.eigvalsh(F)
    v = classify_spectrum(ev, min_decades=min_decades, null_max=null_max, noise_floor=noise_floor)
    raw = classify_spectrum(ev, min_decades=min_decades, null_max=null_max)
    sweep = min_decades_sweep(ev, null_max=null_max, noise_floor=noise_floor)
    fl = noise_floor_of(ev)
    return {"p": v.p, "regime": v.regime, "null_dim": v.null_dim,
            "expected_null": expected_null, "plateau_decades": v.plateau_decades,
            "n_at_noise_floor": int(np.sum(ev <= fl * ev.max())),
            "noise_floor": fl,
            "null_dim_without_floor": raw.null_dim,
            "null_dims_over_min_decades": [r["null_dim"] for r in sweep],
            "eigenvalues": v.eigenvalues, "reason": v.reason}


def head_blocks(adapter, state, h: int) -> dict:
    """QK and OV blocks of head h, restricted to live entries."""
    def m(name):
        out = torch.zeros_like(state.masks[name])
        out[h] = state.masks[name][h]
        return out
    return {"QK": {"W_Q": m("W_Q"), "W_K": m("W_K")},
            "OV": {"W_V": m("W_V"), "W_O": m("W_O")}}


def unit_block(state, i: int) -> dict:
    a = torch.zeros_like(state.masks["W_in"]); a[i] = state.masks["W_in"][i]
    b = torch.zeros_like(state.masks["b_in"]); b[i] = state.masks["b_in"][i]
    c = torch.zeros_like(state.masks["W_out"]); c[:, i] = state.masks["W_out"][:, i]
    return {"W_in": a, "b_in": b, "W_out": c}


# --------------------------------------------------------------------------
# Independent predictions of the structural null dimension (1-layer model)
# --------------------------------------------------------------------------

def gap_rank(A: torch.Tensor, min_decades=3.0, null_max=1e-6):
    """Rank of the row space of A, decided by the same gap rule (no rcond).
    Returns None when the Gram spectrum is under-determined (no gap: the rank is
    then a threshold choice, not a property)."""
    G = (A.double().T @ A.double()).numpy()
    v = classify_spectrum(np.linalg.eigvalsh((G + G.T) / 2), min_decades, null_max,
                          noise_floor="auto")
    if v.regime == UNDER_DETERMINED:
        return None
    return v.p - v.null_dim


@torch.no_grad()
def transformer_predicted_nulls(adapter, state, X, units=()) -> dict:
    """Structural null dimensions predicted from data-subspace ranks, to be
    compared with the Fisher-spectrum verdicts:

      QK(h):   2 d d_h - r_K           r_K = rank{x_s - x_2 : s < 2, all inputs}
      OV(h):   2 d d_h - d_h (r_y + r_o - d_h)
               r_y = rank of the attention-weighted inputs at the query position,
               r_o = rank of the residual directions the logits can see
      unit i:  1 + (d + 1 - r_in) + (d - r_vis)
               r_in = rank of [x, 1] over inputs where the unit is active,
               r_vis = rank(W_U)  (only W_U reads the residual after the MLP)
    """
    eff = adapter.effective(state)
    _, c = adapter.forward_full(eff, state.struct, X)
    d, dh = adapter.d, adapter.dh
    x0 = c["x0"]
    D = torch.cat([x0[:, 0] - x0[:, 2], x0[:, 1] - x0[:, 2]])
    r_K = gap_rank(D)
    # visible residual directions at the last position after attention
    from torch.func import jacrev
    def f(x1_last, x1_full):
        x1 = torch.cat([x1_full[:-1], x1_last[None]], 0)
        act = torch.relu(x1 @ eff["W_in"].T + eff["b_in"]) * (state.struct["unit_type"] == 0).to(x1.dtype)
        x2 = x1 + act @ eff["W_out"].T + eff["b_out"]
        return x2[-1] @ eff["W_U"] + eff["b_U"]
    Gv = torch.zeros(d, d, dtype=torch.float64)
    for n in range(X.shape[0]):
        J = jacrev(f)(c["x1"][n, -1], c["x1"][n]).double()
        Gv += J.T @ J
    wv = np.linalg.eigvalsh(Gv.numpy())
    vv = classify_spectrum(wv, noise_floor="auto")
    r_o = None if vv.regime == UNDER_DETERMINED else vv.p - vv.null_dim
    nn = lambda *r: any(v is None for v in r)
    out = {"r_K": r_K, "r_o": r_o, "QK": None if nn(r_K) else 2 * d * dh - r_K,
           "OV": {}, "r_y": {}, "units": {}}
    for h in range(adapter.H):
        y = torch.einsum("bs,bsd->bd", c["attn"][:, h, -1], x0)
        r_y = gap_rank(y)
        out["r_y"][h] = r_y
        out["OV"][h] = None if nn(r_y, r_o) else 2 * d * dh - dh * (r_y + r_o - dh)
    r_vis = gap_rank(eff["W_U"].T)
    out["r_vis"] = r_vis
    pre = c["x1"][:, -1] @ eff["W_in"].T + eff["b_in"]
    for i in units:
        on = pre[:, i] > 0
        if not bool(on.any()):
            out["units"][i] = 2 * d + 1
            continue
        A = torch.cat([c["x1"][on, -1], torch.ones(int(on.sum()), 1)], 1)
        r_in = gap_rank(A)
        out["units"][i] = None if nn(r_in, r_vis) else 1 + (d + 1 - r_in) + (d - r_vis)
    return out
