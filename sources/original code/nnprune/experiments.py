"""
Experiment drivers with on-disk caching (CSV / JSON in `outdir`).

    run_mlp(...)          two-moons sigmoid MLP (the pruningNN setting)
    run_transformer(...)  1-layer transformer on a + b (mod p), trained from scratch

Each returns a dict of DataFrames and writes them to `outdir`.  With
recompute=False an existing result is loaded instead of recomputed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from .core import PruneConfig, baseline_path, prune_path, summarize, unit_baseline_path
from . import mlp as M
from . import transformer as T
from . import spectrum as S

BASELINES = ("magnitude", "obd", "wanda", "random")

# ---- MLP -------------------------------------------------------------------
MLP_BASE = dict(weight_chunk=1, weight_per_site=8, n_struct=32, n_merge=16, refresh_every=10)
MLP_VARIANTS = {
    "boundary [k_eff>layer]": dict(),
    "boundary [k_eff]": dict(tie_break="k_eff"),
    "boundary [layer]": dict(tie_break="layer"),
    "boundary [no tie-break]": dict(tie_break="none"),
    "weights only [k_eff>layer]": dict(use_struct=False),
    "classical moves [k_eff>layer]": dict(struct_kinds=("unit_mean", "unit_merge")),
    "sat+step moves [k_eff>layer]": dict(struct_kinds=("unit_sat", "unit_step")),
}

# ---- transformer -------------------------------------------------------------
TF_BASE = dict(weight_chunk=64, weight_per_site=3, n_struct=24, n_merge=16,
               screen_n=512, refresh_every=20)
TF_VARIANTS = {
    "boundary [k_eff>layer] loss": dict(criterion="loss"),
    "boundary [k_eff>layer] error": dict(criterion="error"),
    "boundary [layer] error": dict(criterion="error", tie_break="layer"),
    "weights only [k_eff>layer] error": dict(criterion="error", use_struct=False),
}


def _cached(outdir, names):
    return all(os.path.exists(os.path.join(outdir, n)) for n in names)


def _save(outdir, frames: dict):
    os.makedirs(outdir, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)


def _load(outdir, names):
    return {n: pd.read_csv(os.path.join(outdir, f"{n}.csv")) for n in names}


def _run_all(adapter, s0, base_kw, variants, baselines, unit_baseline, seed, verbose, tag):
    paths, bls = {}, {}
    for name, kw in variants.items():
        cfg = PruneConfig(**{**base_kw, **kw, "seed": seed, "verbose": verbose})
        t = time.time()
        df, _ = prune_path(adapter, s0, cfg, label=name)
        df["criterion"] = cfg.criterion
        paths[name] = df
        print(f"[{tag} seed {seed}] {name}: {len(df) - 1} steps, "
              f"{100 * df['frac_removed'].iloc[-1]:.1f}% of k_eff removed, {time.time() - t:.0f}s")
    cfg = PruneConfig(**{**base_kw, "seed": seed, "verbose": False})
    for b in baselines:
        bls[b] = baseline_path(adapter, s0, b, cfg)
    if unit_baseline:
        bls["mean_ablation"] = unit_baseline_path(adapter, s0, cfg)
    return paths, bls


def _stack(d: dict, seed: int, key: str) -> pd.DataFrame:
    out = []
    for name, df in d.items():
        df = df.copy()
        df[key] = name
        df["seed"] = seed
        out.append(df)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------

def run_mlp(outdir="results/mlp", seeds=(123,), variants=None, baselines=BASELINES,
            unit_baseline=True, epochs=2500, recompute=False, verbose=False) -> dict:
    names = ["paths", "baselines", "summary", "base", "train_hist"]
    if not recompute and _cached(outdir, [f"{n}.csv" for n in names]):
        return _load(outdir, names)
    variants = MLP_VARIANTS if variants is None else variants
    P, B, Su, Ba, H = [], [], [], [], []
    for seed in seeds:
        data, _ = M.make_moons_data(seed=seed)
        ad = M.MLPAdapter(data)
        s0, hist = M.train_mlp(ad, epochs=epochs, seed=seed, verbose=False)
        torch.save(s0, os.path.join(_mk(outdir), f"base_state_seed{seed}.pt"))
        Ba.append({"seed": seed, "k_eff": ad.k_eff(s0),
                   **{f"{sp}_{k}": v for sp in ("train", "val", "test")
                      for k, v in ad.metrics(s0, sp).items()}})
        H.append(pd.DataFrame(hist).assign(seed=seed))
        paths, bls = _run_all(ad, s0, MLP_BASE, variants, baselines, unit_baseline, seed, verbose, "mlp")
        P.append(_stack(paths, seed, "method"))
        B.append(_stack(bls, seed, "method"))
        Su.append(summarize(paths, bls, "loss").assign(seed=seed))
    out = {"paths": pd.concat(P, ignore_index=True), "baselines": pd.concat(B, ignore_index=True),
           "summary": pd.concat(Su, ignore_index=True), "base": pd.DataFrame(Ba),
           "train_hist": pd.concat(H, ignore_index=True)}
    _save(outdir, out)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"MLP_BASE": MLP_BASE, "variants": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                                          for kk, vv in v.items()}
                                                      for k, v in variants.items()},
                   "seeds": list(seeds), "epochs": epochs,
                   "defaults": asdict(PruneConfig())}, f, indent=1, default=str)
    return out


def mlp_spectrum(outdir="results/mlp", seed=123, n_fresh=8000, recompute=False) -> dict:
    """Full expected GGN of the trained moons MLP on a FRESH large sample (n > p,
    so the null space is not a sample-size artefact), and the gap verdict."""
    path = os.path.join(outdir, f"spectrum_seed{seed}.json")
    if not recompute and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data, sc = M.make_moons_data(seed=seed)
    ad = M.MLPAdapter(data)
    s0 = torch.load(os.path.join(outdir, f"base_state_seed{seed}.pt"), weights_only=False)
    Xf, _ = M.moons_fresh_sample(sc, n=n_fresh)
    block = {k: torch.ones_like(v, dtype=torch.bool) for k, v in s0.params.items()}
    F = S.block_fisher(ad, s0, block, Xf, batch=500)
    ev = np.linalg.eigvalsh(F)
    v = S.classify_spectrum(ev, noise_floor="auto")
    out = {"p": int(v.p), "n": n_fresh, "regime": v.regime, "null_dim": int(v.null_dim),
           "plateau_decades": float(v.plateau_decades), "reason": v.reason,
           "noise_floor": float(S.noise_floor_of(ev)),
           "null_dim_without_floor": int(S.classify_spectrum(ev).null_dim),
           "sweep": S.min_decades_sweep(ev),
           "eigenvalues": [float(x) for x in np.sort(ev)[::-1]]}
    with open(path, "w") as f:
        json.dump(out, f)
    return out


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------

def run_transformer(outdir="results/transformer", seeds=(0,), p=59, train_frac=0.5, steps=3000,
                    variants=None, baselines=BASELINES, unit_baseline=True, gauge=True,
                    recompute=False, verbose=False, model_kw=None) -> dict:
    names = ["paths", "baselines", "summary_loss", "summary_error", "base", "train_hist"]
    if gauge:
        names.append("gauge")
    if not recompute and _cached(outdir, [f"{n}.csv" for n in names]):
        return _load(outdir, names)
    variants = TF_VARIANTS if variants is None else variants
    model_kw = model_kw or {}
    P, B, SL, SE, Ba, H, G = [], [], [], [], [], [], []
    for seed in seeds:
        data = T.make_modadd_data(p=p, train_frac=train_frac, seed=seed)
        ad = T.TransformerAdapter(data, p=p, **model_kw)
        s0, hist = T.train_transformer(ad, steps=steps, seed=seed, log_every=250, verbose=False)
        torch.save(s0, os.path.join(_mk(outdir), f"base_state_seed{seed}.pt"))
        Ba.append({"seed": seed, "k_eff": ad.k_eff(s0),
                   **{f"{sp}_{k}": v for sp in ("train", "val", "test")
                      for k, v in ad.metrics(s0, sp).items()}})
        H.append(pd.DataFrame(hist).assign(seed=seed))
        if gauge:
            G.append(transformer_gauge_table(ad, s0, data).assign(seed=seed))
        paths, bls = _run_all(ad, s0, TF_BASE, variants, baselines, unit_baseline, seed, verbose, "tf")
        P.append(_stack(paths, seed, "method"))
        B.append(_stack(bls, seed, "method"))
        lp = {k: v for k, v in paths.items() if v["criterion"].iloc[0] == "loss"}
        ep = {k: v for k, v in paths.items() if v["criterion"].iloc[0] == "error"}
        SL.append(summarize(lp, bls, "loss").assign(seed=seed))
        SE.append(summarize(ep, bls, "error").assign(seed=seed))
    out = {"paths": pd.concat(P, ignore_index=True), "baselines": pd.concat(B, ignore_index=True),
           "summary_loss": pd.concat(SL, ignore_index=True),
           "summary_error": pd.concat(SE, ignore_index=True),
           "base": pd.DataFrame(Ba), "train_hist": pd.concat(H, ignore_index=True)}
    if gauge:
        out["gauge"] = pd.concat(G, ignore_index=True)
    _save(outdir, out)
    return out


def transformer_gauge_table(ad, s0, data, n_units=4) -> pd.DataFrame:
    """Gap verdicts for every head's QK and OV block and a few MLP units, next to
    the null dimensions predicted from data-subspace ranks."""
    Xall = torch.cat([data[k][0] for k in ("train", "val", "test")])
    with torch.no_grad():
        _, c = ad.forward_full(ad.effective(s0), s0.struct, Xall)
    frac_on = (c["act"][:, -1] > 0).float().mean(0)
    units = sorted(set([int(frac_on.argmin()), int(frac_on.argmax())] +
                       list(range(n_units - 2))))
    pred = S.transformer_predicted_nulls(ad, s0, Xall, units=units)
    rows = []
    for h in range(ad.H):
        bl = S.head_blocks(ad, s0, h)
        for nm in ("QK", "OV"):
            r = S.classify_block(ad, s0, bl[nm], Xall)
            rows.append({"block": f"head{h}.{nm}", "p": r["p"], "regime": r["regime"],
                         "null_dim": r["null_dim"],
                         "predicted_null": pred["QK"] if nm == "QK" else pred["OV"][h],
                         "symmetry_null": (2 * ad.d * ad.dh - ad.d) if nm == "QK" else ad.dh ** 2,
                         "plateau_decades": r["plateau_decades"],
                         "n_at_noise_floor": r["n_at_noise_floor"],
                         "null_dim_without_floor": r["null_dim_without_floor"],
                         "null_dims_over_min_decades": str(r["null_dims_over_min_decades"])})
    for i in units:
        r = S.classify_block(ad, s0, S.unit_block(s0, i), Xall)
        rows.append({"block": f"unit{i} (active {float(frac_on[i]):.2f})", "p": r["p"],
                     "regime": r["regime"], "null_dim": r["null_dim"],
                     "predicted_null": pred["units"][i], "symmetry_null": 1,
                     "plateau_decades": r["plateau_decades"],
                     "n_at_noise_floor": r["n_at_noise_floor"],
                     "null_dim_without_floor": r["null_dim_without_floor"],
                     "null_dims_over_min_decades": str(r["null_dims_over_min_decades"])})
    df = pd.DataFrame(rows)
    df.attrs["ranks"] = {k: pred[k] for k in ("r_K", "r_o", "r_vis")}
    return df


# --------------------------------------------------------------------------
# Berman et al. (2026) Fisher-distance baselines, added to cached runs
# --------------------------------------------------------------------------

FDIST_MLP = ("fdist_one_shot", "fdist_iterative", "fdist_global", "fdist_exact")
FDIST_TF = ("fdist_one_shot", "fdist_iterative", "fdist_global")   # exact: ~57k probes x K per step
TF_FDIST_FRACS = np.round(np.concatenate([np.arange(0, 0.5, 0.02), np.arange(0.5, 0.96, 0.05)]), 4)


def run_fdist(kind="mlp", outdir=None, seeds=None, schemes=None, fracs=None,
              recompute=False, verbose=False) -> dict:
    """Berman et al. schemes on the cached base models of run_mlp / run_transformer.
    Writes fdist_baselines.csv and fdist_summary_{loss,error}.csv next to them."""
    from .fdist_baseline import fdist_path
    outdir = outdir or f"results/{kind}"
    names = ["fdist_baselines", "fdist_summary_loss", "fdist_summary_error"]
    if not recompute and _cached(outdir, [f"{n}.csv" for n in names]):
        return _load(outdir, names)
    base = pd.read_csv(os.path.join(outdir, "base.csv"))
    seeds = tuple(base["seed"]) if seeds is None else seeds
    B = []
    for seed in seeds:
        if kind == "mlp":
            data, _ = M.make_moons_data(seed=int(seed))
            ad = M.MLPAdapter(data)
            sch, exclude, kw = schemes or FDIST_MLP, (), MLP_BASE
        else:
            data = T.make_modadd_data(p=59, train_frac=0.5, seed=int(seed))
            ad = T.TransformerAdapter(data, p=59)
            sch, exclude, kw = schemes or FDIST_TF, ("W_pos",), TF_BASE
            fracs = TF_FDIST_FRACS if fracs is None else fracs
        s0 = torch.load(os.path.join(outdir, f"base_state_seed{seed}.pt"), weights_only=False)
        cfg = PruneConfig(**{**kw, "seed": int(seed), "verbose": False})
        for scheme in sch:
            t = time.time()
            df = fdist_path(ad, s0, scheme, cfg, fracs=fracs, exclude=exclude, verbose=verbose)
            df["seed"] = int(seed)
            B.append(df)
            print(f"[{kind} seed {seed}] {scheme}: {time.time() - t:.0f}s")
    Bdf = pd.concat(B, ignore_index=True)
    out = {"fdist_baselines": Bdf}
    for crit in ("loss", "error"):
        rows = []
        for (seed, m), df in Bdf.groupby(["seed", "method"]):
            ok = df[df[f"within_tolerance_{crit}"]]
            r = ok.iloc[-1] if len(ok) else df.iloc[0]
            rows.append({"method": m, "frac_removed": r["frac_removed"], "k_eff": int(r["k_eff"]),
                         "val_loss": r["val_loss"], "test_loss": r["test_loss"],
                         "test_acc": r["test_acc"], "steps": np.nan, "seed": int(seed)})
        out[f"fdist_summary_{crit}"] = pd.DataFrame(rows)
    _save(outdir, out)
    return out


# --------------------------------------------------------------------------
# Fair comparison: where does the advantage come from?
# --------------------------------------------------------------------------
#
# Two factors, crossed, with Berman et al.'s iterative Fisher-distance score as
# the single-parameter proposer everywhere:
#   granularity  W = single parameters only | S = structural moves only | W+S = both
#   selection    which data chooses each move: 'train' (no held-out data in the
#                loop) or 'valA' (held-out, our rule)
# Every method -- greedy or one-shot -- stops with the SAME anchored paired 2 sigma
# test on 'valB', a half of the validation split that no method uses to choose
# anything (cut at the first failure; stop_points() recomputes the cut for a
# practical margin delta, since the pure 2 sigma rule also rejects tiny but
# systematic changes).  The select split only ranks candidates and
# forms tie sets (select_admissibility=False), so a larger split cannot end a
# path earlier through a more powerful test.  Greedy paths are continued 60 steps
# past their stop point so that curves can be compared at matched sparsity.  Tie-break: fewest remaining parameters, then lowest loss
# (tie_break='k_eff').  Criterion: per-sample log loss.

FAIR_COMMON = dict(tie_break="k_eff", criterion="loss", stop_split="valB", stop_mode="mark",
                   select_admissibility=False, post_stop_steps=10 ** 9, max_frac=0.97)


def stop_points(fair: dict, rel_tols=(0.0, 0.005, 0.01, 0.02), n_sigma=2.0) -> pd.DataFrame:
    """Stop point of every method under the anchored valB test with a practical
    margin: a point passes iff  mean(l_m - l_0) <= n_sigma * SE + delta,  where
    delta = rel_tol * (base validation loss).  delta = 0 is the pure 2 sigma rule.
    Cut at the first failure, for greedy paths and one-shot grids alike."""
    rows = []
    for kind, D, mcol, scol in (("greedy", fair["paths"], "stop_mean", "stop_se"),
                                ("static", fair["statics"], "anch_mean_loss", "anch_se_loss")):
        for (seed, m), d in D.groupby(["seed", "method"]):
            d = d.sort_values("step" if kind == "greedy" else "weight_frac" if "weight_frac" in d and d["weight_frac"].notna().all() else "frac_removed")
            base = float(d["val_loss"].iloc[0])
            for rt in rel_tols:
                ok = (d[mcol] <= n_sigma * d[scol] + rt * base).values
                ff = int(np.argmax(~ok)) if (~ok).any() else len(d)
                r = d.iloc[max(ff - 1, 0)]
                rows.append({"seed": seed, "method": m, "kind": kind, "rel_tol": rt,
                             "frac_removed": r["frac_removed"], "k_eff": r["k_eff"],
                             "test_loss": r["test_loss"], "test_acc": r["test_acc"],
                             "censored": ff == len(d)})
    return pd.DataFrame(rows)


def fair_variants() -> dict:
    v = {}
    for sel in ("train", "valA"):
        v[f"W | {sel} | fdist"] = dict(select_split=sel, weight_score="fdist", use_struct=False)
        v[f"S | {sel}"] = dict(select_split=sel, weight_per_site=0)
        v[f"W+S | {sel} | fdist"] = dict(select_split=sel, weight_score="fdist")
    v["W | valA | magnitude"] = dict(select_split="valA", weight_score="magnitude", use_struct=False)
    v["W+S | valA | obd"] = dict(select_split="valA", weight_score="obd")
    return v


def split_validation(data: dict, seed: int = 0) -> dict:
    """valA / valB: a fixed random halving of the validation split."""
    X, y = data["val"]
    g = torch.Generator().manual_seed(1000 + int(seed))
    perm = torch.randperm(y.shape[0], generator=g)
    h = y.shape[0] // 2
    data = dict(data)
    data["valA"] = (X[perm[:h]], y[perm[:h]])
    data["valB"] = (X[perm[h:]], y[perm[h:]])
    return data


def run_fair(kind="mlp", outdir=None, seeds=None, variants=None, statics=None,
             recompute=False, verbose=False) -> dict:
    """The crossed design above on the cached base models of run_mlp /
    run_transformer, plus every one-shot baseline re-evaluated with the valB stop."""
    from .core import unit_baseline_path
    from .fdist_baseline import fdist_path
    src = f"results/{kind}"
    outdir = outdir or f"results/fair_{kind}"
    names = ["paths", "statics", "summary"]
    if not recompute and _cached(outdir, [f"{n}.csv" for n in names]):
        return _load(outdir, names)
    base = pd.read_csv(os.path.join(src, "base.csv"))
    seeds = tuple(int(x) for x in base["seed"]) if seeds is None else seeds
    variants = fair_variants() if variants is None else variants
    P, B, Su = [], [], []
    for seed in seeds:
        if kind == "mlp":
            data, _ = M.make_moons_data(seed=seed)
            ad = M.MLPAdapter(split_validation(data, seed))
            kw, exclude = MLP_BASE, ()
            fd = statics if statics is not None else FDIST_MLP
            fracs = None
        else:
            data = T.make_modadd_data(p=59, train_frac=0.5, seed=seed)
            ad = T.TransformerAdapter(split_validation(data, seed), p=59)
            kw, exclude = TF_BASE, ("W_pos",)
            fd = statics if statics is not None else FDIST_TF
            fracs = TF_FDIST_FRACS
        s0 = torch.load(os.path.join(src, f"base_state_seed{seed}.pt"), weights_only=False)
        paths, bls = {}, {}
        for name, v in variants.items():
            cfg = PruneConfig(**{**kw, **FAIR_COMMON, **v, "seed": seed, "verbose": verbose})
            t = time.time()
            df, _ = prune_path(ad, s0, cfg, label=name)
            paths[name] = df
            print(f"[fair {kind} seed {seed}] {name}: {len(df) - 1} steps, "
                  f"{100 * df['frac_removed'].iloc[-1]:.1f}% removed, {time.time() - t:.0f}s", flush=True)
        cfg = PruneConfig(**{**kw, **FAIR_COMMON, "seed": seed, "verbose": False})
        for b in BASELINES:
            bls[b] = baseline_path(ad, s0, b, cfg)
        bls["mean_ablation"] = unit_baseline_path(ad, s0, cfg)
        for sch in fd:
            t = time.time()
            bls[sch] = fdist_path(ad, s0, sch, cfg, fracs=fracs, exclude=exclude)
            print(f"[fair {kind} seed {seed}] {sch}: {time.time() - t:.0f}s", flush=True)
        P.append(_stack(paths, seed, "method"))
        B.append(_stack(bls, seed, "method"))
        Su.append(summarize(paths, bls, "loss").assign(seed=seed))
        out = {"paths": pd.concat(P, ignore_index=True), "statics": pd.concat(B, ignore_index=True),
               "summary": pd.concat(Su, ignore_index=True)}
        _save(outdir, out)                     # checkpoint after every seed
    return out


def _mk(d):
    os.makedirs(d, exist_ok=True)
    return d
