"""Stop points (% of the trained model's parameters removed, Stage 1 included) under the
anchored valB test  mean dL <= 2 SE + rho * base val loss,  for every finished run in
results_gpt_prune/.  Usage: python3 analyze_gpt_prune.py"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RHOS = (0.0, 0.005, 0.01, 0.02, 0.05)


def stops(d):
    d = d.sort_values("step")
    base = d.val_loss.iloc[0]
    out = {}
    for rt in RHOS:
        ok = (d.stop_mean <= 2 * d.stop_se + rt * base).values
        ff = int(np.argmax(~ok)) if (~ok).any() else len(d)
        r = d.iloc[max(ff - 1, 0)]
        out[rt] = (100 * r.frac_removed_orig, r.test_loss, int(r.step), ff == len(d))
    return out


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "results_gpt_prune", "*.csv"))):
        d = pd.read_csv(f)
        for rt, (pct, tl, st, cens) in stops(d).items():
            rows.append(dict(L=int(d.L.iloc[0]), seed=int(d.seed.iloc[0]), method=d.method.iloc[0], rho=rt,
                             removed=pct, test_loss=tl, step=st, censored=cens))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("stop point, % of parameters removed (Stage 1 included), by margin rho:")
    print(df.pivot_table(index=["L", "method", "seed"], columns="rho", values="removed").round(1).to_string())

    print("\nmean over seeds [min, max]:")
    g = df.groupby(["L", "method", "rho"]).removed
    t = (g.mean().round(1).astype(str) + " [" + g.min().round(1).astype(str) + ", " + g.max().round(1).astype(str) + "]")
    print(t.unstack("rho").to_string())

    paths = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "results_gpt_prune", "*.csv")))])
    print("\nstream moves chosen (whole path / before the rho = 5% stop):")
    for (m, s), d in paths.groupby(["method", "seed"]):
        if "stream" in m:
            st = stops(d)[0.05][2]
            print(f"  {m:32s} seed {s}: {(d.kind == 'stream_mean').sum():3d} / {((d.kind == 'stream_mean') & (d.step <= st)).sum():3d}")

    print("\nparameters removed by move kind up to the rho = 5% stop (% of p, mean over seeds; Stage 1 = 2.4):")
    comp = []
    for (m, s), d in paths.groupby(["method", "seed"]):
        st = stops(d)[0.05][2]
        dd = d[(d.step >= 1) & (d.step <= st)]
        for k, gk in dd.groupby("kind"):
            comp.append(dict(method=m, seed=s, kind=k, pct=100 * gk.dk.sum() / d.p_orig.iloc[0], n=len(gk)))
    comp = pd.DataFrame(comp).groupby(["method", "kind"])[["pct", "n"]].sum() / 3
    print(comp.round(1).unstack("method").fillna("").to_string())

    print("\ntest loss / unpruned test loss at matched sparsity (linear interpolation, mean over seeds):")
    mrows = []
    for (m, s), d in paths.groupby(["method", "seed"]):
        d = d.sort_values("step")
        t0 = d.test_loss.iloc[0]
        for f in (0.1, 0.2, 0.3, 0.4):
            mrows.append(dict(method=m, seed=s, at=f, ratio=float(np.interp(f, d.frac_removed_orig, d.test_loss)) / t0
                              if d.frac_removed_orig.max() >= f else np.nan))
    print(pd.DataFrame(mrows).groupby(["method", "at"]).ratio.mean().unstack("at").round(3).to_string())
