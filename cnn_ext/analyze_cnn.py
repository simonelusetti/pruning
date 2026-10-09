"""
Summaries of run_cnn.py, in the format of her round-4 tables.

  1. stop point (% of k_eff removed) under the anchored valB rule, for several margins rho
     (fraction of the base validation loss), mean and range over seeds, with test acc there
  2. test accuracy and test loss / base at matched sparsity (linear interpolation along each path)
  3. which move kinds removed the parameters, up to the stop point (rho = 10%)
Uses her stop_points() so the cut is computed exactly as in her fair_comparison.
Writes results/summary.txt
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "NNpruning"))
from nnprune.experiments import stop_points  # noqa: E402

R = os.environ.get("CNN_RESULTS", os.path.join(HERE, "results"))
P = pd.read_csv(os.path.join(R, "paths.csv"))
S = pd.read_csv(os.path.join(R, "statics.csv"))
lines = []
say = lambda s="": (print(s), lines.append(s))
RHOS = (0.0, 0.01, 0.1, 1.0)

sp = stop_points({"paths": P, "statics": S}, rel_tols=RHOS)
order = ["W+S-cnn+rank | valA | obd", "S-cnn+rank | valA", "rank-only | valA", "W+S-cnn | valA | obd", "W+S-cnn | train | obd", "S-cnn | valA", "W+S-naive | valA | obd",
         "S-naive | valA", "W | valA | obd", "W | valA | magnitude",
         "magnitude", "obd", "random", "mean_ablation"]
seeds = sorted(P["seed"].unique())
say(f"seeds {seeds}")
base = P[P["step"] == 0].groupby("seed").first()
say("base networks: " + "; ".join(f"seed {s}: val loss {r.val_loss:.4f}, test acc {r.test_acc:.3f}" for s, r in base.iterrows()))

say("\n1. STOP POINT: % of parameters removed (mean over seeds, [min, max]); test acc at the stop (mean)")
hdr = f"{'method':26s}" + "".join(f"{'rho=' + format(r, 'g'):>22s}" for r in RHOS)
say(hdr)
for m in order:
    d = sp[sp["method"] == m]
    if d.empty:
        continue
    row = f"{m:26s}"
    for r in RHOS:
        x = d[np.isclose(d["rel_tol"], r)]
        f = 100 * x["frac_removed"]
        cens = "*" if x["censored"].any() else " "
        row += f"  {f.mean():5.1f} [{f.min():4.1f},{f.max():4.1f}] {x['test_acc'].mean():.3f}{cens}"
    say(row)
say("(* = for at least one seed the path never failed the test before its end: the stop is censored)")

say("\n2. MATCHED SPARSITY: test accuracy (mean over seeds) and test loss / base, at % removed")
fr = [0.1, 0.3, 0.5, 0.7, 0.9]
say(f"{'method':26s}" + "".join(f"{int(100 * f):>13d}%" for f in fr))


def interp(d, f, col):
    d = d.sort_values("frac_removed")
    x, y = d["frac_removed"].values, d[col].values
    return np.interp(f, x, y) if x.max() >= f else np.nan


allrows = pd.concat([P.assign(kind_="greedy"), S.assign(kind_="static")], ignore_index=True)
for m in order:
    d = allrows[allrows["method"] == m]
    if d.empty:
        continue
    row = f"{m:26s}"
    for f in fr:
        accs, lr = [], []
        for s_ in seeds:
            ds = d[d["seed"] == s_]
            if ds.empty:
                continue
            accs.append(interp(ds, f, "test_acc"))
            lr.append(interp(ds, f, "test_loss") / float(base.loc[s_, "test_loss"]))
        a, l = np.nanmean(accs) if not np.all(np.isnan(accs)) else np.nan, np.nanmean(lr) if not np.all(np.isnan(lr)) else np.nan
        row += "             -" if np.isnan(a) else f"  {a:.3f}/{l:5.1f}x"
    say(row)

say("\n3. MOVE KINDS up to the stop point (rho = 10%): % of the original parameters removed by each kind (sum of dk), mean over seeds")
cut = sp[np.isclose(sp["rel_tol"], 0.1) & (sp["kind"] == "greedy")]
for m in [o for o in order if o in set(P["method"])]:
    parts = []
    for s_ in seeds:
        d = P[(P["method"] == m) & (P["seed"] == s_)].sort_values("step")
        k_stop = float(cut[(cut["method"] == m) & (cut["seed"] == s_)]["k_eff"].iloc[0])
        k0 = float(d["k_eff"].iloc[0])
        d = d[(d["k_eff"] >= k_stop) & (d["step"] > 0)]
        parts.append(d.groupby("kind")["dk"].sum() / k0 * 100)
    mean = pd.concat(parts, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    say(f"  {m:26s} " + ", ".join(f"{k} {v:.1f}" for k, v in mean.items() if v > 0.05))

with open(os.path.join(R, "summary.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
