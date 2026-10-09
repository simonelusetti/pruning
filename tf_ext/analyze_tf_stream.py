"""Stop points, move composition and matched-sparsity accuracy for run_tf_stream.py,
next to her own transformer paths (NNpruning/results/fair_transformer)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "NNpruning"))
from nnprune.experiments import stop_points  # noqa: E402

OUT = os.environ.get("TF_STREAM_RESULTS", os.path.join(HERE, "results_tf_stream"))
HER = os.path.join(os.path.dirname(HERE), "NNpruning", "results", "fair_transformer")
TOLS = (0.0, 0.01, 0.1, 1.0, 10.0)

mine = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))
                  if f.endswith(".csv") and not f.startswith(("stop", "compo", "matched"))], ignore_index=True)
her = pd.read_csv(os.path.join(HER, "paths.csv"))
her = her[her.method.isin(["W+S | valA | fdist"])]
paths = pd.concat([mine, her], ignore_index=True)
statics = pd.read_csv(os.path.join(HER, "statics.csv"))
sp = stop_points({"paths": paths, "statics": statics}, rel_tols=TOLS)
sp = sp[sp.kind == "greedy"]
tab = sp.pivot(index="method", columns="rel_tol", values="frac_removed") * 100
print("stop point, % of k_eff removed, by margin rho (fraction of base val loss):")
print(tab.round(1).to_string())
print("\ntest acc at those stop points:")
print(sp.pivot(index="method", columns="rel_tol", values="test_acc").round(3).to_string())
sp.to_csv(os.path.join(OUT, "stop_points.csv"), index=False)

rows = []
for (m), d in paths.groupby("method"):
    d = d.sort_values("step")
    k0 = d.k_eff.iloc[0]
    for rt in (0.01, 1.0):
        base = d.val_loss.iloc[0]
        ok = (d.stop_mean <= 2 * d.stop_se + rt * base).values
        ff = int(np.argmax(~ok)) if (~ok).any() else len(d)
        dd = d.iloc[1:max(ff, 1)]
        for kind, g in dd.groupby("kind"):
            rows.append(dict(method=m, rho=rt, kind=kind, pct=100 * g.dk.sum() / k0, n=len(g)))
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(OUT, "composition.csv"), index=False)
print("\nparameters removed by move kind up to the stop point (% of p, number of moves):")
for rt in (0.01, 1.0):
    c = comp[comp.rho == rt]
    print(f"rho = {rt}")
    print(c.assign(cell=c.pct.round(1).astype(str) + " (" + c.n.astype(str) + ")")
          .pivot(index="kind", columns="method", values="cell").fillna("").to_string())

mrows = []
for m, d in paths.groupby("method"):
    d = d.sort_values("frac_removed")
    for f in (0.5, 0.7, 0.8, 0.9):
        mrows.append(dict(method=m, at=f, test_acc=float(np.interp(f, d.frac_removed, d.test_acc)) if d.frac_removed.max() >= f else np.nan))
mt = pd.DataFrame(mrows).pivot(index="method", columns="at", values="test_acc")
mt.to_csv(os.path.join(OUT, "matched.csv"))
print("\ntest accuracy at matched sparsity:")
print(mt.round(3).to_string())
