"""Builds fair_comparison.ipynb: where does the advantage come from?"""
import nbformat as nbf

FINDINGS = open("fair_findings.md").read() if __import__("os").path.exists("fair_findings.md") else "_(run, then write findings)_"

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(r"""# Fair comparison: where does the advantage come from?

Two questions:

1. Is our advantage only the greedy validation-driven search, or does it come
   from the moves themselves?
2. Does it come from **composite / structural moves** rather than working at
   the level of single parameters?

**Design.** Two factors are crossed. Berman et al.'s iterative Fisher-distance
score $\sqrt{F_{kk}(\theta)}\,|\theta_k|$, computed with their own code,
proposes every single-parameter move.

| factor | levels |
|---|---|
| granularity | **W**: single parameters only · **S**: structural moves only (merge, mean, saturation, step, head / attention / rank moves) · **W+S**: both |
| selection data | **static**: one-shot ranking, no data in the loop · **train**: each move chosen on training data · **valA**: each move chosen on held-out data (our rule) |

**One stopping rule for everybody.** The validation split is halved once, at
random. **valA** may be used to choose moves. **valB** is used by every method,
greedy or one-shot, only for the same anchored paired 2σ stop test against the
original model. The test split is reported only.

**Tie-break:** fewest remaining parameters, then lowest loss. **Criterion:**
per-sample log loss. No retraining anywhere.

The *static* row covers:
- W: Berman et al.'s four schemes, plus magnitude, OBD, Wanda and random;
- S: classical mean-ablation.

`W | valA | magnitude` and `W+S | valA | obd` are the same design with a
different single-parameter proposer.""")

code(r"""import os, sys
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath("."))
from nnprune import experiments as E
RECOMPUTE = False      # True: rerun (MLP ~30 min for 3 seeds, transformer ~1.5 h on a laptop CPU)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")

PAL  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK2, GRID = "#52514e", "#e4e3de"
plt.rcParams.update({"figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5})

def stop_table(fair, rel_tol=0.0):
    S = E.stop_points(fair, rel_tols=(rel_tol,)); g = S.groupby("method")
    T = pd.DataFrame({"% removed": 100 * g["frac_removed"].mean(), "sd": 100 * g["frac_removed"].std(),
                      "params left": g["k_eff"].mean(), "test loss": g["test_loss"].mean(),
                      "test acc": g["test_acc"].mean()})
    return T.sort_values("% removed", ascending=False)

def grid_table(fair, rel_tol=0.0, static_W="fdist_iterative", static_S="mean_ablation"):
    T = stop_table(fair, rel_tol)["% removed"]
    get = lambda m: T.get(m, np.nan)
    return pd.DataFrame({
        "static (one-shot)": [get(static_W), get(static_S), np.nan],
        "train-selected":    [get("W | train | fdist"), get("S | train"), get("W+S | train | fdist")],
        "valA-selected":     [get("W | valA | fdist"), get("S | valA"), get("W+S | valA | fdist")],
    }, index=["W  (single parameters)", "S  (structural)", "W+S"]).rename_axis(
        f"% removed at the valB stop, margin {100 * rel_tol:g}% of base loss")

def margin_table(fair, methods, rel_tols=(0.0, 0.005, 0.01, 0.02)):
    SP = E.stop_points(fair, rel_tols=rel_tols)
    T = SP[SP.method.isin(methods)].groupby(["method", "rel_tol"])["frac_removed"].mean().unstack() * 100
    T.columns = [f"margin {100 * c:g}%" for c in T.columns]
    return T.loc[[m for m in methods if m in T.index]]

def at_sparsity(fair, metric, grid=(0.1, 0.3, 0.5, 0.7, 0.9)):
    # value of `metric` at the last point with <= f removed, mean over seeds
    D = pd.concat([fair["paths"], fair["statics"]])
    rows = {}
    for m, dm in D.groupby("method"):
        vals = []
        for f in grid:
            per_seed = []
            for _, ds in dm.groupby("seed"):
                d = ds[ds.frac_removed <= f + 1e-9].sort_values("frac_removed")
                per_seed.append(d[metric].iloc[-1] if len(d) and d.frac_removed.max() >= f - 0.05 else np.nan)
            vals.append(np.nanmean(per_seed) if np.isfinite(per_seed).any() else np.nan)
        rows[m] = vals
    return pd.DataFrame(rows, index=[f"{int(100 * f)}%" for f in grid]).T

def curves(fair, metric, ylabel, title, seed, logy=False):
    show = ["W+S | valA | fdist", "S | valA", "W | valA | fdist", "W+S | train | fdist", "W | train | fdist",
            "fdist_iterative"]
    D = pd.concat([fair["paths"], fair["statics"]]); D = D[D.seed == seed]
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for k, m in enumerate(show):
        d = D[D.method == m].sort_values("frac_removed")
        if d.empty: continue
        ax.plot(100 * d.frac_removed, d[metric], color=PAL[k], lw=2.0 if k == 0 else 1.5, label=m)
        sp = E.stop_points({"paths": fair["paths"][fair["paths"].seed == seed],
                            "statics": fair["statics"][fair["statics"].seed == seed]}, rel_tols=(0.01,))
        r = sp[sp.method == m].iloc[0]
        ax.plot(100 * r.frac_removed, r[metric], "o", ms=8, mfc="#fcfcfb", mec=PAL[k], mew=2)
    if logy: ax.set_yscale("log")
    ax.set_xlabel("% of parameters removed (k_eff)   ·   ring = valB stop, margin 1%"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    ax.legend(loc="best"); plt.tight_layout(); plt.show()""")

md(r"""## Moons MLP (3 seeds)""")
code(r"""fm = E.run_fair("mlp", recompute=RECOMPUTE)
display(grid_table(fm, 0.0))
grid_table(fm, 0.01)""")
md(r"""**How the stop depends on the margin.** The margin is $\delta$ in
$\overline{\Delta\ell}\le 2\,\widehat{SE}+\delta$, with $\delta$ a fraction of
the base validation loss. With $\delta=0$ the test also rejects tiny but
systematic changes: their paired SE is tiny too.""")
code(r"""KEY = ["W+S | valA | fdist", "S | valA", "W | valA | fdist", "W+S | train | fdist", "S | train",
       "W | train | fdist", "W | valA | magnitude", "W+S | valA | obd",
       "fdist_iterative", "fdist_exact", "fdist_global", "fdist_one_shot", "magnitude", "mean_ablation"]
margin_table(fm, KEY)""")
code(r"""stop_table(fm, 0.01)""")
md(r"""**Test loss at matched sparsity** (mean over seeds; blank where a method's
curve does not reach that sparsity before it stops or ends).""")
code(r"""at_sparsity(fm, "test_loss").sort_values("50%")""")
code(r"""curves(fm, "test_loss", "test log loss", "MLP, seed 123", 123, logy=True)""")

md(r"""## One-layer transformer, $a+b \bmod 59$ (seed 0)""")
md(r"""The base loss is $8\cdot10^{-4}$ on a deterministic task, so margins of 0
and 1% coincide. Larger margins are shown too: 100% means the loss may double,
to about $1.6\cdot10^{-3}$.""")
code(r"""ft = E.run_fair("transformer", recompute=RECOMPUTE)
display(grid_table(ft, 0.0))
grid_table(ft, 1.0)""")
code(r"""margin_table(ft, KEY, rel_tols=(0.0, 0.01, 0.1, 1.0, 10.0))""")
code(r"""stop_table(ft, 1.0)""")
code(r"""display(at_sparsity(ft, "test_acc").sort_values("50%", ascending=False))
at_sparsity(ft, "test_loss").sort_values("50%")""")
code(r"""curves(ft, "test_acc", "test accuracy", "Transformer, seed 0", 0)""")

md("## Findings\n\n" + FINDINGS)

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "fair_comparison.ipynb")
print("written fair_comparison.ipynb with", len(C), "cells")
