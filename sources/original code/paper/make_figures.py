"""
Figures and matched-sparsity numbers for paper/main.tex, from the cached runs.

    cd NNpruning && python paper/make_figures.py

Writes paper/figures/{mlp_loss,tf_accuracy,head_spectra}.pdf and prints the
numbers quoted in the text.  Curves are linear interpolations of each method's
(k_eff removed, metric) path on a common grid, averaged over seeds; stop points
use experiments.stop_points (anchored paired 2 sigma test on valB plus margin).
The head-0 Fisher spectra are computed once and cached in
results/transformer/head0_spectra.npz.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from nnprune import experiments as E, spectrum as S, transformer as T  # noqa: E402

OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

ACCENT, INK, QUIET, GRID = "#2A78D6", "#1F1E1C", "#6B6A65", "#E4E3DE"
MUTED = "#8F8D86"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": QUIET, "xtick.color": QUIET, "ytick.color": QUIET,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
})

GRID_PCT = np.arange(0, 0.961, 0.02)

SERIES = {   # slug: (method in the cached runs, label, colour, alpha, lw, dash)
    "ws": ("W+S | valA | fdist", "composite moves (W+S)", ACCENT, 1.0, 2.2, "-"),
    "s": ("S | valA", "structural only (S)", MUTED, 0.55, 1.4, (0, (9, 2.5))),
    "w": ("W | valA | fdist", "single parameters, same rule (W)", MUTED, 1.0, 1.4, "-"),
    "berman": ("fdist_iterative", "Berman et al. (2026), iterative", MUTED, 0.8, 1.4, (0, (5, 3))),
    "mag": ("magnitude", "magnitude", MUTED, 0.6, 1.4, (0, (1.2, 2.2))),
}


def curve(D, method, metric, seeds, base=None):
    out = []
    for s in seeds:
        d = D[(D.method == method) & (D.seed == s)].sort_values("frac_removed")
        x = d.frac_removed.values
        y = d[metric].values.astype(float) / (base[s] if base else 1.0)
        xs, idx = np.unique(x, return_index=True)
        out.append(np.where(GRID_PCT <= xs.max() + 1e-9, np.interp(GRID_PCT, xs, y[idx]), np.nan))
    return np.nanmean(np.vstack(out), 0) if len(out) > 1 else out[0]


def stops(fair, rel_tol, base=None):
    """Mean over seeds of the stop point and of the metric there (test-loss ratio
    for the MLP, test accuracy for the transformer), from experiments.stop_points."""
    S = E.stop_points(fair, rel_tols=(rel_tol,))
    S["ratio"] = [r.test_loss / base[int(r.seed)] for r in S.itertuples()] if base else np.nan
    return S.groupby("method")[["frac_removed", "ratio", "test_acc"]].mean()


def at(c, f):
    c = np.asarray(c)
    ok = np.isfinite(c)
    return float(np.interp(f, GRID_PCT[ok], c[ok]))


# --------------------------------------------------------------------------- MLP
fm = E.run_fair("mlp")
Dm = pd.concat([fm["paths"], fm["statics"]])
seeds_m = (123, 7, 42)
base_m = {int(r.seed): r.test_loss for r in pd.read_csv("results/mlp/base.csv").itertuples()}
spm = stops(fm, 0.01, base_m)

fig, ax = plt.subplots(figsize=(6.3, 3.3))
ax.set_yscale("log", base=2)
ax.axhline(1.0, color=QUIET, lw=0.8, ls=(0, (4, 3)), zorder=1)
for slug in ("mag", "berman", "w", "ws"):
    m, lab, col, al, lw, ls = SERIES[slug]
    c = curve(Dm, m, "test_loss", seeds_m, base_m)
    ax.plot(100 * GRID_PCT, c, color=col, alpha=al, lw=lw, ls=ls, zorder=3 if slug == "ws" else 2)
    f, yv = float(spm.loc[m, "frac_removed"]), float(spm.loc[m, "ratio"])
    ax.plot(100 * f, yv, "o", ms=6.5, mfc="white", mec=col, mew=1.6, alpha=max(al, 0.8), zorder=4)
ax.set_ylim(0.78, 40)
ax.set_xlim(0, 100)
ax.set_yticks([1, 2, 4, 8, 16, 32])
ax.set_yticklabels([r"$1\times$", r"$2\times$", r"$4\times$", r"$8\times$", r"$16\times$", r"$32\times$"])
ax.set_xticks(range(0, 101, 20))
ax.set_xticklabels([f"{p}%" for p in range(0, 101, 20)])
ax.set_xlabel(r"parameters removed ($k_{\mathrm{eff}}$)")
ax.set_ylabel("test loss / unpruned test loss")
ax.text(52, 31, "magnitude", color=QUIET, ha="center", fontsize=8.5)
ax.text(68, 14.5, "Berman et al. (2026)", color=QUIET, ha="center", fontsize=8.5)
ax.text(86, 4.0, "single parameters,\nsame rule", color=QUIET, ha="center", fontsize=8.5, linespacing=1.0)
ax.text(52, 1.12, "composite moves (W+S)", color=INK, ha="center", fontsize=9, fontname="cmb10")
ax.text(97.5, 1.38, f"stops at {100 * spm.loc[SERIES['ws'][0], 'frac_removed']:.0f}%", color=INK, ha="right", fontsize=8.5)
lo = min(spm.loc[SERIES[s][0], "frac_removed"] for s in ("w", "berman", "mag"))
hi = max(spm.loc[SERIES[s][0], "frac_removed"] for s in ("w", "berman", "mag"))
ax.annotate(f"the other three stop at {100 * lo:.0f} to {100 * hi:.0f}%", xy=(100 * hi, 0.96),
            xytext=(14, 0.855), color=QUIET, fontsize=8.5, va="center",
            arrowprops=dict(arrowstyle="-", color=QUIET, lw=0.6))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mlp_loss.pdf"))
plt.close(fig)

print("MLP stop points (margin 1%):", spm.loc[[v[0] for v in SERIES.values()]].round(3).to_dict("index"))
print("MLP test-loss ratio at matched sparsity (interpolated, mean of seeds)")
for slug in ("ws", "s", "w", "berman", "mag"):
    m = SERIES[slug][0]
    c = curve(Dm, m, "test_loss", seeds_m, base_m)
    print(f"  {m:24s}", [round(at(c, f), 3) for f in (0.1, 0.3, 0.5, 0.7, 0.9)])

# --------------------------------------------------------------------------- transformer
ft = E.run_fair("transformer")
Dt = pd.concat([ft["paths"], ft["statics"]])
spt = stops(ft, 1.0)
fig, ax = plt.subplots(figsize=(6.3, 3.3))
for slug in ("mag", "berman", "w", "s", "ws"):
    m, lab, col, al, lw, ls = SERIES[slug]
    c = curve(Dt, m, "test_acc", (0,))
    ax.plot(100 * GRID_PCT, 100 * c, color=col, alpha=al, lw=lw, ls=ls, label=lab,
            zorder=3 if slug == "ws" else 2)
    f, yv = float(spt.loc[m, "frac_removed"]), float(spt.loc[m, "test_acc"])
    ax.plot(100 * f, 100 * yv, "o", ms=6.5, mfc="white", mec=col, mew=1.6,
            alpha=max(al, 0.8), zorder=4)
cw = curve(Dt, SERIES["ws"][0], "test_acc", (0,))
cb = curve(Dt, SERIES["berman"][0], "test_acc", (0,))
ax.axvline(70, color=QUIET, lw=0.8, ls=(0, (3, 3)), zorder=1)
ax.plot([70], [100 * at(cw, 0.7)], "o", ms=4, color=ACCENT, zorder=5)
ax.plot([70], [100 * at(cb, 0.7)], "o", ms=4, color=MUTED, zorder=5)
ax.text(69, 104.5, f"at 70% removed: composite {at(cw, 0.7):.2f}, Berman et al. {at(cb, 0.7):.2f}",
        ha="right", fontsize=8.5, color=INK)
ax.set_ylim(-3, 109)
ax.set_xlim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels([f"{v}%" for v in (0, 25, 50, 75, 100)])
ax.set_xticks(range(0, 101, 20))
ax.set_xticklabels([f"{p}%" for p in range(0, 101, 20)])
ax.set_xlabel(r"parameters removed ($k_{\mathrm{eff}}$)")
ax.set_ylabel("test accuracy")
h, l = ax.get_legend_handles_labels()
order = [l.index(SERIES[s][1]) for s in ("ws", "s", "w", "berman", "mag")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower left",
          bbox_to_anchor=(0.01, 0.06), fontsize=8.3, handlelength=2.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tf_accuracy.pdf"))
plt.close(fig)

print("TF stop points (margin = base loss):", spt.loc[[v[0] for v in SERIES.values()]].round(3).to_dict("index"))
print("TF test accuracy at matched sparsity (interpolated)")
for m in ("W+S | valA | fdist", "W+S | train | fdist", "S | valA", "S | train", "W | valA | fdist",
          "W | train | fdist", "W | valA | magnitude", "fdist_iterative", "fdist_global", "fdist_one_shot",
          "magnitude", "mean_ablation"):
    c = curve(Dt, m, "test_acc", (0,))
    print(f"  {m:24s}", [round(at(c, f), 3) if f <= GRID_PCT[np.isfinite(c)].max() + 1e-9 else None
                          for f in (0.1, 0.3, 0.5, 0.7, 0.9)])

# --------------------------------------------------------------------------- spectra
cache = "results/transformer/head0_spectra.npz"
if not os.path.exists(cache):
    data = T.make_modadd_data(p=59, train_frac=0.5, seed=0)
    ad = T.TransformerAdapter(data, p=59)
    s0 = torch.load("results/transformer/base_state_seed0.pt", weights_only=False)
    X = torch.cat([data[k][0] for k in ("train", "val", "test")])
    bl = S.head_blocks(ad, s0, 0)
    ev = {blk: np.sort(np.linalg.eigvalsh(S.block_fisher(ad, s0, bl[blk], X, R=2)))[::-1]
          for blk in ("QK", "OV")}
    np.savez(cache, **ev)
ev = dict(np.load(cache))
floor = max(S.noise_floor_of(ev[b]) for b in ev)
lgf = np.log10(floor)
fig, ax = plt.subplots(figsize=(6.3, 3.5))
ax.axhspan(-18.6, lgf, color=MUTED, alpha=0.13, lw=0, zorder=0)
ax.axhline(lgf, color=MUTED, lw=0.8, ls=(0, (5, 4)), zorder=1)
info = {}
for blk, col, lw, lab in (("OV", MUTED, 1.6, r"value-output block ($W_V, W_O$)"),
                          ("QK", ACCENT, 2.2, r"query-key block ($W_Q, W_K$)")):
    mu = ev[blk] / ev[blk][0]
    lg = np.log10(np.clip(mu, 1e-18, None))
    i = np.arange(1, len(lg) + 1)
    ax.plot(i, lg, color=col, lw=lw, label=lab, zorder=3 if blk == "QK" else 2)
    v = S.classify_spectrum(ev[blk], noise_floor="auto")
    info[blk] = (v, lg)
vq, lq = info["QK"]
live = len(lq) - vq.null_dim
ax.plot([live, live + 1], [lq[live - 1], lq[live]], "o", ms=4, color=ACCENT, zorder=5)
ax.text(live - 8, lq[live - 1] - 1.1, f"{live} live directions, then a cliff", ha="right",
        fontsize=9, fontname="cmb10", color=INK)
ax.text(live - 8, lq[live - 1] - 2.1, f"{vq.plateau_decades:.1f}-decade gap down to the noise floor",
        ha="right", fontsize=8.5, color=QUIET)
ax.text(live - 8, lq[live - 1] - 3.1, rf"{vq.null_dim} nulls, as predicted by $2\,d\,d_h - r_K$",
        ha="right", fontsize=8.5, color=QUIET)
ax.text(170, -6.6, "a slope, no gap: no count", fontsize=9, fontname="cmb10", color=QUIET)
ax.text(1.15, -17.75, "below the noise floor: numerically zero", fontsize=8.5, color=QUIET)
ax.set_xscale("log")
ax.set_xlim(1, 2048)
ax.set_ylim(-18.6, 0.6)
ax.set_xticks([1, 10, 100, 1000, 2048])
ax.set_xticklabels(["1", "10", "100", "1000", "2048"])
ax.minorticks_off()
ax.set_yticks([0, -3, -6, -9, -12, -15, -18])
ax.set_yticklabels(["$1$", "$10^{-3}$", "$10^{-6}$", "$10^{-9}$", "$10^{-12}$", "$10^{-15}$",
                    r"$\leq 10^{-18}$"])
ax.set_xlabel("eigen-direction, ranked (log scale)")
ax.set_ylabel(r"eigenvalue / largest eigenvalue")
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "head_spectra.pdf"))
plt.close(fig)
print("head 0:", {b: (info[b][0].regime, info[b][0].null_dim, round(info[b][0].plateau_decades, 2),
                      int(np.sum(ev[b] <= floor * ev[b].max()))) for b in info}, "floor", floor)
