"""
Worked example of Veronica's boundary pruning on the moons MLP (seed 123).

Uses her package (../NNpruning/nnprune) and her trained network
(../NNpruning/results/mlp/base_state_seed123.pt); nothing is retrained.

Sections
  1. the network and its data splits
  2. the cheap move: merging two near-duplicate units vs zeroing one of them
  3. replacing a nearly constant unit by its mean (and saturating it instead)
  4. one decision step of the greedy loop, candidate by candidate
  5. the whole path, and what is left of the network at the stop point
  6. figure: decision boundary before / after, and loss along the path

Run from anywhere:  python mlp_walkthrough.py
Outputs go next to this file: walkthrough_output.txt, walkthrough.png
"""
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NNP = os.path.join(HERE, "..", "NNpruning")
sys.path.insert(0, NNP)

from nnprune import mlp as M                                     # noqa: E402
from nnprune.core import (PruneConfig, paired_stats, zero_entries,  # noqa: E402
                          refresh, candidates, evaluate_candidates,
                          select_move, prune_path, REMOVED, STEP)
from nnprune.experiments import MLP_BASE, FAIR_COMMON, split_validation  # noqa: E402

torch.set_num_threads(4)
SEED = 123
out_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


def delta(ad, s_new, s_ref, split="val"):
    """Paired change of per-sample log loss, new minus reference: (mean, SE)."""
    return paired_stats(ad.per_sample(s_new, split, "loss"), ad.per_sample(s_ref, split, "loss"))


# ----------------------------------------------------------------------------
# 1. network and data
# ----------------------------------------------------------------------------
data, _ = M.make_moons_data(seed=SEED)
ad = M.MLPAdapter(split_validation(data, SEED))
s0 = torch.load(os.path.join(NNP, "results", "mlp", f"base_state_seed{SEED}.pt"), weights_only=False)

say("=" * 78)
say("1. THE NETWORK")
say("=" * 78)
say(f"architecture 2-50-50-50-1, sigmoid units; live parameters k_eff = {ad.k_eff(s0)}")
for sp in ("train", "valA", "valB", "test"):
    m = ad.metrics(s0, sp)
    say(f"  {sp:5s}: n = {ad.data[sp][1].shape[0]:4d}   loss {m['loss']:.4f}   accuracy {m['acc']:.3f}")

X, _ = ad.data["train"]
with torch.no_grad():
    _, acts = ad.logits(ad.effective(s0), s0.struct, X, return_acts=True)

# ----------------------------------------------------------------------------
# 2. duplicate units: merge vs zero
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("2. NEAR-DUPLICATE UNITS: MERGE vs ZERO")
say("=" * 78)
for l, a in enumerate(acts):
    C = torch.corrcoef(a.T)
    C.fill_diagonal_(0)
    say(f"  hidden layer h{l}: pairs with |corr| > 0.999: {int((C.abs() > 0.999).sum()) // 2:4d}"
        f"   > 0.99: {int((C.abs() > 0.99).sum()) // 2:4d}")

l = 1
a = acts[l]
C = torch.corrcoef(a.T)
C.fill_diagonal_(0)
flat = int(C.abs().argmax())
i, i2 = divmod(flat, C.shape[0])
say(f"\nmost correlated pair in h{l}: units {i} and {i2}, corr = {float(C[i, i2]):.5f}")
xi, xj = a[:, i] - a[:, i].mean(), a[:, i2] - a[:, i2].mean()
alpha = float((xi * xj).mean() / xj.pow(2).mean())
beta = float(a[:, i].mean() - alpha * a[:, i2].mean())
say(f"least-squares fit  a_{i}(x) ~ {alpha:.4f} * a_{i2}(x) + {beta:+.4f}")
for k in range(3):
    say(f"  training point {k}: a_{i} = {float(a[k, i]):.4f}   "
        f"alpha*a_{i2}+beta = {alpha * float(a[k, i2]) + beta:.4f}")

s_zero = s0.clone()
col = torch.zeros_like(s_zero.masks[f"W{l + 1}"])
col[:, i] = True
zero_entries(s_zero, f"W{l + 1}", col)
ad.canonicalize(s_zero)
s_merge = ad.apply(s0, M.Move("unit_merge", f"h{l}", l,
                              {"l": l, "i": i, "i2": i2, "alpha": alpha, "beta": beta}))

say(f"\n(a) zero unit {i}'s outgoing weights (classical pruning; the unit then dies):")
mz, sz = delta(ad, s_zero, s0)
say(f"    parameters removed {ad.k_eff(s0) - ad.k_eff(s_zero)},  val loss change {mz:+.5f} (SE {sz:.5f})")
say(f"(b) merge unit {i} into unit {i2}: W[:, {i2}] += alpha*W[:, {i}],  bias += beta*W[:, {i}]:")
mm, sm = delta(ad, s_merge, s0)
say(f"    parameters removed {ad.k_eff(s0) - ad.k_eff(s_merge)},  val loss change {mm:+.6f} (SE {sm:.6f})")

# ----------------------------------------------------------------------------
# 3. nearly constant unit: mean vs saturation vs zero
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("3. A NEARLY CONSTANT UNIT: MEAN, SATURATION, ZERO")
say("=" * 78)
best = None
for l_, a_ in enumerate(acts):
    sd = a_.std(0)                              # the unit whose output varies least over the data
    j = int(sd.argmin())
    if best is None or float(sd[j]) < best[0]:
        best = (float(sd[j]), l_, j)
_, l, j = best
aj = acts[l][:, j]
say(f"unit {j} of h{l}: activation mean {float(aj.mean()):.4f}, std {float(aj.std()):.5f}, "
    f"range [{float(aj.min()):.4f}, {float(aj.max()):.4f}]")
for name, kind, c in (("replace by its mean (unit_mean)", "unit_mean", float(aj.mean())),
                      ("saturate to 0  (b -> -inf)    ", "unit_sat", 0.0),
                      ("saturate to 1  (b -> +inf)    ", "unit_sat", 1.0)):
    s_m = ad.apply(s0, M.Move(kind, f"h{l}", l, {"l": l, "i": j, "c": c}))
    mu, se = delta(ad, s_m, s0)
    say(f"  {name}: removes {ad.k_eff(s0) - ad.k_eff(s_m)} params, val loss change {mu:+.5f} (SE {se:.5f})")
s_z = s0.clone()
col = torch.zeros_like(s_z.masks[f"W{l + 1}"])
col[:, j] = True
zero_entries(s_z, f"W{l + 1}", col)
ad.canonicalize(s_z)
mu, se = delta(ad, s_z, s0)
say(f"  zero its outgoing weights       : removes {ad.k_eff(s0) - ad.k_eff(s_z)} params, "
    f"val loss change {mu:+.5f} (SE {se:.5f})")

# ----------------------------------------------------------------------------
# 4. one decision step
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("4. ONE STEP OF THE GREEDY LOOP (round-4 setting: choose on valA, stop on valB)")
say("=" * 78)
base_val = ad.metrics(s0, "val")["loss"]
delta_margin = 0.01 * base_val
cfg = PruneConfig(**{**MLP_BASE, **FAIR_COMMON, "select_split": "valA", "weight_score": "obd",
                     "seed": SEED, "verbose": False})
ctx = refresh(ad, s0, cfg)
moves = candidates(ad, s0, ctx, cfg)
L0 = ad.per_sample(s0, "valA", "loss")
res = evaluate_candidates(ad, s0, moves, L0, cfg)
chosen, n_adm, n_ties = select_move(res, cfg)
best_r = min(res, key=lambda r: (r["mean_crit"], r["mean_loss"]))
for r in res:
    m_, se_ = paired_stats(r["crit"], best_r["crit"])
    r["tied"] = (m_ <= cfg.n_sigma * se_) or (r is best_r)
say(f"{len(moves)} candidates proposed: {Counter(m.kind for m in moves)}")
say("each is applied to a copy of the network and scored on valA (n=250), paired against the original")
say(f"best candidate = lowest valA loss; tie set = candidates within 2 paired SE of it: {n_ties} of {len(res)}")
say("tie-break: the tied candidate that removes the most parameters\n")
say(f"{'move':48s} {'removes':>7s} {'dLoss vs original':>18s} {'SE':>8s}  tied  chosen")
rows = sorted(res, key=lambda r: (-r["dk"], r["anch_mean"]))
show = rows[:12] + [r for r in rows[12:] if r["move"].kind == "zero"][:3] \
    + sorted([r for r in rows if not r["tied"]], key=lambda r: r["anch_mean"])[-3:]
seen = set()
for r in show:
    if id(r) in seen:
        continue
    seen.add(id(r))
    desc = r["move"].describe()
    if r["move"].kind == "unit_merge":
        p = r["move"].payload
        desc = f"unit_merge[h{p['l']}] {p['i']}->{p['i2']} (corr {p['corr']:.4f})"
    elif r["move"].kind in ("unit_mean", "unit_sat"):
        p = r["move"].payload
        desc = f"{r['move'].kind}[h{p['l']}] unit {p['i']} -> {p['c']:.3f}"
    say(f"{desc[:48]:48s} {r['dk']:7d} {r['anch_mean']:+18.5f} {r['anch_se']:8.5f}  "
        f"{'yes' if r['tied'] else 'no ':4s}  {'<==' if r is chosen else ''}")
sm_, sse_ = paired_stats(ad.per_sample(chosen["state"], "valB", "loss"), ad.per_sample(s0, "valB", "loss"))
say(f"\nstop test on valB for the chosen move: dLoss {sm_:+.5f} <= 2*SE + margin = "
    f"{2 * sse_ + delta_margin:.5f}?  {'pass -> apply it, next step' if sm_ <= 2 * sse_ + delta_margin else 'FAIL -> stop'}")

# ----------------------------------------------------------------------------
# 5. the whole path
# ----------------------------------------------------------------------------
say("")
say("=" * 78)
say("5. THE WHOLE PATH (stop margin = 1% of the base validation loss)")
say("=" * 78)
cfg_full = PruneConfig(**{**MLP_BASE, **FAIR_COMMON, "select_split": "valA", "weight_score": "obd",
                          "seed": SEED, "verbose": False, "stop_mode": "mark", "max_frac": 0.97})
df, _ = prune_path(ad, s0, cfg_full, label="W+S | valA | obd")
ok = (df["stop_mean"] <= 2 * df["stop_se"] + delta_margin).values
ff = int(np.argmax(~ok)) if (~ok).any() else len(df)
stop_step = int(df.iloc[ff - 1]["step"])
cfg_stop = PruneConfig(**{**MLP_BASE, **FAIR_COMMON, "select_split": "valA", "weight_score": "obd",
                          "seed": SEED, "verbose": False, "stop_mode": "halt",
                          "max_steps": stop_step, "max_frac": 1.0})
df_s, s_end = prune_path(ad, s0, cfg_stop, label="replay")
censored = ff == len(df)
say(f"path continued to {100 * df['frac_removed'].iloc[-1]:.1f}% removed; "
    + (f"the valB stop rule never failed, so the path ends at the 97% cap (step {stop_step})"
       if censored else f"the valB stop rule halts it after step {stop_step}"))
say(f"(cached result for this method and seed: results/fair_mlp/summary.csv)")
say(f"\n{'step':>4s} {'move':12s} {'removes':>7s} {'% removed':>9s} {'test loss':>9s} {'test acc':>8s}")
for _, r in df_s.iterrows():
    if r["step"] in (0, 1, 2, 3, 5, 10, 20, 30, 50, 75) or r["step"] == stop_step:
        say(f"{int(r['step']):4d} {str(r['kind']):12s} {int(r['dk']):7d} {100 * r['frac_removed']:9.1f} "
            f"{r['test_loss']:9.4f} {r['test_acc']:8.3f}")
say("\nwhere the removed parameters came from (dk summed by move kind):")
by = df_s.iloc[1:].groupby("kind")["dk"].agg(["count", "sum"]).sort_values("sum", ascending=False)
for k, r in by.iterrows():
    say(f"  {k:11s} {int(r['count']):4d} moves  {int(r['sum']):5d} parameters "
        f"({100 * r['sum'] / ad.k_eff(s0):4.1f}% of the original)")

say("\nwhat is left:")
widths = []
for l in range(ad.L - 1):
    t = s_end.struct[f"type{l}"]
    alive = int((t != REMOVED).sum())
    steps = int((t == STEP).sum())
    widths.append(alive)
    say(f"  hidden layer h{l}: {alive:2d} of 50 units alive"
        + (f" ({steps} of them Heaviside steps)" if steps else ""))
for k in s_end.params:
    say(f"  {k}: {int(s_end.masks[k].sum()):4d} live of {s_end.masks[k].numel()}")
say(f"  -> a 2-{'-'.join(map(str, widths))}-1 network with k_eff = {ad.k_eff(s_end)} "
    f"(was {ad.k_eff(s0)})")
for sp in ("train", "valB", "test"):
    m0_, m1_ = ad.metrics(s0, sp), ad.metrics(s_end, sp)
    say(f"  {sp:5s}: loss {m0_['loss']:.4f} -> {m1_['loss']:.4f}   accuracy {m0_['acc']:.3f} -> {m1_['acc']:.3f}")

# ----------------------------------------------------------------------------
# 6. figure
# ----------------------------------------------------------------------------
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

Xte, yte = ad.data["test"]
g = torch.linspace(-2.6, 2.6, 300)
GX, GY = torch.meshgrid(g, g, indexing="xy")
grid = torch.stack([GX.reshape(-1), GY.reshape(-1)], 1)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (s_, title) in zip(axs[:2], ((s0, f"original: {ad.k_eff(s0)} parameters"),
                                     (s_end, f"pruned: {ad.k_eff(s_end)} parameters"))):
    with torch.no_grad():
        p = torch.sigmoid(ad.logits(ad.effective(s_), s_.struct, grid)).reshape(GX.shape)
    ax.contourf(GX, GY, p, levels=20, cmap="RdBu_r", alpha=0.55)
    ax.contour(GX, GY, p, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="RdBu_r", s=6, edgecolors="none")
    acc = ad.metrics(s_, "test")["acc"]
    ax.set_title(f"{title}\ntest accuracy {acc:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
ax = axs[2]
ax.plot(100 * df["frac_removed"], df["test_loss"] / df["test_loss"].iloc[0], lw=1.5, label="test loss")
ax.axvline(100 * df_s["frac_removed"].iloc[-1], color="k", ls="--", lw=1, label="end of path: 97% cap, stop rule never failed" if censored else "stop (valB rule)")
ax.set_xlabel("% of parameters removed")
ax.set_ylabel("test loss / original test loss")
ax.set_title("loss along the greedy path")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "walkthrough.png"), dpi=130)

with open(os.path.join(HERE, "walkthrough_output.txt"), "w") as f:
    f.write("\n".join(out_lines) + "\n")
print("\nwrote walkthrough_output.txt and walkthrough.png")
