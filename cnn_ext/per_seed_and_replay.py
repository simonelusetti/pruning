"""Per-seed paired comparison of the CNN-aware vs naive moves, and a replay of one
pruned network to the stop point (seed 0, W+S-cnn | valA | obd, rho = 10%) to show
what is left.  Appends to results/summary.txt."""
import os, sys
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cnn import CNNAdapter, make_digits_data, CNN_KINDS
from nnprune.core import PruneConfig, prune_path, REMOVED
from nnprune.experiments import FAIR_COMMON, stop_points, split_validation
from run_cnn import BASE, VARIANTS
R = os.environ.get("CNN_RESULTS", os.path.join(HERE, "results"))
P = pd.read_csv(os.path.join(R, "paths.csv")); S = pd.read_csv(os.path.join(R, "statics.csv"))
sp = stop_points({"paths": P, "statics": S}, rel_tols=(0.01, 0.1, 1.0))
out = ["", "4. PER SEED: % removed at the stop, CNN-aware vs naive moves"]
for rho in (0.01, 0.1, 1.0):
    x = sp[np.isclose(sp.rel_tol, rho)].pivot(index="seed", columns="method", values="frac_removed") * 100
    for a, b in (("S-cnn | valA", "S-naive | valA"), ("W+S-cnn | valA | obd", "W+S-naive | valA | obd")):
        out.append(f"  rho={rho:<5g} {a:22s} vs {b:24s} " + "  ".join(f"seed {s}: {x.loc[s, a]:5.1f} vs {x.loc[s, b]:5.1f}" for s in x.index))
# replay
seed, name, rho = 0, "W+S-cnn | valA | obd", 0.1
row = sp[(sp.seed == seed) & (sp.method == name) & np.isclose(sp.rel_tol, rho)].iloc[0]
d = P[(P.seed == seed) & (P.method == name)]
step = int(d[d.k_eff == row.k_eff].step.iloc[0])
ad = CNNAdapter(split_validation(make_digits_data(seed), seed))
s0 = torch.load(os.path.join(R, f"base_state_seed{seed}.pt"), weights_only=False)
cfg = PruneConfig(**{**BASE, **FAIR_COMMON, **VARIANTS[name], "seed": seed, "verbose": False,
                     "stop_mode": "mark", "post_stop_steps": 10**9, "max_steps": step})
df, s = prune_path(ad, s0, cfg, label=name)
out.append("")
out.append(f"5. REPLAY: seed {seed}, {name}, stop at rho={rho}: step {step}, k_eff {ad.k_eff(s)} (csv {int(row.k_eff)}) of {ad.k_eff(s0)}")
out.append(f"  conv1 channels alive {int((s.struct['type0'] != REMOVED).sum())}/16, conv2 channels alive {int((s.struct['type1'] != REMOVED).sum())}/32, "
           f"region bias B2 live: {bool(s.masks['B2'].any())}")
for k in s.params:
    out.append(f"  {k}: {int(s.masks[k].sum())} live of {s.masks[k].numel()}")
for sp_ in ("valB", "test"):
    a, b = ad.metrics(s0, sp_), ad.metrics(s, sp_)
    out.append(f"  {sp_}: loss {a['loss']:.4f} -> {b['loss']:.4f}, acc {a['acc']:.3f} -> {b['acc']:.3f}")
print("\n".join(out))
with open(os.path.join(R, "summary.txt"), "a") as fh:
    fh.write("\n".join(out) + "\n")
