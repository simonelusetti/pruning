"""
Same experiment as run_cnn.py, split into small resumable units (one pruning arm, or
the one-shot baselines, for one seed), so it can run in short sessions.  Each call
runs units until the time budget is spent, saving each finished unit to
<results>/units/; finished units are skipped next time.  When every unit is done,
the units are merged into <results>/paths.csv and statics.csv (same format as
run_cnn.py, so analyze_cnn.py works unchanged).

Usage:  CNN_RESULTS=results_local python run_resumable.py [budget_seconds] [seeds...]
"""
import os
import sys
import time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn import CNNAdapter, make_digits_data, train_cnn  # noqa: E402
from nnprune.core import PruneConfig, baseline_path, prune_path, unit_baseline_path  # noqa: E402
from nnprune.experiments import FAIR_COMMON, split_validation  # noqa: E402
from run_cnn import BASE, VARIANTS  # noqa: E402

torch.set_num_threads(4)
OUT = os.environ.get("CNN_RESULTS", os.path.join(HERE, "results"))
UNITS = os.path.join(OUT, "units")


def unit_file(seed, name):
    return os.path.join(UNITS, f"seed{seed}__{name.replace(' | ', '__').replace(' ', '_')}.csv")


def setup(seed):
    data = make_digits_data(seed)
    ad = CNNAdapter(split_validation(data, seed))
    f = os.path.join(OUT, f"base_state_seed{seed}.pt")
    if os.path.exists(f):
        s0 = torch.load(f, weights_only=False)
    else:
        s0 = train_cnn(ad, seed=seed)
        torch.save(s0, f)
    return ad, s0


def run_unit(seed, name):
    ad, s0 = setup(seed)
    if name == "baselines":
        cfg = PruneConfig(**{**BASE, **FAIR_COMMON, "seed": seed, "verbose": False})
        ds = [baseline_path(ad, s0, b, cfg) for b in ("magnitude", "obd", "random")]
        ds.append(unit_baseline_path(ad, s0, cfg))
        df = pd.concat(ds, ignore_index=True)
    else:
        cfg = PruneConfig(**{**BASE, **FAIR_COMMON, **VARIANTS[name], "post_stop_steps": 300,
                             "seed": seed, "verbose": False})
        df, _ = prune_path(ad, s0, cfg, label=name)
    df["seed"] = seed
    df.to_csv(unit_file(seed, name), index=False)


if __name__ == "__main__":
    os.makedirs(UNITS, exist_ok=True)
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    seeds = [int(a) for a in sys.argv[2:]] or [0, 1, 2]
    todo = [(s, n) for s in seeds for n in list(VARIANTS) + ["baselines"]
            if not os.path.exists(unit_file(s, n))]
    t0 = time.time()
    for seed, name in todo:
        if time.time() - t0 > budget:
            break
        t = time.time()
        run_unit(seed, name)
        print(f"[seed {seed}] {name}: done in {time.time() - t:.0f}s", flush=True)
    left = [(s, n) for s in seeds for n in list(VARIANTS) + ["baselines"] if not os.path.exists(unit_file(s, n))]
    print(f"{len(left)} units left", flush=True)
    if not left:
        P = pd.concat([pd.read_csv(unit_file(s, n)) for s in seeds for n in VARIANTS], ignore_index=True)
        S = pd.concat([pd.read_csv(unit_file(s, "baselines")) for s in seeds], ignore_index=True)
        P.to_csv(os.path.join(OUT, "paths.csv"), index=False)
        S.to_csv(os.path.join(OUT, "statics.csv"), index=False)
        print("merged into paths.csv and statics.csv")
