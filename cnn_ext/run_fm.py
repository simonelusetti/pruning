"""
Her round-4 protocol on Fashion-MNIST, resumable, for short sessions.

Network: cnn_fm.py (conv 16 -> max-pool -> conv 32 -> GAP -> fc; 5 130 parameters),
trained 12 epochs on 50 000 images.  Stop test on valB = 8 000 held-out images
(vs about 180 on the 8x8 digits); moves chosen on valA = 2 000 images.

Arms (her prune_path logic via resumable.prune_path_resumable, FAIR_COMMON settings):
  W | valA | magnitude    single weights (chunks of 16) only
  S-naive | valA          her MLP moves ported as they are
  S-cnn | valA            + output-side constants and the spatial merge
  W+S-cnn | valA | obd    everything
One-shot baselines (her baseline_path / unit_baseline_path, same valB stop):
  magnitude, obd, random, mean_ablation.

Usage (repeat until it prints "0 units left"):
  CNN_RESULTS=results_fm python run_fm.py [budget_seconds] [seeds...]      default 150 s, seeds 0 1
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn_fm import CNN_KINDS, NAIVE_KINDS, CNNAdapterFM, load_fashion, train_epochs  # noqa: E402
from nnprune.core import PruneConfig, baseline_path, unit_baseline_path  # noqa: E402
from nnprune.experiments import FAIR_COMMON  # noqa: E402
from resumable import prune_path_resumable  # noqa: E402

torch.set_num_threads(4)
OUT = os.environ.get("CNN_RESULTS", os.path.join(HERE, "results_fm"))
UNITS = os.path.join(OUT, "units")
EPOCHS = 12
BASE = dict(weight_chunk=16, weight_per_site=3, n_struct=24, n_merge=8, refresh_every=10, screen_n=1000)
COMMON = {**FAIR_COMMON, "post_stop_steps": 50}
VARIANTS = {
    "W | valA | magnitude": dict(select_split="valA", weight_score="magnitude", use_struct=False),
    "S-naive | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=NAIVE_KINDS),
    "S-cnn | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=CNN_KINDS),
    "W+S-cnn | valA | obd": dict(select_split="valA", weight_score="obd", struct_kinds=CNN_KINDS),
}
STATICS = ("magnitude", "obd", "random", "mean_ablation")
FRACS = np.round(np.concatenate([np.arange(0, 0.5, 0.04), np.arange(0.5, 0.96, 0.05)]), 4)


def ufile(seed, name):
    return os.path.join(UNITS, f"seed{seed}__{name.replace(' | ', '__').replace(' ', '_')}.csv")


def get_data(seed):
    data, fit = load_fashion(seed, n_valA=2000, n_prune_train=2000)
    return CNNAdapterFM(data, stats_n=1000), fit


def base_state(seed, deadline):
    f = os.path.join(OUT, f"base_state_seed{seed}.pt")
    ad, fit = get_data(seed)
    if os.path.exists(f):
        return ad, torch.load(f, weights_only=False)
    s = train_epochs(ad, fit, os.path.join(OUT, f"train_ckpt_seed{seed}.pt"), seed=seed,
                     epochs=EPOCHS, lr=3e-3, deadline=deadline)
    if s is None:
        return ad, None
    torch.save(s, f)
    os.remove(os.path.join(OUT, f"train_ckpt_seed{seed}.pt"))
    m = {k: ad.metrics(s, k) for k in ("valA", "valB", "test")}
    print(f"[seed {seed}] trained: " + "  ".join(f"{k} loss {v['loss']:.4f} acc {v['acc']:.3f}" for k, v in m.items()), flush=True)
    return ad, s


def run_unit(seed, name, ad, s0, deadline):
    if name in STATICS:
        cfg = PruneConfig(**{**BASE, **COMMON, "seed": seed, "verbose": False})
        df = unit_baseline_path(ad, s0, cfg) if name == "mean_ablation" else baseline_path(ad, s0, name, cfg, fracs=FRACS)
        df["method"] = name
    else:
        cfg = PruneConfig(**{**BASE, **COMMON, **VARIANTS[name], "seed": seed, "verbose": False})
        df = prune_path_resumable(ad, s0, cfg, name, ufile(seed, name) + ".ckpt", deadline)
        if df is None:
            return False
    df["seed"] = seed
    df.to_csv(ufile(seed, name), index=False)
    return True


if __name__ == "__main__":
    os.makedirs(UNITS, exist_ok=True)
    t0 = time.time()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    seeds = [int(a) for a in sys.argv[2:]] or [0, 1]
    deadline = t0 + budget
    names = list(VARIANTS) + list(STATICS)
    for seed in seeds:
        todo = [n for n in names if not os.path.exists(ufile(seed, n))]
        if not todo:
            continue
        ad, s0 = base_state(seed, deadline)
        if s0 is None:
            print(f"[seed {seed}] training in progress", flush=True)
            break
        stop = False
        for name in todo:
            if time.time() > deadline or (name in STATICS and time.time() - t0 > 25):
                stop = True                      # one-shot baselines are not resumable: start them early in a call
                break
            t = time.time()
            done = run_unit(seed, name, ad, s0, deadline)
            print(f"[seed {seed}] {name}: {'done' if done else 'checkpointed'} ({time.time() - t:.0f}s)", flush=True)
            if not done:
                stop = True
                break
        if stop:
            break
    left = [(s, n) for s in seeds for n in names if not os.path.exists(ufile(s, n))]
    print(f"{len(left)} units left", flush=True)
    if not left:
        P = pd.concat([pd.read_csv(ufile(s, n)) for s in seeds for n in VARIANTS], ignore_index=True)
        S = pd.concat([pd.read_csv(ufile(s, n)) for s in seeds for n in STATICS], ignore_index=True)
        P.to_csv(os.path.join(OUT, "paths.csv"), index=False)
        S.to_csv(os.path.join(OUT, "statics.csv"), index=False)
        print("merged into paths.csv and statics.csv")
