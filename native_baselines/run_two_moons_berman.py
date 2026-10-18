#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))

from nnprune.core import PruneConfig  # noqa: E402
from nnprune.experiments import checkpoint_sha256, load_benchmark, split_validation  # noqa: E402
from nnprune.fdist_baseline import fdist_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, default=.02)
    parser.add_argument("--maximum", type=float, default=.98)
    parser.add_argument("--fisher-examples", type=int, default=500)
    parser.add_argument("--validation-split-seed", type=int, default=0)
    parser.add_argument("--max-loss-error", type=float, default=.05)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "NNpruning/results/moons_berman.json")
    args = parser.parse_args()
    if not 0 < args.spacing <= args.maximum < 1:
        parser.error("require 0 < spacing <= maximum < 1")

    torch.set_num_threads(args.threads)
    adapter, state = load_benchmark("moons", args.device)
    adapter.data = split_validation(adapter.data, args.validation_split_seed)
    cfg = PruneConfig(n_sigma=2, tol_frac=args.max_loss_error, select_split="valA",
                      stop_split="valB", verbose=False)
    fractions = np.arange(0, args.maximum + args.spacing / 2, args.spacing)
    path = fdist_path(adapter, state, "fdist_iterative", cfg, fractions,
                      subset_size=args.fisher_examples, subset_seed=42,
                      verbose=True, active_total=True)
    flags = path["within_tolerance_loss"].to_numpy()
    failed = np.flatnonzero(~flags)
    stop_index = int(failed[0] - 1) if len(failed) else len(path) - 1
    stop_reference = float(adapter.per_sample(state, "valB").mean())
    artifact = {
        "result_schema": 3, "status": "complete", "benchmark": "moons",
        "method": "Berman et al. iterative", "checkpoint_sha256": checkpoint_sha256("moons"),
        "validation_split_seed": args.validation_split_seed, "device": args.device,
        "stop_index": stop_index,
        "protocol": {"select_split": "valA", "stop_split": "valB",
                     "criterion": "loss", "n_sigma": 2.0, "tol_abs": 0.0,
                     "tol_frac": args.max_loss_error, "stop_reference": stop_reference,
                     "fisher_examples": min(args.fisher_examples,
                                            len(adapter.data["train"][0])),
                     "fisher_subset_seed": 42, "spacing": args.spacing},
        "history": json.loads(path.to_json(orient="records")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
