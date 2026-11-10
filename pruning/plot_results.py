from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

PLOTTERS = {"opt": "opt", "opt_cpu": "opt", "berman": "berman", "moons": "two_moons", "modadd": "modadd", "network": "network"}


def main():
    parser = argparse.ArgumentParser(description="Refresh a benchmark plot from saved Forge result.json artifacts.")
    parser.add_argument("benchmark", nargs="?", choices=(*PLOTTERS, "two_moons"))
    parser.add_argument("--run", type=Path, help="Completed Forge run directory; defaults to FORGE_RUN_PATH")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-running", action="store_true")
    parser.add_argument("--matched-opt-heldout", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("FORGE_STATUS") == "failed":
        return
    trigger = args.run or (Path(os.environ["FORGE_RUN_PATH"]) if "FORGE_RUN_PATH" in os.environ else None)
    benchmark = "moons" if args.benchmark == "two_moons" else args.benchmark
    if trigger is not None:
        trigger = trigger.resolve()
        result = json.loads((trigger / "result.json").read_text())
        config = OmegaConf.load(trigger.parent / "config.yaml")
        if result.get("status") != "complete" and not args.include_running:
            print(f"Plot unchanged: {trigger} is not complete")
            return
        if not (result.get("curve") or result.get("history")):
            print(f"Plot unchanged: {trigger} has no pruning curve artifact")
            return
        if benchmark is not None and benchmark != config.benchmark.name:
            parser.error("benchmark does not match --run")
        benchmark = config.benchmark.name
        if args.store is None:
            args.store = trigger.parents[2]
    if benchmark not in PLOTTERS:
        parser.error("supply a supported benchmark or a Forge run with a pruning curve")
    store = (args.store or ROOT / "outputs").resolve()
    folder = PLOTTERS[benchmark] if benchmark != "opt_cpu" else "opt_cpu"
    output = args.output or store / "plots" / folder / "comparison.png"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from pruning.plots.common import load_curves
    plotter = importlib.import_module(f"pruning.plots.{PLOTTERS[benchmark]}")
    # Lock before loading so concurrent completion hooks cannot publish an older snapshot last.
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        curves = load_curves(store, benchmark, include_running=args.include_running)
        if not plotter.plot(curves, output, trigger):
            print(f"Plot unchanged: no saved {benchmark} curves with the required loss measurements")


if __name__ == "__main__":
    main()
