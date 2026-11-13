from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Refresh a benchmark's comparison plot from saved Forge results.")
    parser.add_argument("benchmark", nargs="?")
    parser.add_argument("--run", type=Path, help="Completed Forge run directory; defaults to FORGE_RUN_PATH")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-running", action="store_true")
    args = parser.parse_args()
    if os.environ.get("FORGE_STATUS") == "failed":
        return
    trigger = args.run or (Path(os.environ["FORGE_RUN_PATH"]) if "FORGE_RUN_PATH" in os.environ else None)
    benchmark = args.benchmark
    if trigger is not None:
        trigger = trigger.resolve()
        if json.loads((trigger / "result.json").read_text()).get("status") != "complete" and not args.include_running:
            print(f"Plot unchanged: {trigger} is not complete")
            return
        benchmark = OmegaConf.load(trigger.parent / "config.yaml").benchmark.name
        args.store = args.store or trigger.parents[2]
    if benchmark is None:
        parser.error("supply a benchmark name or a Forge run")
    store = (args.store or ROOT / "outputs").resolve()
    output = (args.output or store / "plots" / benchmark / "comparison.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    from pruning.plots import load_runs, plot
    # Lock before loading so concurrent completion hooks cannot publish an older snapshot last.
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not plot(load_runs(store, benchmark, include_running=args.include_running), output, benchmark):
            print(f"Plot unchanged: no saved {benchmark} runs with a pruning curve")


if __name__ == "__main__":
    main()
