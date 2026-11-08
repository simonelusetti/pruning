from __future__ import annotations

import hydra
from pathlib import Path
import sys
from forge import start_run
from omegaconf import DictConfig

from pruning.runner import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.forge.on_finish and list(cfg.forge.on_finish[:2]) == ["python3", "pruning/plot_results.py"]:
        cfg.forge.on_finish = [sys.executable, str(Path(__file__).resolve().parent / "pruning/plot_results.py"),
                               *cfg.forge.on_finish[2:]]
    run = start_run(cfg)
    if run.status == "done":
        return 0
    run.finish(run_experiment(cfg, run))
    return 0


if __name__ == "__main__":
    main()
