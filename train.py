from __future__ import annotations

import hydra
from forge import start_run
from omegaconf import DictConfig

from pruning_experiments import execute


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    run = start_run(cfg)
    if run.status == "done":
        return 0
    run.finish(execute(cfg, run))
    return 0


if __name__ == "__main__":
    main()
