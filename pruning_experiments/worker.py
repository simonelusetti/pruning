from __future__ import annotations

import importlib
import sys


def main() -> None:
    module_name, *arguments = sys.argv[1:]
    if module_name not in {
            "moons_berman", "berman", "berman_matched", "modadd",
            "modadd_sparsegpt", "opt_classic", "opt_cpu", "opt_fisher",
            "opt_matched", "opt_b_sweep", "compensation", "native",
            "opt_sparsegpt_curve"}:
        raise ValueError(f"unknown experiment implementation: {module_name}")
    module = importlib.import_module(f"pruning_experiments.{module_name}")
    module.main(arguments)


if __name__ == "__main__":
    main()
