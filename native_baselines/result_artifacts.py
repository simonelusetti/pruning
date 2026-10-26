from __future__ import annotations

import json
import os
from pathlib import Path


def update_plot(group: str):
    if group == "opt":
        from plot_opt_published import main
    elif group in {"two_moons", "berman", "modadd"}:
        from plot_comparisons import main
    else:
        raise ValueError(f"unknown plot group: {group}")
    main()


def write_json(path: Path, value: dict, plot_group: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)
    if plot_group is not None:
        update_plot(plot_group)
