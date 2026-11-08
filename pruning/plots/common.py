from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forge import ExperimentRun, ExperimentStore


@dataclass
class SavedCurve:
    run: ExperimentRun
    result: dict
    method: str

    @property
    def rows(self):
        return self.result.get("history", self.result.get("curve", []))

    @property
    def config(self):
        return self.run.experiment.config


def method(config):
    name, options = config.branch.name, config.branch.get("options", {})
    if name == "fisher_general":
        return "discrete"
    if name == "fisher_complement":
        if options.get("progressive") and options.get("mask_rule") == "sparsegpt":
            return "continuous"
        return None
    if name == "fisher_sparsegpt":
        return "sparsegpt" if options.get("geometry") == "axi" else None
    if name in {"continuous", "continuous_neuron"}:
        unit = options.get("deletion_unit", "weight")
        return {"weight": "continuous", "hidden_neuron": "continuous_neuron"}.get(unit)
    return name if name in {"classic", "discrete", "berman", "sparsegpt", "wanda", "flap"} else None


def load_curves(store: Path, benchmark: str, *, include_running=False):
    database = ExperimentStore(store)
    curves = []
    for artifact in sorted(database.xps_dir.glob("*/*/result.json")):
        try:
            run = database.load_run(*artifact.relative_to(database.xps_dir).parts[:2])
            if run.experiment.config.benchmark.name != benchmark:
                continue
            if run.status not in ({"done", "running"} if include_running else {"done"}):
                continue
            result = json.loads(artifact.read_text())
            kind = method(run.experiment.config)
            if kind and (result.get("history") or result.get("curve")):
                curves.append(SavedCurve(run, result, kind))
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Skipping unreadable Forge artifact {artifact}: {error}")
    return curves


def removed(row):
    return row.get("removed_percent", 100 * row.get("frac_removed", 0.))


def accepted(curve, index, row):
    if "accepted" in row:
        return row["accepted"]
    stop = curve.result.get("stop_index")
    if isinstance(stop, int) and stop >= 0:
        return index <= stop
    return row.get("admissible", row.get("admissible_loss", True))


def save_figure(figure, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=180)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    print(output)


def plot_supervised(curves, output, title):
    latest = {}
    for curve in curves:
        key = curve.run.experiment.signature
        if key not in latest or curve.run.launched_on > latest[key].run.launched_on:
            latest[key] = curve
    figure, axis = plt.subplots(figsize=(10, 6))
    count = 0
    margins = set()
    for curve in sorted(latest.values(), key=lambda item: (item.method, item.run.launched_on)):
        rows = [(index, row) for index, row in enumerate(curve.rows) if "test_loss" in row]
        if not rows or not rows[0][1]["test_loss"]:
            continue
        base = rows[0][1]["test_loss"]
        good = [row for index, row in rows if accepted(curve, index, row)]
        bad = next((row for index, row in rows if not accepted(curve, index, row)), None)
        if not good:
            continue
        name = curve.method
        if name == "classic":
            name = "Classic W+S" if curve.config.branch.options.get("weights", False) else "Classic S"
        label = f"{name} · {curve.run.experiment.signature}"
        line, = axis.plot([removed(row) for row in good], [row["test_loss"] / base for row in good],
                          marker="o", markersize=3, label=label)
        if bad is not None:
            axis.scatter(removed(bad), bad["test_loss"] / base, marker="x", color=line.get_color(), s=65)
        protocol = curve.result.get("protocol", {})
        rho = protocol.get("rho", protocol.get("tol_frac", curve.config.benchmark.get("rho", .05)))
        margins.add(rho)
        count += 1
    if not count:
        plt.close(figure)
        return False
    for rho in sorted(margins):
        axis.axhline(1 + rho, color="#ae7a22", linestyle="--", label=f"{100*rho:g}% base margin")
    axis.set(xlabel="Total parameters removed (%)", ylabel="Held-out loss / original", title=title)
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.text(.5, .015, "× First rejected proposal · stopping uses validation, not this held-out loss",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, 1))
    save_figure(figure, output)
    return True
