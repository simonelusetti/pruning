"""One four-panel comparison per benchmark, built only from saved Forge results."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forge import ExperimentStore


def load_runs(store: Path, benchmark: str, *, include_running=False):
    """Latest run per method label on `benchmark`; each branch config is its own method."""
    database = ExperimentStore(store)
    latest = {}
    for artifact in sorted(database.xps_dir.glob("*/*/result.json")):
        try:
            run = database.load_run(*artifact.relative_to(database.xps_dir).parts[:2])
            config = run.experiment.config
            if (config.benchmark.name != benchmark or
                    run.status not in ({"done", "running"} if include_running else {"done"})):
                continue
            result = json.loads(artifact.read_text())
            name = config.branch.get("label", config.branch.name)
            if result.get("curve") and (name not in latest or run.launched_on > latest[name][0].launched_on):
                latest[name] = run, result
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Skipping unreadable Forge artifact {artifact}: {error}")
    # Runs from before branch labels carry only the base branch name; a labelled run of that branch supersedes them.
    labelled = {run.experiment.config.branch.name for run, _ in latest.values() if "label" in run.experiment.config.branch}
    return {name: value for name, value in sorted(latest.items())
            if "label" in value[0].experiment.config.branch or name not in labelled}


def model_removed(row, result):
    if "model_removed_percent" in row:
        return row["model_removed_percent"]
    total = result.get("total_model_parameters")
    return row["removed_percent"] * result["parameter_denominator"] / total if total else None


def amount(value, unit):
    scale = next(scale for scale in (1e9, 1e6, 1e3, 1, 1e-3) if value >= scale or scale == 1e-3)
    prefix = {1e9: "G", 1e6: "M", 1e3: "k", 1: "", 1e-3: "m"}[scale]
    return f"{value / scale:.3g} {prefix}{unit}"


# Validated categorical order (adjacent CVD separation); a ninth method reuses hues with a dotted line.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
OURS = ("classic", "discrete", "continuous", "canonical", "fisher")


def styles(names):
    """One hue per method in name order; our methods are solid circles, baselines dashed squares."""
    return {name: {"color": PALETTE[index % len(PALETTE)], "marker": "o" if name.startswith(OURS) else "s",
                   "linestyle": ("-" if name.startswith(OURS) else "--") if index < len(PALETTE) else ":"}
            for index, name in enumerate(names)}


def plot(runs, output: Path, title: str):
    if not runs:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(17, 11))
    curves, bars = axes[0], axes[1]
    style = styles(runs)
    colors = {name: value["color"] for name, value in style.items()}
    margins, series = set(), []
    for name, (run, result) in runs.items():
        rows = [row for row in result["curve"] if "heldout_damage_percent" in row]
        good = [row for row in rows if row.get("accepted", True)]
        rejected = next((row for row in rows if row.get("accepted") is False), None)
        if result.get("selection", {}).get("enabled"):
            margins.add(result["selection"]["rho"])
        series.append((name, run, result, good, rejected))
    # The margin sits near the top: the region below the breaking point gets most of the height.
    accepted_damage = [row["heldout_damage_percent"] for *_, good, _ in series for row in good]
    top = 130 * max(margins) if margins else 1.1 * max(accepted_damage)
    bottom = min(0., min(accepted_damage)) - .04 * (top - min(0., min(accepted_damage)))
    for name, run, result, good, rejected in series:
        label = f"{name} · {run.signature} · pool {result['parameter_denominator']:,}"
        for axis, x in zip(curves, (lambda row: row["removed_percent"], lambda row: model_removed(row, result))):
            if x(good[0]) is None:
                continue
            axis.plot([x(row) for row in good], [row["heldout_damage_percent"] for row in good],
                      markersize=4, linewidth=2, label=label, **style[name])
            if rejected is not None:
                damage = rejected["heldout_damage_percent"]
                axis.scatter(x(rejected), min(damage, top), marker="x" if damage <= top else "^", s=70,
                             linewidths=2, color=colors[name], zorder=5, clip_on=False)
    for index, (axis, xlabel) in enumerate(zip(curves, ("Considered parameters removed (%)",
                                                        "Total model parameters removed (%)"))):
        xs = [line.get_xdata() for line in axis.get_lines()] + [collection.get_offsets()[:, 0]
                                                                for collection in axis.collections]
        low = min([0., *[min(values) for values in xs if len(values)]])
        high = max([5., *[max(values) for values in xs if len(values)]])
        for rho in sorted(margins):
            axis.axhline(100 * rho, color="#8a6a2a", linestyle=(0, (4, 3)), linewidth=1.2)
        axis.set(xlabel=xlabel, ylabel="Held-out loss increase from original (%)",
                 xlim=(low - .02 * (high - low), high + .04 * (high - low)), ylim=(bottom, top))
        axis.grid(color="#e6e6e6", linewidth=.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    measured = {name: result["inference"] for name, (_, result) in runs.items() if "inference" in result}
    for axis, key, unit, ylabel in zip(bars, ("seconds", "parameter_bytes"), ("s", "B"),
                                       ("Inference time, % of dense", "Parameter memory, % of dense")):
        names = list(measured)
        heights = [100 * measured[name]["endpoint"][key] / measured[name]["dense"][key] for name in names]
        drawn = axis.bar(range(len(names)), heights, color=[colors[name] for name in names])
        for bar, name in zip(drawn, names):
            axis.annotate(amount(measured[name]["endpoint"][key], unit), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha="center", va="bottom", fontsize=8)
        axis.axhline(100, color="#555555", linewidth=1)
        axis.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        axis.set(ylabel=ylabel, ylim=(0, max([110, *heights]) * 1.1), xlim=(-.6, max(len(names), 6) - .4))
        axis.grid(color="#e6e6e6", linewidth=.8, axis="y")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if not names:
            axis.text(.5, .5, "No run has inference measurements yet", ha="center", va="center", transform=axis.transAxes)
    handles, labels = curves[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=2, fontsize=8, frameon=False)
    figure.suptitle(f"{title} · pruning comparison", fontsize=14, fontweight="bold")
    figure.text(.5, .01, "Latest completed run per method · solid: our methods, dashed: baselines · × first rejected step (▲ at the top edge when above the plotted range)\n"
                "Brown line: validation margin · inference at the last accepted state after dropping dead units, "
                "relative to the same run's dense measurement; labels give absolute values",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .955 - .018 * ((len(labels) + 1) // 2)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=output.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=160)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    print(output)
    return True
