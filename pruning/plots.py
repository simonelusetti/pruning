"""One four-panel comparison per benchmark, built only from saved Forge results."""
from __future__ import annotations

import json
import re
from pathlib import Path
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forge import ExperimentStore


# Labels recorded before the branches were renamed.
RENAMED = {"continuous": "singular", "discrete": "joint"}


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
            # Runs from before method labels were recorded predate the current methods and are not drawn.
            name = re.sub(r"^(continuous|discrete)", lambda match: RENAMED[match[1]], result.get("label") or "")
            if name and result.get("curve") and (name not in latest or run.launched_on > latest[name][0].launched_on):
                latest[name] = run, result
        except (OSError, ValueError, KeyError) as error:
            warnings.warn(f"Skipping unreadable Forge artifact {artifact}: {error}")
    return dict(sorted(latest.items()))


def model_removed(row, result):
    if "model_removed_percent" in row:
        return row["model_removed_percent"]
    total = result.get("total_model_parameters")
    return row["removed_percent"] * result["parameter_denominator"] / total if total else None


def amount(value, unit):
    scale = next(scale for scale in (1e9, 1e6, 1e3, 1, 1e-3) if value >= scale or scale == 1e-3)
    prefix = {1e9: "G", 1e6: "M", 1e3: "k", 1: "", 1e-3: "m"}[scale]
    return f"{value / scale:.3g} {prefix}{unit}"


# One validated categorical hue per branch, fixed across benchmarks. The assignment maximises the smallest colour
# difference between branches that share a benchmark; branches that never meet (singular and classic, joint
# and SliceGPT, SparseGPT and LLM-Pruner) may share a hue.
FAMILIES = {"singular": "#e87ba4", "joint_simple": "#6250d6", "joint": "#008300", "classic": "#e34948",
            "sparsegpt": "#2a78d6", "wanda": "#eb6834", "flap": "#1baf7a", "berman": "#eda100",
            "slicegpt": "#008300", "llm_pruner": "#2a78d6", "reference": "#898781"}
OURS = ("singular", "joint_simple", "joint", "classic")
DEFAULTS = {"singular": "global · both · affine · kfac", "joint_simple": "global · both · affine · kfac",
            "joint": "global · both · affine · fisher"}


def family(name):
    head = name.split(" · ")[0].removeprefix("native_")
    return next((key for key in FAMILIES if head == key or head.startswith(key + "_")), head)


def setting(name):
    """What distinguishes a method inside its branch."""
    if name.startswith("native_"):
        return "authors' code"
    rest = name.removeprefix(family(name)).lstrip(" ·_")
    if not rest:
        return "default" if family(name) in OURS else "re-implementation"
    return f"{rest} (default)" if DEFAULTS.get(family(name)) == rest else rest


def oklab(rgb, inverse=False):
    """sRGB (0-1) to OKLab, or back with inverse=True (Ottosson's matrices)."""
    import numpy as np
    m1 = np.array([[.4122214708, .5363325363, .0514459929], [.2119034982, .6806995451, .1073969566],
                   [.0883024619, .2817188376, .6299787005]])
    m2 = np.array([[.2104542553, .7936177850, -.0040720468], [1.9779984951, -2.4285922050, .4505937099],
                   [.0259040371, .7827717662, -.8086757660]])
    if inverse:
        lms = np.linalg.solve(m2, rgb) ** 3
        linear = np.linalg.solve(m1, lms)
        return np.clip(np.where(linear > .0031308, 1.055 * np.abs(linear) ** (1 / 2.4) - .055, 12.92 * linear), 0, 1)
    linear = np.where(rgb > .04045, ((rgb + .055) / 1.055) ** 2.4, rgb / 12.92)
    return m2 @ np.cbrt(m1 @ linear)


def shades(base, count):
    """One method keeps the base hue; several get evenly spaced OKLab lightness steps, darkest first."""
    import numpy as np
    lab = oklab(np.array([int(base[index:index + 2], 16) / 255 for index in (1, 3, 5)]))
    levels = [lab[0]] if count == 1 else np.linspace(.42 if count > 3 else .46, .80 if count > 3 else .76, count)
    return ["#" + "".join(f"{round(255 * value):02x}" for value in oklab(np.array([level, *lab[1:]]), inverse=True))
            for level in levels]


def ordered(names):
    """Branches in palette order; within a branch the default comes first."""
    return sorted(names, key=lambda name: (list(FAMILIES).index(family(name)) if family(name) in FAMILIES else len(FAMILIES),
                                           not setting(name).endswith(("default", "(default)", "re-implementation")), name))


def styles(names):
    """Branch hue for every method; settings of one branch are shades of it, the default darkest."""
    groups = {}
    for name in ordered(names):
        groups.setdefault(family(name), []).append(name)
    return {name: color for members in groups.values()
            for name, color in zip(members, shades(FAMILIES.get(family(members[0]), "#52514e"), len(members)))}


def plot(runs, output: Path, title: str):
    if not runs:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(17, 11))
    curves, bars = axes[0], axes[1]
    colors = styles(runs)
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
        for axis, x in zip(curves, (lambda row: row["removed_percent"], lambda row: model_removed(row, result))):
            if x(good[0]) is None:
                continue
            axis.plot([x(row) for row in good], [row["heldout_damage_percent"] for row in good], marker="o",
                      markersize=3.5, linewidth=2, color=colors[name], label=name)
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
        names = ordered(measured)
        positions = [index + .6 * len({family(other) for other in names[:index + 1]}) for index in range(len(names))]
        heights = [100 * measured[name]["endpoint"][key] / measured[name]["dense"][key] for name in names]
        drawn = axis.bar(positions, heights, color=[colors[name] for name in names])
        for bar, name in zip(drawn, names):
            axis.annotate(amount(measured[name]["endpoint"][key], unit), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha="center", va="bottom", fontsize=8)
        axis.axhline(100, color="#555555", linewidth=1)
        axis.set_xticks(positions, [f"{family(name)}: {setting(name)}" for name in names], rotation=30, ha="right", fontsize=8)
        axis.set(ylabel=ylabel, ylim=(0, max([110, *heights]) * 1.1), xlim=(-.6, max(positions, default=0) + 1))
        axis.grid(color="#e6e6e6", linewidth=.8, axis="y")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if not names:
            axis.text(.5, .5, "No run has inference measurements yet", ha="center", va="center", transform=axis.transAxes)
    # One legend column per branch: a bold header, then the settings that distinguish its curves.
    handles = dict(zip(*reversed(curves[0].get_legend_handles_labels())))
    columns = {}
    for name in ordered(handles):
        columns.setdefault(family(name), []).append(name)
    depth = max(map(len, columns.values()))
    blank = plt.Line2D([], [], linestyle="none")
    entries = []
    for branch, members in columns.items():
        kind = "ours" if branch in OURS else "control" if branch == "reference" else "baseline"
        entries += [(blank, f"{branch} ({kind})")] + [(handles[name], setting(name)) for name in members]
        entries += [(blank, "")] * (depth - len(members))
    legend = figure.legend(*zip(*entries), loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(columns), fontsize=8,
                           frameon=False, handlelength=1.6, columnspacing=1.8)
    for text in legend.get_texts()[::depth + 1]:
        text.set_fontweight("bold")
    figure.suptitle(f"{title} · pruning comparison", fontsize=14, fontweight="bold")
    figure.text(.5, .01, "Latest completed run per method · hue: branch, shade: its settings (darkest = default) · × first rejected step "
                "(▲ at the top edge when above the plotted range)\nBrown line: validation margin · inference at the last accepted state "
                "after dropping dead units, relative to the same run's dense measurement; labels give absolute values",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .955 - .019 * (depth + 1)))
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
