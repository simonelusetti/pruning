from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from forge import ExperimentStore

ROOT = Path(__file__).resolve().parent


def curve(result: dict, benchmark: str):
    rows = result.get("history", result.get("curve", []))
    if not rows:
        return None
    x = [100 * row["frac_removed"] if "frac_removed" in row else
         row["removed_percent"] for row in rows]
    if benchmark in {"berman", "moons", "modadd"}:
        if "test_loss" not in rows[0]:
            return None
        base = rows[0]["test_loss"]
        y = [row["test_loss"] / base for row in rows]
        ylabel = "Held-out test loss / original"
    else:
        if "loss_damage_percent" not in rows[0]:
            return None
        y = [row["loss_damage_percent"] for row in rows]
        ylabel = "Selection loss increase (%)"
    stop = result.get("stop_index")
    if isinstance(stop, int) and stop >= 0:
        x, y = x[:stop + 1], y[:stop + 1]
    return x, y, ylabel


def opt_historical_overlay(store: Path, output: Path) -> None:
    legacy = ROOT / "plots" / "opt" / "comparison_legacy.png"
    image = plt.imread(legacy)
    height, width = image.shape[:2]
    if (width, height) != (2773, 1238):
        raise ValueError("historical OPT image dimensions changed; axis anchors need updating")
    candidates = {"module_obs": [], "sparsegpt": []}
    for selection in ExperimentStore(store).all_selections():
        config = selection.experiment.config
        if config.benchmark.name != "opt" or config.branch.name != "fisher_complement":
            continue
        options = config.branch.options
        if (options.get("b_refresh") != "layer" or
                int(options.get("max_layers", 12)) != 12 or
                options.get("modules", "all") != "all"):
            continue
        rule = options.get("mask_rule", "module_obs")
        if rule in candidates:
            candidates[rule].extend(
                run for run in selection.runs or []
                if run.status in {"done", "running"} and
                (run.path / "result.json").exists())
    if not candidates["module_obs"]:
        raise SystemExit("no completed refreshed-B OPT complement Forge run found")
    run = max(candidates["module_obs"], key=lambda item: item.launched_on)
    result = json.loads((run.path / "result.json").read_text())
    if result.get("status") != "complete":
        raise ValueError("Forge OPT result is incomplete")
    rows = result["curve"]
    accepted = [row for row in rows if row.get("admissible", True)]
    rejected = next((row for row in rows if not row.get("admissible", True)), None)

    # The legacy source data is gone; these anchors map its rasterized axes to data.
    x0, x52, y0, y14 = 125, 2123, 1098, 133
    xpixel = lambda value: x0 + (x52 - x0) * value / 52
    ypixel = lambda value: y0 - (y0 - y14) * value / 14
    x = [xpixel(row["removed_percent"]) for row in accepted]
    y = [ypixel(row["loss_damage_percent"]) for row in accepted]
    purple = "#753a99"
    figure = plt.figure(figsize=(width / 180, height / 180), dpi=180)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(image, extent=(0, width, height, 0))
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.axis("off")
    axis.add_patch(Rectangle((120, 63), 1350, 44, facecolor="white",
                             edgecolor="none", zorder=7))
    axis.text(125, 86, "128×2048 C4 calibration · historical selection offsets differ · seed 0",
              color="#555555", fontsize=11, va="center", zorder=8)
    axis.plot(x, y, color=purple, linewidth=3, marker="o", markersize=8,
              markerfacecolor="white", markeredgewidth=2.5, zorder=5)
    if rejected:
        rx, ry = xpixel(rejected["removed_percent"]), ypixel(rejected["loss_damage_percent"])
        axis.plot((x[-1], rx), (y[-1], ry), color=purple, linewidth=2,
                  linestyle=":", zorder=5)
        axis.scatter(rx, ry, marker="x", color=purple, linewidths=3, s=175, zorder=6)
    current = None
    if candidates["sparsegpt"]:
        run = max(candidates["sparsegpt"], key=lambda item: item.launched_on)
        current = json.loads((run.path / "result.json").read_text())
        completed = [row for row in current["curve"]
                     if row.get("admissible") is True]
        if len(completed) > 1:
            green = "#00856a"
            axis.plot([xpixel(row["removed_percent"]) for row in completed],
                      [ypixel(row["loss_damage_percent"]) for row in completed],
                      color=green, linewidth=3.5, marker="D", markersize=8,
                      markerfacecolor="white", markeredgewidth=2.5, zorder=6)
            rejected = next((row for row in current["curve"]
                             if row.get("admissible") is False), None)
            if rejected:
                last = completed[-1]
                axis.plot((xpixel(last["removed_percent"]),
                           xpixel(rejected["removed_percent"])),
                          (ypixel(last["loss_damage_percent"]),
                           ypixel(rejected["loss_damage_percent"])),
                          color=green, linewidth=2, linestyle=":", zorder=6)
                axis.scatter(xpixel(rejected["removed_percent"]),
                             ypixel(rejected["loss_damage_percent"]),
                             marker="x", color=green, linewidths=3, s=175, zorder=7)
    axis.text(2180, 300, "FORGE AXB UPDATES", color=purple, fontsize=16,
              fontweight="bold", va="top")
    axis.plot((2180, 2245), (365, 365), color=purple, linewidth=3, marker="o",
              markersize=7, markerfacecolor="white", markeredgewidth=2, zorder=5)
    axis.text(2265, 365, "Module-wide OBS mask", fontsize=12, va="center")
    axis.text(2180, 410, "10–30% accepted; 40% rejected (×)",
              color=purple, fontsize=11, va="top")
    if current is not None:
        green = "#00856a"
        axis.plot((2180, 2245), (485, 485), color=green, linewidth=3,
                  marker="D", markersize=7, markerfacecolor="white",
                  markeredgewidth=2, zorder=6)
        axis.text(2265, 485, "SparseGPT blockwise mask", fontsize=12,
                  va="center")
        accepted_targets = [round(100 * row["target_sparsity"])
                            for row in current["curve"]
                            if row.get("admissible") is True and
                            row.get("target_sparsity", 0) > 0]
        next_target = next((round(100 * run["target_sparsity"])
                            for run in current.get("runs", [])
                            if run.get("status") == "running"), None)
        status = (f"; {next_target}% running" if next_target is not None else "")
        axis.text(2180, 530, f"10–{max(accepted_targets)}% accepted{status}"
                  if accepted_targets else "No accepted targets yet",
                  color=green, fontsize=11, va="top")
    axis.text(2180, 625, "Both: B refreshed each block", fontsize=11, va="top")
    axis.text(2180, 665, "16,256 gradient samples", fontsize=11, va="top")
    axis.text(2180, 775, "Historical curves: raster only\n"
              "Colored overlays: exact Forge coordinates\n"
              "Protocols are not fully matched",
              fontsize=10, color="#555555", va="top", linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, pad_inches=0)
    plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("moons", "modadd", "berman", "opt", "opt_cpu"))
    parser.add_argument("--store", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical-opt-overlay", action="store_true")
    args = parser.parse_args()
    if args.historical_opt_overlay:
        if args.benchmark != "opt":
            parser.error("--historical-opt-overlay requires the opt benchmark")
        opt_historical_overlay(
            args.store, args.output or ROOT / "plots" / "opt" / "comparison.png")
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    found = 0
    for selection in ExperimentStore(args.store).all_selections():
        config = selection.experiment.config
        if config.benchmark.name != args.benchmark:
            continue
        if args.benchmark == "opt" and (
                int(config.branch.options.get("max_layers", 12)) != 12 or
                config.branch.options.get("modules", "all") != "all"):
            continue
        completed = [run for run in selection.runs or [] if run.status == "done"]
        if not completed:
            continue
        run = max(completed, key=lambda value: value.launched_on)
        files = [path for path in run.path.glob("*.json")
                 if path.name not in {"meta.json", "metrics.json", "checkpoint.json"}]
        if len(files) != 1:
            continue
        values = curve(json.loads(files[0].read_text()), args.benchmark)
        if values is None:
            continue
        x, y, ylabel = values
        axis.plot(x, y, marker="o", markersize=3,
                  label=f"{config.branch.name} ({selection.experiment.signature})")
        found += 1
    if not found:
        raise SystemExit(f"no completed Forge curves found for {args.benchmark}")
    axis.set(xlabel="Total parameters removed (%)", ylabel=ylabel,
             title=f"{args.benchmark}: Forge experiments")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    target = args.output or ROOT / "plots" / "forge" / f"{args.benchmark}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
