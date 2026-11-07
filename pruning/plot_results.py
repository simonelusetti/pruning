from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from forge import ExperimentStore

ROOT = Path(__file__).resolve().parents[1]


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


def opt_matched_heldout(store: Path, output: Path) -> None:
    selected = {}
    selection_only = None
    for selection in ExperimentStore(store).all_selections():
        config = selection.experiment.config
        if config.benchmark.name != "opt":
            continue
        if (config.benchmark.model != "facebook/opt-125m" or
                config.benchmark.nsamples != 128 or config.benchmark.seqlen != 2048 or
                config.benchmark.seed != 0 or config.benchmark.rho != .05 or
                config.benchmark.n_sigma != 2.):
            continue
        options = config.branch.options
        if (options.get("deletion") != "obs" or options.get("structure") != "none" or
                options.get("decision") != "validate"):
            continue
        if config.branch.name == "fisher_complement" and options.get("progressive", False):
            if (options.get("compensation") != "complement" or
                    options.get("mask_rule") != "sparsegpt" or options.get("mu") != 0):
                continue
            key = "progressive"
        elif (config.branch.name == "fisher_sparsegpt" and
              options.get("compensation") == "sparsegpt" and
              options.get("geometry") == "axb"):
            key = "fisher_sparsegpt"
        elif (config.branch.name == "fisher_sparsegpt" and
              options.get("compensation") == "sparsegpt" and
              options.get("geometry") == "axi"):
            key = "sparsegpt"
        else:
            continue
        for run in selection.runs or []:
            path = run.path / "result.json"
            if run.status not in {"done", "running"} or not path.exists():
                continue
            result = json.loads(path.read_text())
            heldout = result.get("heldout") or {}
            calibration = result.get("calibration") or {}
            selection_protocol = result.get("selection") or {}
            protocol = result.get("protocol") or {}
            if (result.get("status") not in {"running", "complete"} or
                    any(calibration.get(name) != value for name, value in {
                        "dataset": "c4", "nsamples": 128, "seqlen": 2048, "seed": 0,
                        "b_sequences": 128, "b_context": 128, "b_tokens": 16256}.items()) or
                    any(selection_protocol.get(name) != value for name, value in {
                        "dataset": "c4", "seed": 12345, "count": 4,
                        "rho": .05, "n_sigma": 2.}.items()) or
                    protocol.get("blocksize") != 128 or protocol.get("percdamp") != .01 or
                    result.get("model_dtype") != "float32" or
                    protocol.get("b_refresh") != ("frozen" if key == "sparsegpt" else "layer") or
                    protocol.get("b_damp_ratio") != .01):
                continue
            if (key == "progressive" and not heldout and run.status == "done" and
                    result["status"] == "complete" and
                    (selection_only is None or run.launched_on > selection_only[0])):
                selection_only = (run.launched_on, result)
            if heldout.get("count") != 128 or heldout.get("seed") != 314159:
                continue
            if key not in selected or run.launched_on > selected[key][0]:
                selected[key] = (run.launched_on, result)
    if not selected:
        raise SystemExit("no current matched OPT runs found")

    colors = {"progressive": "#087e94", "fisher_sparsegpt": "#d1495b",
              "sparsegpt": "#333333"}
    labels = {"progressive": "Progressive AXB complement",
              "fisher_sparsegpt": "Fisher-aware SparseGPT",
              "sparsegpt": "SparseGPT (A only)"}
    sparsegpt_run = selected.get("sparsegpt")
    matched_sparsegpt_points = sum(
        row.get("target_sparsity", 0) > 0 and row.get("admissible", True)
        for row in (sparsegpt_run[1].get("curve", []) if sparsegpt_run else []))
    baseline = next(iter(selected.values()))[1]["heldout"]["reference_loss"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharex=True, sharey=True)
    fields = ("loss_damage_percent", "heldout_damage_percent")
    for axis, field in zip(axes, fields):
        axis.set(xlim=(-1, 53), ylim=(-.015, 8),
                 xlabel="Decoder-linear parameters removed (%)",
                 ylabel="Loss increase from original (%)")
        axis.set_title("Selection C4 · 4 sequences" if field == fields[0] else
                       "Held-out C4 · 128 sequences")
        axis.set_yscale("symlog", linthresh=.25, linscale=1)
        axis.set_yticks([0, .1, .2, .5, 1, 2, 5, 7])
        axis.set_yticklabels(["0", "0.1", "0.2", "0.5", "1", "2", "5", "7"])
        axis.grid(alpha=.25)
        axis.axhline(5, color="#ae7a22", linestyle="--", linewidth=1.6,
                     label="5% base margin" if field == fields[0] else None)
        if field == "loss_damage_percent" and matched_sparsegpt_points < 3:
            axis.plot([0, 10, 20, 30, 40, 45], [0, .3, .6, 1.4, 3.1, 4.6],
                      color="#555555", linestyle=":", linewidth=2,
                      label="SparseGPT · historical raster estimate")
        if field == "loss_damage_percent" and selection_only is not None:
            earlier = [row for row in selection_only[1]["curve"]
                       if row.get("target_sparsity", 0) > 0 and row.get("admissible", True)]
            axis.plot([0] + [row["removed_percent"] for row in earlier],
                      [0] + [row[field] for row in earlier], color="#63a9b6",
                      linestyle="--", linewidth=1.8,
                      label="Progressive AXB complement · completed selection-only run")
        limit_labeled = False
        for key in ("progressive", "fisher_sparsegpt", "sparsegpt"):
            if key not in selected:
                continue
            result = selected[key][1]
            if abs(result["heldout"]["reference_loss"] - baseline) > 1e-6:
                raise ValueError("current OPT runs have different held-out reference losses")
            available = [item for item in result["curve"]
                         if item.get("target_sparsity", 0) > 0 and field in item]
            accepted = [item for item in available if item.get("admissible", True)]
            if accepted:
                axis.plot([0] + [item["removed_percent"] for item in accepted],
                          [0] + [item[field] for item in accepted],
                          color=colors[key], linewidth=2.5, marker="o", markersize=6,
                          label=labels[key])
            for item in available:
                if item.get("admissible") is False:
                    axis.scatter(item["removed_percent"], item[field],
                                 marker="x", color=colors[key], s=110,
                                 linewidths=2.5, zorder=5,
                                 label=f"{labels[key]} · rejected" if not accepted else None)
                if field == "heldout_damage_percent":
                    axis.errorbar(item["removed_percent"], item[field],
                                  yerr=200 * item["heldout_se"] / baseline,
                                  color=colors[key], capsize=3, linewidth=1)
                else:
                    selection_base = result["curve"][0]["selection_loss"]
                    limit = 100 * (result["selection"]["rho"] +
                                   result["selection"]["n_sigma"] *
                                   item["selection_se"] / selection_base)
                    axis.scatter(item["removed_percent"], limit, marker="_",
                                 color="#ae7a22", s=350, linewidths=2.5,
                                 label="Actual limit: 5% + 2 SE" if
                                 not limit_labeled else None, zorder=6)
                    limit_labeled = True
    handles, labels_shown = [], []
    for axis in axes:
        found_handles, found_labels = axis.get_legend_handles_labels()
        for handle, label in zip(found_handles, found_labels):
            if label not in labels_shown:
                handles.append(handle)
                labels_shown.append(label)
    figure.legend(handles, labels_shown, loc="upper center", ncol=3,
                  bbox_to_anchor=(.5, .89), frameon=False)
    figure.suptitle("OPT-125M · current matched experiment", fontsize=16, fontweight="bold")
    figure.text(.5, .91,
                "Same 128×2048 C4 calibration; held-out seed 314159 · stopping uses selection C4 at 5% + 2 SE",
                ha="center", fontsize=10, color="#555555")
    figure.text(.5, .045,
                ("Historical SparseGPT: approximate, different selection offset · "
                 "dashed progressive: earlier run" if matched_sparsegpt_points < 3 else
                 "Dashed progressive curve: earlier selection-only run"),
                ha="center", fontsize=9, color="#555555")
    pending = [labels[key] for key in ("progressive", "fisher_sparsegpt")
               if key not in selected or selected[key][1].get("status") != "complete"]
    if pending:
        figure.text(.5, .015, f"Incomplete saved run: {', '.join(pending)} · held-out comparison near 50% is pending",
                    ha="center", fontsize=10, color="#555555")
    figure.tight_layout(rect=(0, .08, 1, .80))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("moons", "modadd", "berman", "opt", "opt_cpu"))
    parser.add_argument("--store", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matched-opt-heldout", action="store_true")
    args = parser.parse_args()
    if args.matched_opt_heldout and args.benchmark != "opt":
        parser.error("--matched-opt-heldout requires opt")
    if args.benchmark == "opt":
        opt_matched_heldout(
            args.store, args.output or ROOT / "outputs" / "plots" / "opt" / "comparison.png")
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
    target = args.output or ROOT / "outputs" / "plots" / "forge" / f"{args.benchmark}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
