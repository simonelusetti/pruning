from __future__ import annotations

from functools import lru_cache
import matplotlib.pyplot as plt
from pruning.plots.common import accepted, removed, save_figure


def comparison_key(curve):
    result = curve.result
    calibration, heldout, selection = (result.get(key, {}) for key in ("calibration", "heldout", "selection"))
    options = curve.config.branch.options
    return (result.get("model"), result.get("model_dtype"),
            options.get("max_layers", curve.config.benchmark.get("max_layers", 12)),
            result.get("parameter_denominator"), heldout.get("dataset"),
            heldout.get("seqlen", calibration.get("seqlen")),
            selection.get("enabled", True), selection.get("rho", .05), selection.get("n_sigma", 2.))


def endpoint(curve):
    return max((row for index, row in enumerate(curve.rows) if accepted(curve, index, row)), key=removed)


def performance(curve):
    row = endpoint(curve)
    return removed(row), -row.get("loss_damage_percent", float("inf")), curve.run.launched_on


def pruning_scope(curve):
    if curve.method == "continuous_neuron":
        return "fc1,fc2"
    return ",".join(sorted(curve.config.branch.options.get("modules", "all").split(",")))


@lru_cache(maxsize=None)
def model_blueprint(model_name):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    return model


def model_parameter_count(model_name):
    return sum(parameter.numel() for parameter in model_blueprint(model_name).parameters())


def pruning_capacity(curve):
    import torch
    model = model_blueprint(curve.result["model"])
    if curve.method.startswith("discrete_simple") and curve.result.get("pruning_parameters"):
        parameters = dict(model.named_parameters())
        return sum(parameters[name].numel() for name in curve.result["pruning_parameters"])
    count = curve.config.branch.options.get("max_layers", curve.config.benchmark.get("max_layers", 12))
    scope = pruning_scope(curve)
    names = scope.split(",")
    capacity = 0
    for layer in model.model.decoder.layers[:count]:
        if curve.method == "continuous_neuron":
            capacity += layer.fc1.weight.numel() + layer.fc1.bias.numel() + layer.fc2.weight.numel()
        else:
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear) and (scope == "all" or name in names):
                    capacity += module.weight.numel()
                    if (curve.method == "classic" or
                            ((curve.method.startswith("canonical_sparsegpt") or curve.method == "discrete_simple" or curve.result.get("posthoc_canonicalization"))
                             and name == "fc1")) and module.bias is not None:
                        capacity += module.bias.numel()
    return capacity


def plot(curves, output, trigger=None):
    eligible = [curve for curve in curves if curve.result.get("heldout", {}).get("count")
                and any("heldout_damage_percent" in row and removed(row) > 0 for row in curve.rows)]
    if not eligible:
        return False
    reference = next((curve for curve in eligible if trigger == curve.run.path),
                     max(eligible, key=lambda curve: curve.run.launched_on))
    best = {}
    for curve in eligible:
        if comparison_key(curve) != comparison_key(reference):
            continue
        if curve.run.status == "running":
            continue
        key = curve.method, pruning_scope(curve)
        if key not in best or performance(curve) > performance(best[key]):
            best[key] = curve
    selected = list(sorted(best.items()))
    best_paths = {curve.run.path for curve in best.values()}
    for curve in eligible:
        if (comparison_key(curve) == comparison_key(reference) and curve.run.path not in best_paths
                and (curve.run.status == "running" or curve.run.path == trigger)):
            selected.append(((curve.method, pruning_scope(curve)), curve))
    names = {"continuous": "Continuous AXB", "continuous_global": "Continuous AXB · global OBS",
             "discrete_simple": "Discrete simple",
             "discrete_simple_global": "Canonical L0 · global fixed λ · permanent deletions",
             "continuous_neuron": "Neuron continuous",
             "canonical_sparsegpt": "Canonical SparseGPT",
             "canonical_sparsegpt_control": "Canonical SparseGPT · power 0 control",
             "sparsegpt": "SparseGPT", "classic": "Classic", "wanda": "Wanda", "flap": "FLAP"}
    colors = {"continuous": "#087e94", "continuous_global": "#4264c4",
              "discrete_simple": "#ab5f16",
              "discrete_simple_global": "#a84e91",
              "continuous_neuron": "#8051a8", "sparsegpt": "#333333",
              "canonical_sparsegpt": "#c43c64", "canonical_sparsegpt_control": "#73964b",
              "classic": "#2776c4", "wanda": "#bf7635", "flap": "#51945d"}
    figure, axes = plt.subplots(1, 2, figsize=(17, max(6.5, 1.05 * len(selected))), sharey=True)
    largest, maximum_removed = .25, [0., 0.]
    rho = reference.result.get("selection", {}).get("rho", .05)
    full_total = reference.result.get("total_model_parameters") or model_parameter_count(reference.result["model"])
    for axis in axes:
        axis.axhline(100 * rho, color="#ae7a22", linestyle="--", linewidth=1.6,
                     label=f"{100*rho:g}% base margin")
    for (kind, modules), curve in selected:
        result = curve.result
        calibration, heldout = result["calibration"], result["heldout"]
        rows = [(index, row) for index, row in enumerate(curve.rows) if "heldout_damage_percent" in row]
        good = [row for index, row in rows if accepted(curve, index, row)]
        bad = next((row for index, row in rows if not accepted(curve, index, row)), None)
        name = names[kind]
        if kind == "discrete_simple":
            name += " · canonical L0" if curve.config.branch.options.get("canonicalization", False) else " · L0 IHT"
        if kind == "canonical_sparsegpt":
            name += f" · power {curve.config.branch.options.completion_power:g}"
        if kind == "continuous_neuron" or modules == "fc1,fc2":
            name += " · MLP only"
        elif modules != "all":
            name += f" · {modules}"
        elif kind in {"continuous", "continuous_global"}:
            name += " · all linear modules"
        cleanup = result.get("posthoc_canonicalization")
        if cleanup:
            name += " · endpoint cleaned"
        if kind == "classic":
            name += " W+S" if curve.config.branch.options.get("weights", False) else " S"
        budget = f"A {calibration['nsamples']}×{calibration['seqlen']}"
        if kind.startswith(("continuous", "discrete_simple")):
            budget += f", B {calibration['b_sequences']}×{calibration['b_context']}"
            full = calibration['b_sequences'] * (calibration['b_context'] - 1)
            if calibration.get('b_tokens', full) not in (0, full):
                budget += f" ({calibration['b_tokens']} gradients)"
        capacity = pruning_capacity(curve)
        label = f"{name}\n{budget}; held-out {heldout['count']} sequences\nEligible pool: {capacity:,} parameters"
        if kind.startswith(("continuous", "discrete_simple")):
            refresh = result.get("method_options", {}).get("b_refresh", curve.config.branch.options.get("b_refresh"))
            label += "\nB refresh: " + {"layer": "per transformer block", "step": "per proposal",
                                          "frozen": "initial snapshot"}.get(refresh, str(refresh))
        if curve.run.status == "running":
            label += " · running"
        elif curve.run.path not in best_paths:
            label += " · requested comparison"
        color = "#c77c26" if kind == "continuous" and modules == "fc1,fc2" else colors[kind]
        positive = [row for row in good if removed(row) > 0]
        total = result.get("total_model_parameters") or model_parameter_count(result["model"])
        scales = (result["parameter_denominator"] / capacity, result["parameter_denominator"] / total)
        for index, (axis, scale) in enumerate(zip(axes, scales)):
            axis.plot([0.] + [scale * removed(row) for row in positive],
                      [0.] + [row['heldout_damage_percent'] for row in positive],
                      color=color, marker="s" if curve.run.path not in best_paths else "o", markersize=5,
                      linestyle="--" if curve.run.path not in best_paths else "-", linewidth=2, label=label)
            if cleanup:
                point = cleanup["point"]
                before, after = scale * cleanup["source_removed_percent"], scale * removed(point)
                damage = point["heldout_damage_percent"]
                axis.plot([before, after], [damage, damage], color=color, linestyle=":", linewidth=2)
                axis.scatter(before, damage, facecolors="none", edgecolors=color, s=50, zorder=6)
                axis.scatter(after, damage, marker="*", color=color, s=110, zorder=7)
            for row in good + ([bad] if bad is not None else []):
                largest = max(largest, row['heldout_damage_percent'])
                maximum_removed[index] = max(maximum_removed[index], scale * removed(row))
                if row is bad:
                    axis.scatter(scale * removed(row), row['heldout_damage_percent'], marker="x", color=color,
                                 s=100, linewidths=2.5, zorder=5)
                if "heldout_se" in row:
                    error = 200 * row['heldout_se'] / heldout['reference_loss']
                    largest = max(largest, row['heldout_damage_percent'] + error)
                    axis.errorbar(scale * removed(row), row['heldout_damage_percent'], yerr=error,
                                  color=color, capsize=3, linewidth=1)
        print(f"{name}: {curve.run.signature} · eligible pool {capacity:,} · accepted removal "
              f"{scales[0] * removed(endpoint(curve)):.3f}% of eligible, "
              f"{scales[1] * removed(endpoint(curve)):.3f}% of model")
    for axis, maximum, label, title in zip(axes, maximum_removed,
            ("Branch-prunable parameters", "All model parameters"),
            ("Branch-prunable parameters\nDenominator varies by curve (see legend)",
             f"All model parameters\nDenominator: {full_total:,}")):
        axis.set(xlim=(-1, max(5, maximum * 1.05)), ylim=(-.015, max(100*rho*1.15, largest*1.15)),
                 xlabel=f"{label} removed (%)", title=title)
        axis.set_yscale("symlog", linthresh=.25, linscale=1)
        ticks = [value for value in (0, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200) if value < axis.get_ylim()[1]]
        axis.set_yticks(ticks, [f"{value:g}" for value in ticks])
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Held-out loss increase from original (%)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    figure.suptitle(f"{reference.result['model']} · pruning comparison", fontsize=14, fontweight="bold")
    figure.text(.5, .915, "Best completed configuration per branch/scope · dashed: active or requested comparisons",
                ha="center", fontsize=10)
    figure.text(.5, .02, "× First rejected proposal · bars: ±2 SE · acceptance uses validation, not held-out loss\n"
                "Evaluation budgets may differ; each run is normalized to its own original-model loss."
                + ("\n★ Exact cleanup of saved endpoint only; open circle: before cleanup. Earlier points unchanged."
                   if any(curve.result.get("posthoc_canonicalization") for _, curve in selected) else ""),
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, .91))
    save_figure(figure, output)
    return True
