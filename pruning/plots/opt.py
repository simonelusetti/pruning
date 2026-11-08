from __future__ import annotations

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
        if curve.method not in best or performance(curve) > performance(best[curve.method]):
            best[curve.method] = curve
    names = {"continuous": "Continuous AXB", "continuous_neuron": "Neuron continuous",
             "sparsegpt": "SparseGPT", "classic": "Classic", "wanda": "Wanda", "flap": "FLAP"}
    colors = {"continuous": "#087e94", "continuous_neuron": "#8051a8", "sparsegpt": "#333333",
              "classic": "#2776c4", "wanda": "#bf7635", "flap": "#51945d"}
    figure, axis = plt.subplots(figsize=(11, 6))
    largest, maximum_removed = .25, 0.
    rho = reference.result.get("selection", {}).get("rho", .05)
    axis.axhline(100 * rho, color="#ae7a22", linestyle="--", linewidth=1.6,
                 label=f"{100*rho:g}% base margin")
    for kind, curve in sorted(best.items()):
        result = curve.result
        calibration, heldout = result["calibration"], result["heldout"]
        rows = [(index, row) for index, row in enumerate(curve.rows) if "heldout_damage_percent" in row]
        good = [row for index, row in rows if accepted(curve, index, row)]
        bad = next((row for index, row in rows if not accepted(curve, index, row)), None)
        name = names[kind]
        modules = curve.config.branch.options.get("modules", "all")
        if kind == "continuous_neuron" or modules == "fc1,fc2":
            name += " · MLP only"
        elif modules != "all":
            name += f" · {modules}"
        if kind == "classic":
            name += " W+S" if curve.config.branch.options.get("weights", False) else " S"
        budget = f"A {calibration['nsamples']}×{calibration['seqlen']}"
        if kind.startswith("continuous"):
            budget += f", B {calibration['b_sequences']}×{calibration['b_context']}"
            full = calibration['b_sequences'] * (calibration['b_context'] - 1)
            if calibration.get('b_tokens', full) not in (0, full):
                budget += f" ({calibration['b_tokens']} gradients)"
        label = f"{name}\n{budget}; held-out {heldout['count']} sequences"
        if curve.run.status == "running":
            label += " · running"
        positive = [row for row in good if removed(row) > 0]
        axis.plot([0.] + [removed(row) for row in positive],
                  [0.] + [row['heldout_damage_percent'] for row in positive],
                  color=colors[kind], marker="o", markersize=5, linewidth=2, label=label)
        for row in good + ([bad] if bad is not None else []):
            largest = max(largest, row['heldout_damage_percent'])
            maximum_removed = max(maximum_removed, removed(row))
            if row is bad:
                axis.scatter(removed(row), row['heldout_damage_percent'], marker="x", color=colors[kind],
                             s=100, linewidths=2.5, zorder=5)
            if "heldout_se" in row:
                error = 200 * row['heldout_se'] / heldout['reference_loss']
                largest = max(largest, row['heldout_damage_percent'] + error)
                axis.errorbar(removed(row), row['heldout_damage_percent'], yerr=error,
                              color=colors[kind], capsize=3, linewidth=1)
        print(f"{name}: {curve.run.signature} · accepted removal {removed(endpoint(curve)):.3f}%")
    axis.set(xlim=(-1, max(5, maximum_removed * 1.05)), ylim=(-.015, max(100*rho*1.15, largest*1.15)),
             xlabel="Decoder-linear parameters removed (%)", ylabel="Held-out loss increase from original (%)")
    axis.set_yscale("symlog", linthresh=.25, linscale=1)
    ticks = [value for value in (0, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200) if value < axis.get_ylim()[1]]
    axis.set_yticks(ticks, [f"{value:g}" for value in ticks])
    axis.grid(alpha=.25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    figure.suptitle(f"{reference.result['model']} · best configuration per branch", fontsize=15, fontweight="bold")
    figure.text(.5, .915, "Settings chosen by accepted removal, then validation damage; each curve is one saved run",
                ha="center", fontsize=10)
    figure.text(.5, .02, "× First rejected proposal · bars: ±2 SE · acceptance uses validation, not held-out loss\n"
                "Evaluation budgets may differ; each run is normalized to its own original-model loss.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .07, 1, .91))
    save_figure(figure, output)
    return True
