#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "native_baselines" / "results" / "opt_published"
OUTPUT = ROOT / "plots" / "opt" / "comparison.png"


def load(name):
    data = json.loads((RESULTS / name).read_text())
    if data.get("status") != "complete":
        raise ValueError(f"incomplete result: {name}")
    return data


def main():
    exact = (
        ("classic structural exact", "classic_exact.json", "#2878c8", "-"),
        ("classic W+S exact", "classic_weights_exact.json", "#8b4ca5", "-"),
    )
    classics = (exact if all((RESULTS / name).exists() for _, name, _, _ in exact)
                else (("classic fast", "classic_fast.json", "#2878c8", "-"),))
    loaded = [(label, load(name), color, style)
              for label, name, color, style in classics]
    sparse_path = RESULTS / "sparsegpt_curve.json"
    sparse = json.loads(sparse_path.read_text()) if sparse_path.exists() else None
    fisher_sparse_path = RESULTS / "fisher_sparsegpt.json"
    fisher_sparse = (json.loads(fisher_sparse_path.read_text())
                     if fisher_sparse_path.exists() else None)
    fisher_miqp = None
    for name in ("fisher_local_miqp_10pct.json", "fisher_local_miqp_full.json"):
        path = RESULTS / name
        if path.exists():
            candidate = json.loads(path.read_text())
            if len(candidate.get("curve", ())) > 1:
                fisher_miqp = candidate
                break
    denominator = loaded[0][1]["prunable_decoder_linear_parameters"]

    fig, axis = plt.subplots(figsize=(12, 7))
    for label, classic, color, style in loaded:
        history = classic["history"]
        base = history[0]["val_b_loss"]
        classic_x = [100 * row["removed"] / denominator for row in history]
        classic_y = [100 * (row["val_b_loss"] / base - 1) for row in history]
        axis.plot(classic_x, classic_y, color=color, linestyle=style,
                  linewidth=2.8, label=label)
        stop = classic.get("stop_index")
        endpoint = stop if stop is not None else len(history) - 1
        axis.scatter(classic_x[endpoint], classic_y[endpoint], facecolors="white",
                     edgecolors=color, linewidths=2.2, s=105, zorder=4)
        rejected = classic.get("rejected_candidate")
        if rejected:
            axis.scatter(100 * rejected["removed"] / denominator,
                         100 * rejected["relative_damage"], marker="x",
                         color=color, linewidths=2.2, s=85, zorder=4)

    if sparse:
        curve = sparse["curve"]
        sparse_x = [row["removed_percent"] for row in curve]
        sparse_y = [row["loss_damage_percent"] for row in curve]
        stop = sparse.get("stop_index")
        endpoint = stop + 1 if stop is not None else len(curve)
        axis.plot(sparse_x[:endpoint], sparse_y[:endpoint], color="#333333",
                  linewidth=2.8, label="SparseGPT")
        if endpoint < len(curve):
            axis.plot(sparse_x[stop:], sparse_y[stop:], color="#333333",
                      linewidth=2, linestyle=":", alpha=.65,
                      label="SparseGPT post-stop diagnostic")
        if stop is not None:
            axis.scatter(sparse_x[stop], sparse_y[stop], facecolors="white",
                         edgecolors="#333333", linewidths=2.2, s=105, zorder=4)

    if fisher_sparse and len(fisher_sparse.get("curve", ())) > 1:
        curve = fisher_sparse["curve"]
        label = "Fisher-aware SparseGPT"
        if fisher_sparse.get("status") != "complete":
            label += " (running)"
        axis.plot([row["removed_percent"] for row in curve],
                  [row["loss_damage_percent"] for row in curve],
                  color="#d1495b", linewidth=2.8, marker="o", label=label)

    general_path = RESULTS / "fisher_general_miqp.json"
    if general_path.exists():
        general = json.loads(general_path.read_text())
        sweep = general.get("sweep", [])
        stop = next((i for i, row in enumerate(sweep)
                     if not row.get("admissible", True)), None)
        shown = sweep if stop is None else sweep[:stop + 1]
        if shown:
            axis.plot([100 * row["frac_removed"] for row in shown],
                      [row["loss_damage_percent"] for row in shown],
                      color="#60813e", linewidth=2.2, marker=".",
                      label="Fisher General MIQP (earlier candidate domain)")
        if stop is not None:
            row = sweep[stop]
            axis.scatter(100 * row["frac_removed"], row["loss_damage_percent"],
                         marker="x", color="#60813e", s=85, zorder=4)

    complement_path = RESULTS / "fisher_complement.json"
    if complement_path.exists():
        complement = json.loads(complement_path.read_text())
        curve = complement.get("curve", [])
        if len(curve) > 1:
            label = "AXB OBS: full-layer compensation"
            if complement.get("status") != "complete":
                label += f" ({complement.get('status', 'unknown')})"
            axis.plot([row["removed_percent"] for row in curve],
                      [row["loss_damage_percent"] for row in curve],
                      color="#e09f3e", linewidth=2.2, marker="s", label=label)
        else:
            status = complement.get("status", "unknown")
            axis.text(.02, .03,
                      f"Full-layer AXB compensation: {status}; no evaluated target yet",
                      transform=axis.transAxes, fontsize=10, color="#a66b13")

    if fisher_miqp and len(fisher_miqp.get("curve", ())) > 1:
        curve = fisher_miqp["curve"]
        x = [row["removed_percent"] for row in curve]
        y = [row["loss_damage_percent"] for row in curve]
        label = "Fisher General: local AXB MIQP"
        if fisher_miqp.get("status") != "complete":
            label += " (running)"
        axis.plot(x, y, color="#2a9d8f", linewidth=2.2, marker="D",
                  markersize=7, label=label)
        axis.annotate(f"{x[-1]:.3f}%", (x[-1], y[-1]), xytext=(16, 14),
                      textcoords="offset points", color="#2a9d8f", fontsize=9,
                      arrowprops={"arrowstyle": "-", "color": "#2a9d8f"})

    axis.axhline(5, color="#666666", linestyle="--", linewidth=1.4,
                 label="5% practical margin")
    axis.set_xlim(0, 52)
    axis.set_xlabel("decoder-linear parameters removed")
    axis.set_ylabel("C4 selection-loss increase from original")
    axis.xaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    axis.yaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    axis.grid(axis="y", color="#dddddd")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2)
    mode = ("exact classic pruning" if all(
        (RESULTS / name).exists() for _, name, _, _ in exact) else "classic fast")
    fisher_names = []
    if fisher_sparse:
        fisher_names.append("Fisher-aware SparseGPT")
    if fisher_miqp:
        fisher_names.append("local Fisher MIQP")
    fisher_extra = f", {', '.join(fisher_names)}" if fisher_names else ""
    axis.set_title(f"OPT-125M: {mode}{fisher_extra}, and published SparseGPT", loc="left",
                   fontsize=20, fontweight="bold", pad=24)
    subtitle = "128×2048 C4 calibration pool · identical C4 selection split · seed 0"
    if not sparse:
        subtitle += " · SparseGPT sweep running"
    axis.text(0, 1.02, subtitle,
              transform=axis.transAxes, color="#555555", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
