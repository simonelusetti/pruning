#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "native_baselines/results/opt_published/b_estimation_sweep_10pct.json"
CONFIRMATION = ROOT / "native_baselines/results/opt_published/b_confirmation_10pct_128seq.json"
COMPLEMENT = ROOT / "native_baselines/results/opt_published/axb_complement_b128_refresh_10pct.json"
OUTPUT = ROOT / "plots/opt/b_estimation_10pct.png"


def main():
    data = json.loads(RESULT.read_text())
    runs = data["runs"]
    if any(run["status"] != "complete" for run in runs):
        raise ValueError("B estimation sweep has incomplete runs")
    baseline = next(run for run in runs if run["case"] == "sparsegpt_b256")
    refresh = next(run for run in runs if run["b_refresh"] == "layer")
    frozen = [run for run in runs if run["method"] == "axb"
              and run["b_refresh"] == "frozen"]
    confirmed = json.loads(CONFIRMATION.read_text())
    complement_data = json.loads(COMPLEMENT.read_text())
    if confirmed["evaluation"] != complement_data["evaluation"]:
        raise ValueError("full-complement and confirmation evaluation splits differ")
    confirmed_runs = {run["case"]: run for run in confirmed["runs"]}
    complement = complement_data["runs"][0]
    if complement["status"] != "complete":
        raise ValueError("full-complement result is incomplete")

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 13,
                         "axes.labelsize": 11, "legend.fontsize": 9.5})
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.6))
    fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.70, wspace=.19)
    series = ((1e-6, "#2766a0", r"AXB, $B$ damping $10^{-6}$"),
              (1e-2, "#d07a29", r"AXB, $B$ damping $10^{-2}$"))
    for axis, (split, title) in zip(
            axes[:2], (("heldout", "Held-out C4 · 8 sequences"),
                       ("selection", "Selection C4 · 4 sequences"))):
        for damping, color, label in series:
            rows = sorted((run for run in frozen
                           if run["b_damping_ratio"] == damping),
                          key=lambda run: run["b_tokens"])
            axis.plot([run["b_tokens"] for run in rows],
                      [run[split]["damage_percent"] for run in rows],
                      marker="o", markersize=7, linewidth=2.3,
                      color=color, label=label, zorder=3)
        axis.axhline(baseline[split]["damage_percent"], color="#5f6670",
                     linestyle=(0, (5, 3)), linewidth=1.8,
                     label="Fisher-aware SparseGPT · 256 samples", zorder=2)
        axis.scatter(refresh["b_tokens"], refresh[split]["damage_percent"],
                     marker="*", s=240, facecolor="#2d9a74", edgecolor="white",
                     linewidth=1, label="AXB · 16k samples · refresh each layer",
                     zorder=5)
        if split == "selection":
            axis.scatter(16256, complement[split]["damage_percent"],
                         marker="D", s=105, facecolor="#8c52a8", edgecolor="white",
                         linewidth=1, label="AXB full complement · refresh each layer",
                         zorder=6)
        axis.set_xscale("log")
        axis.set_xticks([256, 1016, 4064, 16256, 262016],
                        ["256", "1k", "4k", "16k", "262k"])
        axis.set_xlim(180, 400000)
        axis.set_ylim(.12, .53)
        axis.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _position: f"{value:.2f}%"))
        axis.set_title(title, loc="left", fontweight="semibold", pad=12)
        axis.set_xlabel("Gradient observations used to estimate B")
        axis.grid(axis="y", color="#e4e8ec", linewidth=.9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Loss increase from original · lower is better")
    heldout_runs = (
        confirmed_runs["sparsegpt_b256"],
        confirmed_runs["axb_b16256_d1e2"],
        confirmed_runs["axb_b16256_d1e2_refresh_layer"],
        complement,
    )
    labels_128 = ("SparseGPT\n256", "AXB frozen\n16k", "AXB refresh\n16k",
                  "Full complement\n16k")
    colors_128 = ("#5f6670", "#d07a29", "#2d9a74", "#8c52a8")
    values_128 = [run["heldout"]["damage_percent"] for run in heldout_runs]
    axes[2].bar(range(4), values_128, color=colors_128, width=.68)
    for index, value in enumerate(values_128):
        axes[2].text(index, value + .006, f"{value:.3f}%", ha="center", fontsize=9)
    axes[2].set_xticks(range(4), labels_128)
    axes[2].set_ylim(0, .32)
    axes[2].set_title("Held-out C4 · 128 sequences", loc="left",
                      fontweight="semibold", pad=12)
    axes[2].yaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: f"{value:.2f}%"))
    axes[2].grid(axis="y", color="#e4e8ec", linewidth=.9)
    axes[2].set_axisbelow(True)
    axes[2].spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(.52, .865), frameon=False)
    fig.suptitle("OPT-125M · B estimation at 10% target deletion",
                 x=.065, y=.975, ha="left", fontsize=19, fontweight="bold")
    fig.text(.065, .915, "Same C4 calibration and selection split; held-out sample counts are labeled per panel",
             color="#525b65", fontsize=10)
    fig.text(.065, .055,
             "Actual removal: about 11.05% for SparseGPT variants, 9.99% for full complement. "
             "Deletion masks differ; 262k uses every calibration-token position.",
             color="#525b65", fontsize=9.5)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
