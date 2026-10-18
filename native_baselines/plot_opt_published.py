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
    mode = "exact classic pruning" if len(loaded) == 2 else "classic fast"
    axis.set_title(f"OPT-125M: {mode} and published SparseGPT", loc="left",
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
