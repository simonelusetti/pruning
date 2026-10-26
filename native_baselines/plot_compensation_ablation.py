#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "plots" / "compensation_ablation.png"


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    sources = (
        ("Berman", ROOT / "native_baselines/results_berman/simplenn_mnist/compensation_ablation.json", "test"),
        ("OPT-125M", ROOT / "native_baselines/results/opt_published/compensation_ablation.json", "heldout"),
    )
    colors = {"axb": "#2878c8", "exact_fisher": "#b8792e"}
    for row, (label, path, split) in zip(axes, sources):
        data = json.loads(path.read_text()) if path.exists() else {}
        evaluations = data.get("evaluations", [])
        row[0].set_ylabel(label)
        for geometry in sorted({e["geometry"] for e in evaluations}):
            entries = sorted((e for e in evaluations if e["geometry"] == geometry), key=lambda e: e["alpha"])
            x = [e["alpha"] for e in entries]
            baseline = entries[0]["regularized_objective"]
            name = "Layerwise AXB" if geometry == "axb" else "Exact Fisher, all layers"
            color = colors[geometry]
            row[0].plot(x, [e["regularized_objective"] / baseline for e in entries],
                        marker="o", color=color, label=name)
            row[1].plot(x, [e[split]["output_kl"] for e in entries], marker="o", color=color)
            row[2].errorbar(x, [e[split]["loss_damage_percent"] for e in entries],
                            yerr=[200 * e[split]["paired_se"] / e[split]["reference_loss"] for e in entries],
                            marker="o", capsize=3, color=color)
            row[2].plot(x, [e["validation"]["loss_damage_percent"] for e in entries],
                        linestyle=":", color=color, alpha=.65)
        if not evaluations:
            for ax in row:
                ax.text(.5, .5, data.get("status", "pending"), ha="center", transform=ax.transAxes)
        for ax in row:
            ax.set_xlabel("Compensation scale α (0 = deletion only)")
            ax.grid(alpha=.25)
        row[2].axhline(0, color="#777777", linewidth=.8)
        if evaluations:
            row[0].legend(frameon=False)
        row[0].set_title("Regularized quadratic / deletion-only value")
        row[1].set_title("Held-out output KL, nats per prediction")
        row[2].set_title("Held-out CE increase (%); bars: ±2 SE\nDotted: validation CE increase")
    fig.suptitle("Fixed-mask compensation ablation: 30% weight deletion", fontsize=17)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
