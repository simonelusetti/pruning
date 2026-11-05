from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def opt_complement(result_path: Path, output: Path) -> None:
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete" or result["fisher_strategy"]["compensation"] != "complement":
        raise ValueError("expected a completed OPT full-complement result")
    rows = result["curve"]
    base = rows[0]["selection_loss"]
    accepted = [row for row in rows if row.get("admissible", True)]
    rejected = next((row for row in rows if not row.get("admissible", True)), None)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    purple = "#753a99"
    x = [row["removed_percent"] for row in accepted]
    y = [row["loss_damage_percent"] for row in accepted]
    error = [200 * row.get("selection_se", 0) / base for row in accepted]
    ax.errorbar(x, y, yerr=error, color=purple, marker="o", linewidth=2.5,
                markersize=7, capsize=3, label="Full complement · accepted")
    if rejected:
        rx = rejected["removed_percent"]
        ry = rejected["loss_damage_percent"]
        ax.plot([x[-1], rx], [y[-1], ry], color=purple, linestyle=":", linewidth=2)
        ax.errorbar(rx, ry, yerr=200 * rejected["selection_se"] / base,
                    color=purple, marker="x", linestyle="none", markersize=10,
                    markeredgewidth=2.5, capsize=3, label="Rejected proposal")
    ax.axhline(5, color="#666666", linestyle="--", linewidth=1.3,
               label="5% practical margin")
    ax.set(xlim=(0, 43), ylim=(0, 8.5),
           xlabel="Decoder-linear parameters removed (%)",
           ylabel="C4 selection-loss increase (%)",
           title="OPT-125M · refreshed-B full-complement pruning")
    ax.grid(axis="y", color="#dddddd")
    ax.legend(frameon=False)
    fig.text(.12, .015,
             "Four C4 selection sequences · error bars: ±2 paired SE · B refreshed each block",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .045, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt-result", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "plots/report/opt_full_complement.png")
    args = parser.parse_args()
    opt_complement(args.opt_result, args.output)


if __name__ == "__main__":
    main()
