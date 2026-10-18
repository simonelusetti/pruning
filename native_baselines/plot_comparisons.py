#!/usr/bin/env python3
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "NNpruning" / "results"
BERMAN = ROOT / "native_baselines" / "results_berman" / "simplenn_mnist"
MODADD_GRID = ROOT / "native_baselines" / "results_modadd_grid"
PLOTS = ROOT / "plots"
COLORS = {
    "classic": "#777777",
    "fisher": "#2878c8",
    "dual": "#9457a4",
    "quant002": "#147d92",
    "quant005": "#2a9d6f",
    "quant01": "#42a66c",
    "quant02": "#d5822a",
    "quant_post": "#b05f24",
    "quant04": "#c44e52",
    "sparsegpt": "#222222",
}


def read(path):
    data = json.loads(path.read_text())
    if data.get("status") != "complete":
        raise ValueError(f"incomplete result {path}")
    return data


def local_test_curve(path):
    data = read(path)
    history = data["history"]
    base = history[0]["test_loss"]
    return ([100 * row["frac_removed"] for row in history],
            [row["test_loss"] / base for row in history],
            data.get("stop_index"), None)


def berman_loss_curve(path):
    data = read(path)
    history = data["history"]
    base = history[0]["test_loss"]
    stop = berman_stop(data)
    history = history[:stop + 1] if stop is not None else history
    return ([100 * row["frac_removed"] for row in history],
            [row["test_loss"] / base for row in history],
            stop, None)


def sweep_loss_curve(path):
    data = read(path)
    history = data["history"]
    base = history[0]["test_loss"]
    return ([100 * row["frac_removed"] for row in history],
            [row["test_loss"] / base for row in history],
            data.get("stop_index"), None)


def berman_stop(data):
    if data.get("termination_reason") == "candidate_exhaustion":
        return None
    if data.get("stop_index") is not None:
        return data["stop_index"]
    history = data["history"]
    base = history[0]["val_loss"]
    protocol = data.get("protocol", {})
    n_sigma = protocol.get("n_sigma", 2.)
    rho = protocol.get("rho", .05)
    for index, row in enumerate(history[1:], 1):
        if row["val_delta"] > n_sigma * row["val_se"] + rho * base:
            return index - 1
    return len(history) - 1


def available(specs):
    curves = []
    missing = []
    for label, path, loader, color, style in specs:
        if path.exists():
            x, y, stop, rejected = loader(path)
            if len(x) != len(y) or not x:
                raise ValueError(f"malformed curve in {path}")
            if any(not math.isfinite(value) for value in (*x, *y)):
                raise ValueError(f"non-finite curve value in {path}")
            if any(right < left - 1e-9 for left, right in zip(x, x[1:])):
                raise ValueError(f"removed-parameter count decreases in {path}")
            if stop is not None and not 0 <= stop < len(x):
                raise ValueError(f"invalid stop index in {path}")
            curves.append((label, x, y, stop, rejected, color, style))
        else:
            missing.append(path.name)
    return curves, missing


def validate_local_protocol(paths):
    fingerprints = []
    for path in paths:
        if not path.exists():
            continue
        data = read(path)
        protocol = data.get("protocol", {})
        fingerprints.append((path, (
            data.get("benchmark"),
            data.get("checkpoint_sha256"), data.get("validation_split_seed"),
            data["history"][0]["k_eff"], protocol.get("stop_split"),
            protocol.get("criterion"), protocol.get("n_sigma"),
            protocol.get("tol_abs"), protocol.get("tol_frac"),
            protocol.get("stop_reference"))))
    if fingerprints and any(value != fingerprints[0][1] for _, value in fingerprints[1:]):
        details = ", ".join(f"{path.name}: {value}" for path, value in fingerprints)
        raise ValueError(f"incompatible local experiment protocols: {details}")


def validate_berman_protocol(paths):
    fingerprints = []
    for path in paths:
        if not path.exists():
            continue
        data = read(path)
        protocol = data.get("protocol", {})
        fingerprints.append((path, (
            data.get("checkpoint"), data.get("model"), data.get("dataset"),
            protocol.get("split_seed"), protocol.get("val_b"), protocol.get("test"),
            protocol.get("rho"), protocol.get("n_sigma"))))
    if fingerprints and any(value != fingerprints[0][1] for _, value in fingerprints[1:]):
        details = ", ".join(f"{path.name}: {value}" for path, value in fingerprints)
        raise ValueError(f"incompatible Berman protocols: {details}")


def validate_shared_start(paths):
    starts = []
    for path in paths:
        if not path.exists():
            continue
        data = read(path)
        first = data["history"][0]
        validation = data.get("protocol", {}).get("stop_reference",
                                                   first.get("val_loss"))
        starts.append((path, validation, first["test_loss"]))
    if starts and any(
            not math.isclose(validation, starts[0][1], rel_tol=1e-6)
            or not math.isclose(test, starts[0][2], rel_tol=1e-6)
            for _path, validation, test in starts[1:]):
        details = ", ".join(
            f"{path.name}: val={validation}, test={test}"
            for path, validation, test in starts)
        raise ValueError(f"curves do not share the same starting model: {details}")


def draw(output, title, subtitle, specs, xlim=None, ylim=None,
         ylabel="validation loss increase from original", margin_y=5,
         margin_label=r"5% practical margin $\rho$; $2\,\mathrm{SE}$ is candidate-specific",
         y_formatter=None):
    curves, missing = available(specs)
    if not curves:
        raise FileNotFoundError(f"no complete result artifacts available for {output}")
    fig, ax = plt.subplots(figsize=(12, 7))
    values = []
    has_diagnostic = False
    has_rejected = False
    for label, x, y, stop, rejected, color, style in curves:
        end = stop + 1 if stop is not None else len(x)
        ax.plot(x[:end], y[:end], color=color, linestyle=style, linewidth=2.7,
                label=label)
        if end < len(x):
            has_diagnostic = True
            ax.plot(x[stop:], y[stop:], color=color, linestyle=":", linewidth=2,
                    alpha=.55)
        ax.scatter(x[-1], y[-1], color=color, s=55, zorder=5)
        if stop is not None:
            ax.scatter(x[stop], y[stop], facecolors="white", edgecolors=color,
                       linewidths=2.2, s=105, zorder=6)
        if rejected is not None:
            has_rejected = True
            ax.scatter(*rejected, color=color, marker="x", linewidths=2.2,
                       s=85, zorder=6)
        values.extend(y)
    if has_diagnostic:
        ax.plot([], [], color="#777777", linestyle=":", linewidth=2,
                label="post-stop diagnostic")
    if has_rejected:
        ax.scatter([], [], color="#555555", marker="x", linewidths=2.2,
                   s=70, label="first rejected proposal")
    ax.axhline(margin_y, color="#555555", linewidth=1.4, linestyle="--",
               label=margin_label)
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold", pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=12, color="#555555")
    ax.set_xlabel("total parameters removed")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    ax.yaxis.set_major_formatter(y_formatter or
                                 (lambda value, _position: f"{value:.0f}%"))
    ax.grid(axis="y", color="#dddddd", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    ax.set_xlim(*(xlim or (0, ax.get_xlim()[1])))
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif values:
        ax.set_ylim(bottom=min(-1, min(values) - 1))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"{output}: {len(curves)} curves"
          + (f"; waiting for {', '.join(missing)}" if missing else ""))


def main():
    validate_local_protocol([
        LOCAL / "moons_classic.json", LOCAL / "moons_classic_weights.json",
        LOCAL / "moons_fisher_general.json",
        LOCAL / "moons_berman.json",
    ])
    validate_local_protocol([
        LOCAL / "modadd_classic.json", LOCAL / "modadd_fisher.json",
        LOCAL / "modadd_fisher_dual.json", LOCAL / "modadd_fisher_quantized_g002.json",
        LOCAL / "modadd_fisher_quantized_g005.json", LOCAL / "modadd_fisher_quantized_g02.json",
        LOCAL / "modadd_fisher_quantized_post_g02.json",
    ])
    draw(
        PLOTS / "two_moons" / "comparison.png",
        "Two moons: pruning comparison",
        "same checkpoint, splits, and anchored stopping rule",
        (
            ("classic", LOCAL / "moons_classic.json", local_test_curve,
             COLORS["classic"], "-"),
            ("classic + weight zeroing", LOCAL / "moons_classic_weights.json",
             local_test_curve, COLORS["classic"], "--"),
            ("Fisher general", LOCAL / "moons_fisher_general.json", local_test_curve,
             COLORS["quant005"], "-"),
            ("Berman et al. iterative", LOCAL / "moons_berman.json",
             local_test_curve, COLORS["sparsegpt"], "--"),
        ),
        ylabel="held-out test loss relative to original",
        margin_y=1.01,
        margin_label=r"1.01× practical margin $\rho$; $2\,\mathrm{SE}$ uses validation",
        y_formatter=lambda value, _position: f"{value:.2f}×",
        ylim=(.95, 1.45),
    )
    berman_specs = (
        ("classic structural only", BERMAN / "classic.json", berman_loss_curve,
         COLORS["classic"], "--"),
        ("classic + weight zeroing", BERMAN / "classic_weights.json", berman_loss_curve,
         COLORS["classic"], "-"),
        ("Fisher general + validation selection", BERMAN / "fisher_general.json", berman_loss_curve,
         COLORS["dual"], "-"),
        ("Berman et al. iterative", BERMAN / "berman_iterative.json",
         berman_loss_curve, COLORS["sparsegpt"], "--"),
        ("Berman + validation selection", BERMAN / "berman_validated.json",
         berman_loss_curve, COLORS["quant02"], "-"),
        ("Fisher general, score-only", BERMAN / "fisher_general_score.json",
         berman_loss_curve, COLORS["fisher"], "--"),
    )
    validate_berman_protocol([path for _label, path, _loader, _color, _style
                              in berman_specs])
    draw(
        PLOTS / "berman" / "comparison.png",
        "Berman SimpleNN/MNIST: pruning comparison",
        "official checkpoint · identical splits · 2σ stopping · held-out test loss",
        berman_specs,
        xlim=(0, 70), ylim=(.95, 1.20),
        ylabel="held-out test loss relative to original",
        margin_y=1.05,
        margin_label=r"1.05× practical margin $\rho$; $2\,\mathrm{SE}$ uses validation",
        y_formatter=lambda value, _position: f"{value:.2f}×",
    )
    modadd_specs = (
        ("classic structural", LOCAL / "modadd_classic.json", local_test_curve,
         "#999999", "-"),
        ("classic + weight zeroing", MODADD_GRID / "classic_weights.json", berman_loss_curve,
         COLORS["classic"], "-"),
        ("Berman et al. iterative (all 1740 training examples)",
         MODADD_GRID / "berman_iterative_n1740.json", berman_loss_curve, COLORS["sparsegpt"], "--"),
        ("Berman et al. iterative (500 examples)", MODADD_GRID / "berman_iterative_n500.json",
         berman_loss_curve, COLORS["sparsegpt"], ":"),
        ("SparseGPT", MODADD_GRID / "sparsegpt.json",
         sweep_loss_curve, "#8c564b", "--"),
    )
    validate_berman_protocol([
        path for _label, path, _loader, _color, _style in modadd_specs
        if path.parent == MODADD_GRID
    ])
    validate_shared_start([path for _label, path, _loader, _color, _style
                           in modadd_specs])
    draw(
        PLOTS / "modadd" / "comparison.png",
        "Modular addition transformer: pruning comparison",
        "NNpruning modadd checkpoint · identical splits · 2σ stopping · held-out test loss",
        modadd_specs, xlim=(0, 55), ylim=(.95, 1.25),
        ylabel="held-out test loss relative to original",
        margin_y=1.05,
        margin_label=r"1.05× practical margin $\rho$; $2\,\mathrm{SE}$ uses validation",
        y_formatter=lambda value, _position: f"{value:.2f}×",
    )


if __name__ == "__main__":
    main()
