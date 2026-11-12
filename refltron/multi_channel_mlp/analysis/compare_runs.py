"""Compare the overnight sweep runs: training curves, stability, OOD behaviour, and layer health.

Reads ``logs/sweep/<name>.log`` (one per training run) and, when available, the matching
``analysis/runs/<ckpt_id>/best.json`` written by ``diagnostics.py``.

    uv run python -m refltron.multi_channel_mlp.analysis.compare_runs
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from refltron.common.paths import PROJECT_ROOT
from refltron.multi_channel_mlp.analysis.diagnostics import FIG_DPI, INK, MUTED, SERIES, _style

LOG_DIR = PROJECT_ROOT / "logs" / "sweep"
OUT = PROJECT_ROOT / "analysis" / "runs"
# One training-log line per epoch, e.g.
# "Epoch  12/600 | train(hybrid) 3.1e-02 | val 2.9e-02 | thickness MAE 41.2 Å | OOD 0Å->12.0 Å | OOD 2000Å->790 Å"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+) \| train\(\w+\) ([\d.e+-]+) \| val ([\d.e+-]+) \| thickness MAE ([\d.]+) Å \| "
    r"OOD 0Å->([-\d.]+) Å \| OOD 2000Å->([-\d.]+) Å"
)
# The 8-character W&B run id, which is also the checkpoint folder name.
CKPT_RE = re.compile(r"(?:offline-run-\d+_\d+-|saved in checkpoints/)(\w{8})")
ORDER = ["relu-rq4-norm", "relu-log-norm", "gelu-rq4-norm", "gelu-log-norm"]
VERSION = {"relu-rq4-norm": "v1 relu-rq4", "relu-log-norm": "v2 relu-log", "gelu-rq4-norm": "v3 gelu-rq4"}


def parse_log(path: Path):
    """Read one training log: per-epoch numbers, the checkpoint id and the planned epochs."""
    rows = []
    ckpt = None
    planned = None
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match.group(1))
            planned = int(match.group(2))
            row = [epoch]
            for group in range(3, 8):
                row.append(float(match.group(group)))
            rows.append(row)
        ckpt_match = CKPT_RE.search(line)
        if ckpt_match and ckpt is None:
            ckpt = ckpt_match.group(1)
    if rows:
        arr = np.array(rows)
    else:
        arr = np.zeros((0, 6))
    return {"epoch": arr[:, 0], "train": arr[:, 1], "val": arr[:, 2], "mae": arr[:, 3],
            "ood0": arr[:, 4], "ood2000": arr[:, 5], "ckpt": ckpt, "planned": planned}


def jitter(x, window=100):
    """Median relative epoch-to-epoch change over the last ``window`` epochs."""
    x = x[-window:]
    if len(x) < 3:
        return float("nan")
    return float(np.median(np.abs(np.diff(x)) / np.abs(x[1:])))


def main():
    # Read every sweep log (diagnostics.log is not a training run).
    logs = {}
    for path in sorted(LOG_DIR.glob("*.log")):
        if path.stem != "diagnostics":
            logs[path.stem] = parse_log(path)

    # Known runs first, in ORDER, then any others.
    names = []
    for name in ORDER:
        if name in logs:
            names.append(name)
    for name in logs:
        if name not in ORDER:
            names.append(name)
    if not names:
        print("no sweep logs found")
        return

    colors = {}
    for i in range(len(names)):
        colors[names[i]] = SERIES[i % len(SERIES)]

    # Layer diagnostics written by diagnostics.py, where available.
    diag = {}
    for name in names:
        ckpt = logs[name]["ckpt"]
        json_path = OUT / str(ckpt) / "best.json"
        if ckpt and json_path.exists():
            diag[name] = json.loads(json_path.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), layout="constrained")
    for name in names:
        log = logs[name]
        color = colors[name]
        if not len(log["epoch"]):
            continue
        axes[0, 0].plot(log["epoch"], log["val"], color=color, lw=1.8, label=VERSION.get(name, name))
        axes[0, 0].plot(log["epoch"], log["train"], color=color, lw=1, ls=":", alpha=0.8)
        axes[0, 1].plot(log["epoch"], log["mae"], color=color, lw=1.8, label=name)
        axes[0, 2].plot(log["epoch"], log["val"] - log["train"], color=color, lw=1.5, label=name)
        axes[1, 0].plot(log["epoch"], log["ood0"], color=color, lw=1.2, alpha=0.9, label=name)
        axes[1, 1].plot(log["epoch"], log["ood2000"], color=color, lw=1.2, alpha=0.9, label=name)
    axes[0, 0].set_yscale("log")
    _style(axes[0, 0], "Hybrid loss (solid = val, dotted = train)")
    _style(axes[0, 1], "Validation thickness MAE [Å]")
    axes[0, 2].axhline(0, color=MUTED, lw=1, ls=":")
    _style(axes[0, 2], "Generalization gap (val − train)")
    axes[1, 0].axhline(0, color=MUTED, lw=1, ls=":")
    _style(axes[1, 0], "OOD bare oxide: predicted film thickness [Å] (truth 0)")
    axes[1, 1].axhline(2000, color=MUTED, lw=1, ls=":")
    axes[1, 1].axhline(800, color=MUTED, lw=1, ls="--")
    _style(axes[1, 1], "OOD 2000 Å film: predicted thickness [Å]")
    axes[1, 1].text(600, 770, "training max 800 Å", color=MUTED, fontsize=10, ha="right", va="top")
    axes[1, 1].text(600, 1975, "truth 2000 Å", color=MUTED, fontsize=10, ha="right", va="top")
    for ax in axes.flat[:5]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=11, frameon=False)

    # Panel 6: decoder R^2 for fringe detail per stage, from the morph analysis.
    ax = axes[1, 2]
    for name in names:
        if name not in diag:
            continue
        morph = diag[name].get("morph")
        if morph:
            xs = np.arange(len(morph["stages"]))
            ax.plot(xs, morph["fringe_r2"], "-o", color=colors[name], lw=1.8, ms=4, label=f"{name} fringes")
            ax.plot(xs, morph["curve_r2"], "--", color=colors[name], lw=1.2, alpha=0.8)
            ax.set_xticks(xs, morph["stages"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("decoder R²")
    ax.text(0.98, 0.97, "solid = fringe detail\ndashed = whole curve", transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=INK)
    _style(ax, "Curve information kept per layer (best.pt)")

    fig.suptitle("Refltron (variable-q perceptron) versions — v1 ReLU·rq⁴, v2 ReLU·log10, v3 GELU·rq⁴ (resumed at epoch 143)", fontsize=15.5, color=INK)
    fig.savefig(OUT / "sweep_comparison.png", dpi=FIG_DPI)
    plt.close(fig)

    # Markdown summary table, one row per run.
    lines = ["# Overnight sweep comparison", "",
             "All runs: 600 epochs, hybrid loss (Si-aligned depth profile + label MSE), Adam 5e-4 with cosine decay to 5e-6, "
             "per-channel input standardization, batch 1024, seed 42.", "",
             "![sweep](sweep_comparison.png)", "",
             "| run | ckpt | epochs | best val (epoch) | final val | final train | thick. MAE Å | val jitter (last 100) | OOD 0Å → (last 50 mean ± sd) | OOD 2000Å → | val label MSE | Si-aligned profile L2 | worst dead % | fringe R² after L1 / L2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        log = logs[name]
        if not len(log["epoch"]):
            lines.append(f"| {name} | {log['ckpt']} | 0 | – | – | – | – | – | – | – | – | – | – | – |")
            continue
        best = int(np.argmin(log["val"]))
        d = diag.get(name)

        # Worst dead fraction over all layers, and fringe R^2 after the first two layers.
        dead = "–"
        fringes = "–"
        val_metrics = None
        if d:
            val_metrics = d["val_metrics"]
            worst_dead = 0.0
            for i in range(len(d["layers"])):
                worst_dead = max(worst_dead, 100 * d["layers"][i]["dead_frac"])
            dead = f"{worst_dead:.0f}"
            if d.get("morph"):
                stages = d["morph"]["stages"]
                parts = []
                for i in range(len(stages)):
                    if stages[i].startswith("L1 ") or stages[i].startswith("L2 "):
                        parts.append(f"{d['morph']['fringe_r2'][i]:.2f}")
                fringes = " / ".join(parts)

        if val_metrics:
            metric_cells = f"{val_metrics['label_mse']:.2e} | {val_metrics['profile_l2']:.4f} | "
        else:
            metric_cells = "– | – | "
        lines.append(
            f"| {name} | {log['ckpt']} | {int(log['epoch'][-1])}/{log['planned']} | {log['val'][best]:.4f} ({int(log['epoch'][best])}) | "
            f"{log['val'][-1]:.4f} | {log['train'][-1]:.4f} | {log['mae'][-1]:.1f} | {100*jitter(log['val']):.2f}% | "
            f"{log['ood0'][-50:].mean():.0f} ± {log['ood0'][-50:].std():.0f} | {log['ood2000'][-50:].mean():.0f} ± {log['ood2000'][-50:].std():.0f} | "
            + metric_cells
            + f"{dead} | {fringes} |"
        )
    lines += ["", "Per-run figures: `analysis/runs/<ckpt>/best_morph.png`, `best_cards/`, `best_layer_health.png`, "
              "`best_saliency.png`, `evolution.png`. Flags per run: `analysis/runs/SUMMARY.md`."]
    (OUT / "SWEEP.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'SWEEP.md'} and {OUT / 'sweep_comparison.png'}")


if __name__ == "__main__":
    main()
