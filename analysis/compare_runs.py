"""Compare the overnight sweep runs: training curves, stability, OOD behaviour, and layer health.

Reads ``logs/sweep/<name>.log`` (one per training run) and, when available, the matching
``analysis/runs/<ckpt_id>/best.json`` written by ``diagnostics.py``.

    uv run python analysis/compare_runs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
from diagnostics import FIG_DPI, INK, MUTED, SERIES, _style  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs" / "sweep"
OUT = PROJECT_ROOT / "analysis" / "runs"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+) \| train\(\w+\) ([\d.e+-]+) \| val ([\d.e+-]+) \| thickness MAE ([\d.]+) Å \| "
    r"OOD 0Å->([-\d.]+) Å \| OOD 2000Å->([-\d.]+) Å"
)
CKPT_RE = re.compile(r"(?:offline-run-\d+_\d+-|saved in checkpoints/)(\w{8})")
ORDER = ["relu-rq4-norm", "relu-log-norm", "gelu-rq4-norm", "gelu-log-norm"]
VERSION = {"relu-rq4-norm": "v1 relu-rq4", "relu-log-norm": "v2 relu-log", "gelu-rq4-norm": "v3 gelu-rq4 (e146)"}


def parse_log(path: Path):
    rows, ckpt, planned = [], None, None
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            e, planned = int(m.group(1)), int(m.group(2))
            rows.append([e] + [float(m.group(i)) for i in range(3, 8)])
        c = CKPT_RE.search(line)
        if c and ckpt is None:
            ckpt = c.group(1)
    arr = np.array(rows) if rows else np.zeros((0, 6))
    return {"epoch": arr[:, 0], "train": arr[:, 1], "val": arr[:, 2], "mae": arr[:, 3],
            "ood0": arr[:, 4], "ood2000": arr[:, 5], "ckpt": ckpt, "planned": planned}


def jitter(x, window=100):
    """Median relative epoch-to-epoch change over the last ``window`` epochs."""
    x = x[-window:]
    if len(x) < 3:
        return float("nan")
    return float(np.median(np.abs(np.diff(x)) / np.abs(x[1:])))


def main():
    logs = {p.stem: parse_log(p) for p in sorted(LOG_DIR.glob("*.log")) if p.stem != "diagnostics"}
    names = [n for n in ORDER if n in logs] + [n for n in logs if n not in ORDER]
    if not names:
        print("no sweep logs found")
        return
    colors = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}
    diag = {}
    for n in names:
        ck = logs[n]["ckpt"]
        j = OUT / str(ck) / "best.json"
        if ck and j.exists():
            diag[n] = json.loads(j.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), layout="constrained")
    for n in names:
        L, c = logs[n], colors[n]
        if not len(L["epoch"]):
            continue
        axes[0, 0].plot(L["epoch"], L["val"], color=c, lw=1.8, label=VERSION.get(n, n))
        axes[0, 0].plot(L["epoch"], L["train"], color=c, lw=1, ls=":", alpha=0.8)
        axes[0, 1].plot(L["epoch"], L["mae"], color=c, lw=1.8, label=n)
        axes[0, 2].plot(L["epoch"], L["val"] - L["train"], color=c, lw=1.5, label=n)
        axes[1, 0].plot(L["epoch"], L["ood0"], color=c, lw=1.2, alpha=0.9, label=n)
        axes[1, 1].plot(L["epoch"], L["ood2000"], color=c, lw=1.2, alpha=0.9, label=n)
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
    for n in names:
        mo = diag.get(n, {}).get("morph")
        if mo:
            xs = np.arange(len(mo["stages"]))
            ax.plot(xs, mo["fringe_r2"], "-o", color=colors[n], lw=1.8, ms=4, label=f"{n} fringes")
            ax.plot(xs, mo["curve_r2"], "--", color=colors[n], lw=1.2, alpha=0.8)
            ax.set_xticks(xs, mo["stages"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("decoder R²")
    ax.text(0.98, 0.97, "solid = fringe detail\ndashed = whole curve", transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=INK)
    _style(ax, "Curve information kept per layer (best.pt)")

    fig.suptitle("Refltron (dynamic-q perceptron) versions — v1 ReLU·rq⁴, v2 ReLU·log10, v3 GELU·rq⁴ (stopped at epoch 146)", fontsize=15.5, color=INK)
    fig.savefig(OUT / "sweep_comparison.png", dpi=FIG_DPI)
    plt.close(fig)

    lines = ["# Overnight sweep comparison", "",
             "All runs: 600 epochs, hybrid loss (Si-aligned depth profile + label MSE), Adam 5e-4 with cosine decay to 5e-6, "
             "per-channel input standardization, batch 1024, seed 42.", "",
             "![sweep](sweep_comparison.png)", "",
             "| run | ckpt | epochs | best val (epoch) | final val | final train | thick. MAE Å | val jitter (last 100) | OOD 0Å → (last 50 mean ± sd) | OOD 2000Å → | val label MSE | Si-aligned profile L2 | worst dead % | fringe R² after L1 / L2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        L = logs[n]
        if not len(L["epoch"]):
            lines.append(f"| {n} | {L['ckpt']} | 0 | – | – | – | – | – | – | – | – | – | – | – |")
            continue
        b = int(np.argmin(L["val"]))
        d = diag.get(n)
        dm = d["val_metrics"] if d else None
        dead = f"{max(100*l['dead_frac'] for l in d['layers']):.0f}" if d else "–"
        fr = "–"
        if d and d.get("morph"):
            st = d["morph"]["stages"]
            fr = " / ".join(f"{d['morph']['fringe_r2'][i]:.2f}" for i, s in enumerate(st) if s.startswith(("L1 ", "L2 ")))
        lines.append(
            f"| {n} | {L['ckpt']} | {int(L['epoch'][-1])}/{L['planned']} | {L['val'][b]:.4f} ({int(L['epoch'][b])}) | "
            f"{L['val'][-1]:.4f} | {L['train'][-1]:.4f} | {L['mae'][-1]:.1f} | {100*jitter(L['val']):.2f}% | "
            f"{L['ood0'][-50:].mean():.0f} ± {L['ood0'][-50:].std():.0f} | {L['ood2000'][-50:].mean():.0f} ± {L['ood2000'][-50:].std():.0f} | "
            + (f"{dm['label_mse']:.2e} | {dm['profile_l2']:.4f} | " if dm else "– | – | ")
            + f"{dead} | {fr} |"
        )
    lines += ["", "Per-run figures: `analysis/runs/<ckpt>/best_morph.png`, `best_cards/`, `best_layer_health.png`, "
              "`best_saliency.png`, `evolution.png`. Flags per run: `analysis/runs/SUMMARY.md`."]
    (OUT / "SWEEP.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'SWEEP.md'} and {OUT / 'sweep_comparison.png'}")


if __name__ == "__main__":
    main()
