"""Add the single-channel fixed-q models to the Refltron model card on the Hugging Face Hub.

    uv run --with safetensors python -m refltron.single_channel_mlp.publish_hf --dry-run   # figures + stage only
    uv run --with safetensors python -m refltron.single_channel_mlp.publish_hf             # stage + upload
    uv run --with safetensors python -m refltron.single_channel_mlp.publish_hf --help      # list every option

The Refltron (multi-channel) part of the ``main`` card is left exactly as it is on the
Hub. This script only adds a section at the bottom, between two marker comments, and
replaces that section on every later run.

What happens:
    1. For each run in RUNS: read its best checkpoint, its final validation errors
       (metrics.json) and its per-epoch log (the W&B console log), then predict every
       validation row with the best weights.
    2. Draw the card figures into FIG_DIR: the network inputs, the training curves and
       the prediction quality.
    3. Stage the files under STAGE: ``main`` gets the current Hub card with the new section
       and the figures; each run gets its own branch with weights, config and a short card.
    4. Unless --dry-run, upload ``main`` and one branch per run.
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

# Save figures directly to PNG files without opening an interactive window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from refltron.common import physics
from refltron.common import plot_style as ps
from refltron.common.loss import LABEL_NAMES, label_range
from refltron.common.paths import PROJECT_ROOT
from refltron.single_channel_mlp import train
from refltron.single_channel_mlp.network import OldNet

ps.apply()  # shared figure style (colours, fonts, hairline grid)

# =====================================================================================
# DEFAULT SETTINGS: edit these, or override the run options from the command line.
# =====================================================================================

REPO = "carbon-lab/refltron"     # model repository on the Hugging Face Hub
DRY_RUN = False                  # True = only draw figures and stage files, do not upload

ROOT = PROJECT_ROOT
STAGE = Path("/tmp/hf_stage_single")             # where the files are prepared before upload
FIG_DIR = ROOT / "analysis/runs/single_channel"  # where the card figures are written

# input, checkpoint id (= W&B run id), branch, short description
RUNS = [
    ("s", "c0r4pvqk", "single-fixed-q-s", "log R_s → 5 labels, isotropic films"),
    ("p", "1l8dvwsl", "single-fixed-q-p", "log R_p → 7 labels, uniaxial films"),
    ("anisotropy", "fk4d1wk5", "single-fixed-q-anisotropy", "anisotropy ratio A → 7 labels, uniaxial films"),
]
BEST_INPUT = "anisotropy"        # the run the card recommends

# The card section sits between these two lines, so a later run can replace it.
SECTION_START = "<!-- single-channel-fixed-q:start -->"
SECTION_END = "<!-- single-channel-fixed-q:end -->"

# One line of the training log, for example:
#   Epoch  12/60 | train(mse) 6.41e-03 | val 7.35e-03 | thickness MAE 18.4 Å | profile L2 6.24e-02 | dead outputs 0
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+ \| train\(\w+\) ([\d.e+-]+) \| val ([\d.e+-]+) \| "
                      r"thickness MAE ([\d.]+) Å \| profile L2 ([\d.e+-]+) \| dead outputs (\d+)")

# Colour and name of each input, used by every figure.
INPUT_COLORS = {"s": ps.SLOTS[0], "p": ps.SLOTS[1], "anisotropy": ps.SLOTS[2]}
INPUT_NAMES = {"s": "S  (log R_s)", "p": "P  (log R_p)", "anisotropy": "A  (anisotropy)"}
INPUT_AXIS_LABELS = {
    "s": r"$\ln R_s$",
    "p": r"$\ln R_p$",
    "anisotropy": r"$A = (P - S c)\,/\,(P + S c)$",
}
# How each label is written in figures and tables, and the unit of its error.
LABEL_TEXT = {
    "film_thickness": "film thickness",
    "film_roughness": "film roughness",
    "oxide_thickness": "oxide thickness",
    "delta_xx": "δ_xx",
    "delta_zz": "δ_zz",
    "beta_xx": "β_xx",
    "beta_zz": "β_zz",
}
LABEL_TICKS = {
    "film_thickness": "film\nthickness",
    "film_roughness": "film\nroughness",
    "oxide_thickness": "oxide\nthickness",
    "delta_xx": r"$\delta_{xx}$",
    "delta_zz": r"$\delta_{zz}$",
    "beta_xx": r"$\beta_{xx}$",
    "beta_zz": r"$\beta_{zz}$",
}


def main():
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read every run and predict its validation rows.
    summaries = []
    for run in RUNS:
        summaries.append(summarize(run[0], run[1], run[2], run[3]))

    # 2. Figures.
    plot_inputs(summaries, FIG_DIR / "single_channel_inputs.png")
    plot_training(summaries, FIG_DIR / "single_channel_training.png")
    plot_quality(summaries, FIG_DIR / "single_channel_quality.png")
    section = card_section(summaries, args.repo)
    (FIG_DIR / "card_section.md").write_text(section)
    print(f"figures and card section written to {FIG_DIR}")

    # 3. Stage main and one branch per run.
    main_dir = stage_main(section, args.repo)
    run_dirs = []
    for summary in summaries:
        run_dirs.append(stage_run(summary, args.repo))
    print(f"staged under {STAGE}")
    if args.dry_run:
        return

    # 4. Upload main, then one branch per run.
    api = HfApi()
    api.upload_folder(repo_id=args.repo, folder_path=main_dir,
                      commit_message="Add the single-channel fixed-q baselines to the card")
    for i in range(len(summaries)):
        summary = summaries[i]
        api.create_branch(args.repo, branch=summary["branch"], exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=run_dirs[i], revision=summary["branch"],
                          commit_message=f"{summary['branch']}: {summary['desc']}")
        print(f"uploaded {summary['branch']}")
    print(f"https://huggingface.co/{args.repo}")


# =====================================================================================
# Reading the results of one run
# =====================================================================================


def find_log(ckpt_id):
    """The W&B console log (output.log) of a run."""
    matches = sorted((ROOT / "wandb").glob(f"offline-run-*-{ckpt_id}/files/output.log"))
    if len(matches) == 0:
        raise FileNotFoundError(f"no W&B log for run {ckpt_id} under {ROOT / 'wandb'}")
    return matches[-1]


def parse_log(path):
    """Read every epoch line of a training log into a list of dicts."""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match is None:
            continue
        groups = match.groups()
        rows.append({
            "epoch": int(groups[0]),
            "train": float(groups[1]),
            "val": float(groups[2]),
            "mae": float(groups[3]),
            "profile_l2": float(groups[4]),
            "dead": int(groups[5]),
        })
    return rows


@torch.no_grad()
def summarize(input_name, ckpt_id, branch, desc):
    """Collect everything the card needs about one run into a dict."""
    run_dir = ROOT / "checkpoints" / ckpt_id
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    log_path = find_log(ckpt_id)
    label_names = checkpoint["label_names"]

    # Predict every validation row with the best weights (the same split as train.py).
    dataset, q_grid = train.load_dataset(input_name)
    _, val_set = train.split_dataset(dataset, checkpoint["config"]["val_split"])
    device = train.select_device("auto")
    model = OldNet(n_q=len(q_grid), n_labels=len(label_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    predictions = []
    targets = []
    for x, y in DataLoader(val_set, batch_size=8192):
        predictions.append(model(x.to(device)).cpu())
        targets.append(y)

    # Back from [0, 1] to physical units.
    label_min, label_max = label_range(label_names)
    scale = label_max - label_min
    predicted = torch.cat(predictions).numpy() * scale + label_min
    true = torch.cat(targets).numpy() * scale + label_min

    # One example input curve: the first validation row.
    example_x, _ = val_set[0]

    return {
        "input": input_name,
        "ckpt_id": ckpt_id,
        "branch": branch,
        "desc": desc,
        "config": checkpoint["config"],
        "label_names": label_names,
        "best_epoch": checkpoint["epoch"],
        "best_val": checkpoint["val_loss"],
        "metrics": metrics,
        "log": parse_log(log_path),
        "log_path": log_path,
        "q_grid": q_grid,
        "example_x": example_x.numpy(),
        "example_true": true[0],
        "example_pred": predicted[0],
        "predicted": predicted,
        "true": true,
        "n_val": len(val_set),
    }


# =====================================================================================
# Figures
# =====================================================================================


def theta_degrees(q_grid):
    """Grazing angle [deg] of each q at the fixed-q energy (250 eV)."""
    wavelength = physics.HC_EV_ANGSTROM / physics.FIXED_Q_ENERGY_EV
    return np.degrees(np.arcsin(q_grid * wavelength / (4.0 * math.pi)))


def q_from_theta(theta_deg):
    """q [1/Å] of a grazing angle [deg] at the fixed-q energy (250 eV)."""
    wavelength = physics.HC_EV_ANGSTROM / physics.FIXED_Q_ENERGY_EV
    return 4.0 * math.pi / wavelength * np.sin(np.radians(theta_deg))


def plot_inputs(summaries, path):
    """One panel per model: the example input vector, one marker per input neuron."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), layout="constrained")
    for i in range(len(summaries)):
        s = summaries[i]
        ax = axes[i]
        q = s["q_grid"]
        color = INPUT_COLORS[s["input"]]
        ax.plot(q, s["example_x"], color=color, lw=0.8, alpha=0.45)
        ax.plot(q, s["example_x"], color=color, ls="", marker="o", ms=2.6)

        if s["input"] == "anisotropy":
            # A stays between -1 and +1 and swings near the Brewster angle (theta = 45 deg).
            ax.axhline(-1.0, color=ps.AXIS, lw=1.0)
            ax.axhline(1.0, color=ps.AXIS, lw=1.0)
            q_brewster = q_from_theta(45.0)
            ax.axvline(q_brewster, color=ps.MUTED, ls=(0, (4, 3)), lw=1.0)
            ax.text(q_brewster, 0.04, "θ = 45° ", color=ps.MUTED, fontsize=10, ha="right",
                    transform=ax.get_xaxis_transform())

        # A second x axis on top: the grazing angle of each q at 250 eV.
        top = ax.secondary_xaxis("top", functions=(theta_from_q, q_from_theta))
        top.set_xlabel("grazing angle θ at 250 eV  [°]", color=ps.MUTED)
        ax.set_xlim(0, q[-1] * 1.03)
        ax.set_xlabel(r"$q$  [$\mathrm{\AA}^{-1}$]")
        ax.set_ylabel(INPUT_AXIS_LABELS[s["input"]])
        thickness_true = s["example_true"][0]
        thickness_pred = s["example_pred"][0]
        ax.set_title(f"{INPUT_NAMES[s['input']]} · film {thickness_true:.0f} Å, predicted {thickness_pred:.0f} Å",
                     fontsize=12.5)

    n_q = len(summaries[0]["q_grid"])
    q = summaries[0]["q_grid"]
    fig.suptitle(
        f"What each single-channel model sees: {n_q} numbers, one per marker, always at the same q "
        f"({q[0]:.4f}–{q[-1]:.3f} Å⁻¹, linear) · first validation row of each model",
        fontsize=14, x=0.008, ha="left", color=ps.INK,
    )
    fig.savefig(path, dpi=ps.FIG_DPI)
    plt.close(fig)


def theta_from_q(q):
    """Grazing angle [deg] of q at the fixed-q energy, safe for the axis helper (clips q)."""
    wavelength = physics.HC_EV_ANGSTROM / physics.FIXED_Q_ENERGY_EV
    sine = np.clip(np.asarray(q) * wavelength / (4.0 * math.pi), -1.0, 1.0)
    return np.degrees(np.arcsin(sine))


def plot_training(summaries, path):
    """Validation loss, film-thickness MAE and depth-profile L2 per epoch for every model."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), layout="constrained")
    panels = [
        ("val", "Validation loss (MSE of the scaled labels)", "MSE", True),
        ("mae", "Film thickness MAE on validation rows", "MAE  [Å]", False),
        ("profile_l2", "Depth-profile L2 on validation rows", "L2", True),
    ]
    for p in range(len(panels)):
        key = panels[p][0]
        ax = axes[p]
        end_values = []
        end_names = []
        for s in summaries:
            epochs = []
            values = []
            train_values = []
            for row in s["log"]:
                epochs.append(row["epoch"])
                values.append(row[key])
                train_values.append(row["train"])
            color = INPUT_COLORS[s["input"]]
            ax.plot(epochs, values, color=color, lw=2.0, label=INPUT_NAMES[s["input"]])
            if key == "val":
                # Training loss as a thin dashed line of the same colour.
                ax.plot(epochs, train_values, color=color, lw=1.0, ls=(0, (4, 3)), alpha=0.8)
            end_values.append(values[-1])
            end_names.append(s["input"])
        if panels[p][3]:
            ax.set_yscale("log")
        # Direct label at the end of each line, in ink colour, moved apart so they do not overlap.
        label_heights = spread_labels(end_values, ax.get_ylim(), panels[p][3])
        last_epoch = summaries[0]["log"][-1]["epoch"]
        for i in range(len(end_names)):
            ax.annotate(f" {end_names[i]}", (last_epoch, label_heights[i]), color=ps.INK_2, fontsize=11,
                        va="center", annotation_clip=False)
        ax.set_xlabel("epoch")
        ax.set_ylabel(panels[p][2])
        ax.set_title(panels[p][1], fontsize=12.5)
        ax.set_xlim(0, summaries[0]["log"][-1]["epoch"] + 5)
    axes[0].plot([], [], color=ps.MUTED, lw=1.0, ls=(0, (4, 3)), label="training loss (dashed)")
    axes[0].legend(loc="upper right", fontsize=10.5)
    fig.suptitle(
        "Single-channel fixed-q models · 60 epochs each · S is trained on isotropic films with 5 labels, "
        "P and A on uniaxial films with 7",
        fontsize=14, x=0.008, ha="left", color=ps.INK,
    )
    fig.savefig(path, dpi=ps.FIG_DPI)
    plt.close(fig)


def spread_labels(values, y_limits, log_scale, min_gap=0.06):
    """Heights for end-of-line labels, pushed apart by at least ``min_gap`` of the axis height.

    On a log axis the gap is measured in log10(value).
    """
    positions = []
    for v in values:
        if log_scale:
            positions.append(math.log10(v))
        else:
            positions.append(v)
    if log_scale:
        axis_height = math.log10(y_limits[1]) - math.log10(y_limits[0])
    else:
        axis_height = y_limits[1] - y_limits[0]
    gap = min_gap * axis_height

    # Walk the labels from lowest to highest and lift any that sit too close to the one below.
    order = np.argsort(positions)
    for k in range(1, len(order)):
        below = positions[order[k - 1]]
        if positions[order[k]] - below < gap:
            positions[order[k]] = below + gap

    heights = []
    for position in positions:
        if log_scale:
            heights.append(10.0 ** position)
        else:
            heights.append(position)
    return heights


def plot_quality(summaries, path):
    """Top: validation R² of every label and model. Bottom: predicted vs true film thickness."""
    fig = plt.figure(figsize=(18.0, 10.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[0.6, 1.0])

    # --- R² per label, one bar per model ---
    ax_r2 = fig.add_subplot(grid[0, :])
    x = np.arange(len(LABEL_NAMES))
    width = 0.26
    for m in range(len(summaries)):
        s = summaries[m]
        offset = (m - 1) * width
        for i in range(len(LABEL_NAMES)):
            name = LABEL_NAMES[i]
            if name not in s["metrics"]["labels"]:
                # The S model has no zz labels (isotropic films).
                ax_r2.text(x[i] + offset, 0.62, "n/a", ha="center", va="bottom", fontsize=9, color=ps.MUTED,
                           rotation=90)
                continue
            r2 = s["metrics"]["labels"][name]["r2"]
            ax_r2.bar(x[i] + offset, r2, width=width, color=INPUT_COLORS[s["input"]],
                      edgecolor=ps.SURFACE, linewidth=1.5, zorder=2)
            ax_r2.text(x[i] + offset, r2 + 0.006, f"{r2:.3f}", ha="center", va="bottom", fontsize=8.5,
                       color=ps.INK_2, rotation=90)
    ticks = []
    for name in LABEL_NAMES:
        ticks.append(LABEL_TICKS[name])
    ax_r2.set_xticks(x, ticks, fontsize=11)
    ax_r2.tick_params(axis="x", length=0)
    ax_r2.set_ylim(0.6, 1.13)
    ax_r2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_r2.set_ylabel("R²  (1 = perfect)")
    ax_r2.set_title("Validation R² of every label (best checkpoint, all 200,000 validation rows)", fontsize=12.5)
    ax_r2.grid(axis="x", visible=False)
    legend_handles = []
    for s in summaries:
        legend_handles.append(Patch(color=INPUT_COLORS[s["input"]], label=INPUT_NAMES[s["input"]]))
    ax_r2.legend(handles=legend_handles, loc="upper right", ncol=3, fontsize=10.5)

    # --- predicted vs true film thickness ---
    for m in range(len(summaries)):
        s = summaries[m]
        ax = fig.add_subplot(grid[1, m])
        true_t = s["true"][:, 0]
        pred_t = s["predicted"][:, 0]
        ax.hexbin(true_t, pred_t, gridsize=90, bins="log", cmap=single_hue_map(INPUT_COLORS[s["input"]]),
                  extent=(0, 850, 0, 850), mincnt=1, linewidths=0)
        ax.plot([0, 850], [0, 850], color=ps.INK_2, lw=1.0, ls=(0, (4, 3)))
        mae = s["metrics"]["labels"]["film_thickness"]["mae"]
        r2 = s["metrics"]["labels"]["film_thickness"]["r2"]
        ax.set_xlim(0, 850)
        ax.set_ylim(0, 850)
        ax.set_aspect("equal")
        ax.set_xlabel("true film thickness  [Å]")
        ax.set_ylabel("predicted film thickness  [Å]")
        ax.set_title(f"{INPUT_NAMES[s['input']]} · MAE {mae:.1f} Å · R² {r2:.3f}", fontsize=12.5)
    fig.suptitle("Single-channel fixed-q models: prediction quality on held-out curves",
                 fontsize=14, x=0.008, ha="left", color=ps.INK)
    fig.savefig(path, dpi=ps.FIG_DPI)
    plt.close(fig)


def single_hue_map(color):
    """A light-to-dark colour map of one hue, for the density plots."""
    return LinearSegmentedColormap.from_list("one_hue", ["#ecebe6", color, ps.INK])


# =====================================================================================
# Text of the model cards
# =====================================================================================


def label_error_text(summary, name):
    """'MAE (R²)' of one label, or '–' if the model does not predict it."""
    labels = summary["metrics"]["labels"]
    if name not in labels:
        return "–"
    mae = labels[name]["mae"]
    r2 = labels[name]["r2"]
    if name.startswith("delta") or name.startswith("beta"):
        return f"{mae:.2e} ({r2:.3f})"
    return f"{mae:.2f} Å ({r2:.3f})"


def card_section(summaries, repo):
    """The Markdown section added to the bottom of the main card."""
    best = None
    for s in summaries:
        if s["input"] == BEST_INPUT:
            best = s
    q = summaries[0]["q_grid"]
    theta = theta_degrees(q)

    header = "| model | data | labels | best epoch | val MSE ↓ |"
    rule = "|---|---|---|---|---|"
    for name in LABEL_NAMES:
        header += f" {LABEL_TEXT[name]} MAE (R²) |"
        rule += "---|"
    rows = [header, rule]
    for s in summaries:
        link = f"[`{s['branch']}`](https://huggingface.co/{repo}/tree/{s['branch']})"
        if s["input"] == "s":
            data = "fixed-q-isotropic"
        else:
            data = "fixed-q"
        val_text = f"{s['best_val']:.2e}"
        if s["input"] == BEST_INPUT:
            val_text = f"**{val_text}**"
        row = f"| {link} | {data} | {len(s['label_names'])} | {s['best_epoch']} | {val_text} |"
        for name in LABEL_NAMES:
            row += f" {label_error_text(s, name)} |"
        rows.append(row)
    table = "\n".join(rows)

    p_run = None
    for s in summaries:
        if s["input"] == "p":
            p_run = s
    a_labels = best["metrics"]["labels"]
    p_labels = p_run["metrics"]["labels"]
    better = []
    for name in LABEL_NAMES:
        if a_labels[name]["r2"] > p_labels[name]["r2"]:
            better.append(LABEL_TEXT[name])

    return f"""{SECTION_START}

## Single-channel fixed-q baselines

Alongside Refltron, three small baselines reproduce the original fixed-q Keras network
(Greco et al., `code-old`) in PyTorch (`OldNet`, see
[`network.py` on any `single-fixed-q-*` branch](https://huggingface.co/{repo}/blob/{best['branch']}/network.py)).
They are much simpler than Refltron and only work on **one fixed q grid**:

- **Input:** one curve of {len(q)} numbers. Every curve is sampled at the same {len(q)} q values
  ({q[0]:.4f}–{q[-1]:.3f} Å⁻¹, linear; θ {theta[0]:.2f}–{theta[-1]:.2f}° at {physics.FIXED_Q_ENERGY_EV:.0f} eV),
  so q itself is never passed in. A value's position in the vector says which q it belongs to.
  The models cannot read curves at any other energy or q grid.
- **Network:** Linear {len(q)}→400→800→400→300→200→100→labels, ReLU after every layer, MSE on labels
  scaled to [0, 1], Adam 5e-4, batch 128, 60 epochs, last 20% of rows held out (200,000 curves).
- **Data:** the `fixed-q` (uniaxial) and `fixed-q-isotropic` branches of
  [`carbon-lab/reflectivity-1m`](https://huggingface.co/datasets/carbon-lab/reflectivity-1m/tree/fixed-q).

| input | what the network gets |
|---|---|
| `s` | ln R_s, on isotropic films (δ_xx = δ_zz, β_xx = β_zz), so 5 labels |
| `p` | ln R_p, on uniaxial films, 7 labels |
| `anisotropy` | A = (P − S·c)/(P + S·c) with c = cos²2θ, on uniaxial films, 7 labels. For n ≈ 1, r_p ≈ r_s·cos 2θ, so A removes the purely geometric part of P and keeps the anisotropy. It stays within ±1 and swings near θ = 45° (the Brewster angle). |

**Use `{best['branch']}`.** On the 7-label uniaxial task it reaches about half the validation loss of `p`
({best['best_val']:.2e} vs {p_run['best_val']:.2e}) and has a higher R² for {", ".join(better)}.
`p` is only slightly better on the remaining labels. `s` has a lower loss still, but it solves an easier
problem (isotropic films, 5 labels), so its numbers are not directly comparable.

```bash
hf download {repo} --revision {best['branch']} --local-dir refltron-single-anisotropy
```

Validation errors of the best checkpoint on all held-out curves, as MAE (R²). Lengths are in Å.

{table}

### How the models see the fixed q grid

Each marker is one input neuron. It always holds the value at the same q, for every curve.

![single-channel inputs](figures/single_channel_inputs.png)

### Training

![single-channel training curves](figures/single_channel_training.png)

### Quality

![single-channel prediction quality](figures/single_channel_quality.png)

Each branch also holds a training snapshot (`figures/snapshot_best.png`: one training and one
validation curve, their depth profiles and labels), the weights (`model.safetensors`, `best.pt`),
`config.json`, `metrics.json` and the training log.

{SECTION_END}
"""


def run_card(s, repo):
    """README of one run's branch."""
    lines = []
    for name in s["label_names"]:
        lines.append(f"| {LABEL_TEXT[name]} | {label_error_text(s, name)} |")
    label_table = "\n".join(lines)
    return f"""---
library_name: pytorch
tags: [refltron, x-ray-reflectivity, inverse-problem, mlp, fixed-q]
datasets: [carbon-lab/reflectivity-1m]
---

# Refltron · single-channel fixed-q · `{s['input']}`

Input: {s['desc']}. The original fixed-q Keras network (`OldNet`, `network.py`) trained for
{s['config']['epochs']} epochs; best checkpoint at epoch {s['best_epoch']} (validation MSE {s['best_val']:.2e}).
The input is a curve on the fixed grid of {s['config']['n_q']} q values,
{s['config']['q_range'][0]:.4f}–{s['config']['q_range'][1]:.3f} Å⁻¹ at {physics.FIXED_Q_ENERGY_EV:.0f} eV.
See the [main card](https://huggingface.co/{repo}#single-channel-fixed-q-baselines) for the comparison
with the other inputs.

Labels (in `config.json` order), scaled to [0, 1]: `physical = label_min + y·(label_max − label_min)`.

| label | validation MAE (R²) |
|---|---|
{label_table}

```python
from network import OldNet
from safetensors.torch import load_file

model = OldNet(n_q={s['config']['n_q']}, n_labels={len(s['label_names'])})
model.load_state_dict(load_file("model.safetensors"))
model.eval()
# x: (batch, {s['config']['n_q']}) curves on the fixed q grid, prepared as in `config.json` "input":
#   s -> ln R_s,  p -> ln R_p,  anisotropy -> (P - S c) / (P + S c) with c = cos^2(2 theta) at 250 eV
scaled = model(x)  # labels in [0, 1]
```

![snapshot](figures/snapshot_best.png)
"""


# =====================================================================================
# Staging the files
# =====================================================================================


def stage_main(section, repo):
    """The current Hub card with the single-channel section, plus the figures. Returns the folder."""
    d = STAGE / "main"
    shutil.rmtree(d, ignore_errors=True)
    (d / "figures").mkdir(parents=True)
    card_path = hf_hub_download(repo, "README.md", local_dir=STAGE / "hub_card", force_download=True)
    card = Path(card_path).read_text()
    (d / "README.md").write_text(insert_section(card, section))
    for name in ("single_channel_inputs.png", "single_channel_training.png", "single_channel_quality.png"):
        shutil.copy(FIG_DIR / name, d / "figures" / name)
    return d


def insert_section(card, section):
    """Replace the single-channel section of ``card``, or add it at the end if there is none."""
    # Tag the card with both q-grid kinds (the old tag name was "dynamic-q").
    card = card.replace("mlp, dynamic-q]", "mlp, variable-q, fixed-q]")
    start = card.find(SECTION_START)
    end = card.find(SECTION_END)
    if start != -1 and end != -1:
        after = card[end + len(SECTION_END):].lstrip("\n")
        return card[:start] + section.rstrip("\n") + "\n" + after
    return card.rstrip("\n") + "\n\n" + section


def stage_run(s, repo):
    """Everything one run's branch holds, in STAGE/<branch>/. Returns that folder."""
    d = STAGE / s["branch"]
    shutil.rmtree(d, ignore_errors=True)
    (d / "figures").mkdir(parents=True)
    run_dir = ROOT / "checkpoints" / s["ckpt_id"]

    # Weights as safetensors (each tensor must be contiguous in memory), plus the checkpoint.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    weights = {}
    for name in checkpoint["model_state_dict"]:
        weights[name] = checkpoint["model_state_dict"][name].contiguous()
    save_file(weights, d / "model.safetensors", metadata={"epoch": str(checkpoint["epoch"]), "input": s["input"]})
    shutil.copy(run_dir / "best.pt", d / "best.pt")
    shutil.copy(run_dir / "metrics.json", d / "metrics.json")
    shutil.copy(run_dir / "best.png", d / "figures/snapshot_best.png")
    shutil.copy(ROOT / "refltron/single_channel_mlp/network.py", d / "network.py")

    # Training configuration with the label names and ranges.
    config = checkpoint["config"].copy()
    label_min, label_max = label_range(s["label_names"])
    config["label_names"] = list(s["label_names"])
    config["label_min"] = label_min.tolist()
    config["label_max"] = label_max.tolist()
    config["energy_ev"] = physics.FIXED_Q_ENERGY_EV
    (d / "config.json").write_text(json.dumps(config, indent=2))

    # Only the epoch lines of the training log.
    log_lines = []
    for line in s["log_path"].read_text(errors="ignore").splitlines():
        if EPOCH_RE.search(line):
            log_lines.append(line)
    (d / "training_log.txt").write_text("\n".join(log_lines) + "\n")
    (d / "README.md").write_text(run_card(s, repo))
    return d


def parse_args():
    """Read command-line flags. Every flag defaults to the matching setting at the top of this file."""
    parser = argparse.ArgumentParser(
        description="Add the single-channel fixed-q models to the Refltron model card on the Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # show defaults in --help
    )
    parser.add_argument("--repo", default=REPO, help="model repository on the Hub")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN, help=f"only stage files under {STAGE}")
    return parser.parse_args()


if __name__ == "__main__":
    main()
