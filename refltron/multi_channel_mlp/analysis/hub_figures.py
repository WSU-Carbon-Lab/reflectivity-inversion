"""Figures for the Refltron Hugging Face page: hero, benchmark and training curves.

    uv run python -m refltron.multi_channel_mlp.analysis.eval_versions     # first: scores + per-curve predictions
    uv run python -m refltron.multi_channel_mlp.analysis.hub_figures       # writes analysis/runs/hub/*.png

hero.png        what the model does: one held-out curve -> its depth profile, plus
                predicted vs true thickness over the whole validation split
benchmark.png   the three versions compared on the same 100k validation curves
training.png    training curves of all versions; training_<version>.png per version
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch


from refltron.common import plot_style as ps
from refltron.common.loss import LABEL_NAMES
from refltron.common.paths import PROJECT_ROOT as ROOT
from refltron.multi_channel_mlp.analysis.diagnostics import load_model
from refltron.common.depth_profile import compute_depth_profile, estimate_substrate_sld

ps.apply()

RUNS = {"v1-relu-rq4": ("d0z13nm4", "relu-rq4-norm"),
        "v2-relu-log": ("pj7cue8j", "relu-log-norm"),
        "v3-gelu-rq4": ("al8du562", "gelu-rq4-norm")}
HERO = "v3-gelu-rq4"
V3_RESUME_EPOCH = 143
TRAIN_MIN = 20.0   # film thickness range of the training data [Å]
TRAIN_MAX = 800.0
VAL_SHARD = ROOT / "data/reflectivity-1m/data/train-00009-of-00010.parquet"
VAL_ROW0 = 900_000
EVAL = json.loads((ROOT / "analysis/runs/version_eval.json").read_text())
OUT = ROOT / "analysis/runs/hub"
OUT.mkdir(parents=True, exist_ok=True)

# Sequential blue ramp (light -> dark) for densities.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
LABEL_TEXT = {
    "film_thickness": "film thickness", "film_roughness": "film roughness", "oxide_thickness": "oxide thickness",
    "delta_xx": "δ in-plane", "delta_zz": "δ out-of-plane", "beta_xx": "β in-plane", "beta_zz": "β out-of-plane",
}
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+ \| train\(\w+\) ([\d.e+-]+) \| val ([\d.e+-]+) \| thickness MAE ([\d.]+) Å \| "
                      r"OOD 0Å->([-\d.]+) Å \| OOD 2000Å->([-\d.]+) Å")


# ----------------------------------------------------------------------------- helpers
def read_log(name):
    """Per-epoch numbers from one sweep training log."""
    rows = []
    for line in (ROOT / "logs/sweep" / f"{name}.log").read_text().splitlines():
        match = EPOCH_RE.search(line)
        if match:
            rows.append(match.groups())
    a = np.array(rows, dtype=float)
    return {"epoch": a[:, 0], "train": a[:, 1], "val": a[:, 2], "mae": a[:, 3], "ood0": a[:, 4], "ood2000": a[:, 5]}


def rolling(x, w=25):
    """Centered moving average (edges use the available points)."""
    k = np.ones(w)
    return np.convolve(x, k, "same") / np.convolve(np.ones_like(x), k, "same")


def load_val_row(idx):
    """One validation row as (s, p, q) tensors of shape (1, 256) and its 7 labels."""
    df = pl.read_parquet(VAL_SHARD).slice(idx - VAL_ROW0, 1)
    s = torch.from_numpy(df["s"].to_numpy().copy())
    p = torch.from_numpy(df["p"].to_numpy().copy())
    q = torch.from_numpy(df["q"].to_numpy().copy())
    labels = df.select(LABEL_NAMES).to_numpy()[0]
    return s, p, q, labels


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=ps.FIG_DPI)
    plt.close(fig)
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- hero
def pick_hero_row(pred, truth):
    """A typical held-out film: 250-400 Å, smooth enough to show fringes, median v3 error."""
    thickness = truth[:, 0]
    roughness = truth[:, 1]
    candidates = np.where((thickness > 250) & (thickness < 400) & (roughness < 6))[0]
    error = np.abs(pred[candidates, 0] - thickness[candidates])
    median_position = np.argsort(error)[len(candidates) // 2]
    return int(candidates[median_position])


def draw_profile(ax, true_lab, pred_lab, sld, color_pred=True):
    """Truth (faint) and prediction (dashed) depth profiles, with the true layers shaded."""
    top = max(true_lab[0] + true_lab[2], pred_lab[0] + pred_lab[2])
    h = np.linspace(-12, top * 1.12 + 15, 1500)
    th, _, tc = compute_depth_profile(true_lab, h, sld)
    _, _, pc = compute_depth_profile(pred_lab, h, sld)
    edges = [h[-1], th[0], th[1], th[2], h[0]]
    layers = ("vacuum", "film", "oxide", "Si")
    for i in range(len(layers)):
        layer = layers[i]
        ax.axvspan(edges[i + 1], edges[i], color=ps.LAYER_SHADES[layer], lw=0, zorder=0)
        mid = 0.5 * (edges[i] + edges[i + 1])
        if edges[i] - edges[i + 1] > 0.06 * (h[-1] - h[0]):
            ax.text(mid, 0.97, layer, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=10.5, color=ps.MUTED)
    film_mid = 0.5 * (th[0] + th[1])
    j = np.argmin(np.abs(h - film_mid))  # the height point in the middle of the film
    for key in ps.COMPONENT_COLORS:
        color = ps.COMPONENT_COLORS[key][0]
        tex = ps.COMPONENT_COLORS[key][1]
        ax.plot(h, tc[key], color=color, lw=5, alpha=0.28, solid_capstyle="butt", zorder=2)
        ax.plot(h, pc[key], color=color, lw=1.8, ls=(0, (5, 2.2)), zorder=3)
        ax.text(h[j], max(pc[key][j], tc[key][j]), f"{tex}\n", color=color, fontsize=12.5, va="center",
                ha="center", zorder=4)
    ax.set_xlim(h[-1], h[0])
    lowest = np.inf
    highest = -np.inf
    for key in tc:
        lowest = min(lowest, tc[key].min())
        highest = max(highest, tc[key].max())
    ax.set_ylim(min(0, lowest) - 3e-4, highest * 1.25)
    return th


def make_hero(npz):
    pred = npz[HERO]
    truth = npz["truth"]
    row = pick_hero_row(pred, truth)
    s, p, q, true_lab = load_val_row(VAL_ROW0 + row)
    model, crit, _ = load_model(ROOT / "checkpoints" / RUNS[HERO][0] / "best.pt")
    with torch.no_grad():
        pred_lab = crit.denormalize(model(s, p, q))[0].numpy()
    q = q[0].numpy()
    s = s[0].numpy()
    p = p[0].numpy()
    sld = estimate_substrate_sld(q)
    ev = EVAL[HERO]

    fig = plt.figure(figsize=(17.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.95], wspace=0.32,
                          left=0.05, right=0.955, top=0.78, bottom=0.16)
    a_in = fig.add_subplot(gs[0])
    a_prof = fig.add_subplot(gs[1])
    a_acc = fig.add_subplot(gs[2])

    fig.text(0.05, 0.955, "Refltron reads a thin film's structure from its X-ray reflectivity",
             fontsize=21, weight="medium", color=ps.INK, va="top")
    fig.text(0.05, 0.885,
             f"Version v3 (GELU)  ·  film-thickness error {ev['mae']['film_thickness']:.0f} Å (MAE)  ·  "
             f"R² {ev['r2']['film_thickness']:.3f}  ·  measured on {ev['n_rows']:,} held-out simulated curves",
             fontsize=13, color=ps.INK_2, va="top")

    # A. input: the s and p curves, each labelled at its right end.
    a_in.plot(q, s, color=ps.POLARIZATION_COLORS["s"], lw=1.6)
    a_in.text(q[-1], s[-1], "  s-pol.", color=ps.POLARIZATION_COLORS["s"], fontsize=11, va="center")
    a_in.plot(q, p, color=ps.POLARIZATION_COLORS["p"], lw=1.6)
    a_in.text(q[-1], p[-1], "  p-pol.", color=ps.POLARIZATION_COLORS["p"], fontsize=11, va="center")
    a_in.set_yscale("log")
    ps.no_minor_grid(a_in)
    a_in.set_xlabel("q  [Å⁻¹]")
    a_in.set_ylabel("reflectivity")
    a_in.set_title("1 · Input: s- and p-polarized reflectivity")
    a_in.set_xlim(0, q[-1] * 1.16)

    # B. depth profile
    draw_profile(a_prof, true_lab, pred_lab, sld)
    a_prof.set_xlabel("height above the silicon interface  [Å]")
    a_prof.set_ylabel("optical constants  δ, β")
    a_prof.set_title("2 · Output: depth profile of the film")
    a_prof.text(0.0, -0.215, f"Faint = truth, dashed = Refltron. A typical held-out curve (median error): "
                f"film {true_lab[0]:.0f} Å, predicted {pred_lab[0]:.0f} Å.",
                transform=a_prof.transAxes, fontsize=10.5, color=ps.INK_2)

    # C. accuracy over the validation split
    t = truth[:, 0]
    y = pred[:, 0]
    hb = a_acc.hexbin(t, y, gridsize=70, extent=(0, 820, 0, 820), cmap=BLUES, norm=LogNorm(), mincnt=1, lw=0)
    a_acc.plot([0, 820], [0, 820], color=ps.INK, lw=1.0)
    a_acc.set_xlim(0, 820)
    a_acc.set_ylim(0, 820)
    a_acc.set_aspect("equal")
    a_acc.grid(False)
    a_acc.set_xlabel("true film thickness  [Å]")
    a_acc.set_ylabel("predicted  [Å]")
    a_acc.set_title(f"3 · All {ev['n_rows'] // 1000}k held-out films")
    a_acc.text(560, 585, "perfect prediction", rotation=45, rotation_mode="anchor", fontsize=10, color=ps.INK,
               ha="center", va="bottom", transform_rotates_text=True)
    cb = fig.colorbar(hb, ax=a_acc, fraction=0.045, pad=0.03)
    cb.set_label("curves per cell", color=ps.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=ps.AXIS, labelcolor=ps.MUTED)

    # arrow between input and output
    fig.add_artist(FancyArrowPatch((0.298, 0.47), (0.325, 0.47), transform=fig.transFigure,
                                   arrowstyle="-|>,head_length=7,head_width=4", lw=1.6, color=ps.INK_2))
    save(fig, "hero.png")


# ----------------------------------------------------------------------------- benchmark
def benchmark_value(evaluation, key):
    """One benchmark number of one version: "thick" is the film-thickness MAE, else EVAL[version][key]."""
    if key == "thick":
        return evaluation["mae"]["film_thickness"]
    return evaluation[key]


def hbar_panel(ax, key, title, fmt):
    """Horizontal bars of one benchmark number for every version; the best is marked."""
    versions = list(RUNS)
    vals = []
    colors = []
    labels = []
    for v in versions:
        vals.append(benchmark_value(EVAL[v], key))
        colors.append(ps.VERSION_COLORS[v])
        labels.append(ps.VERSION_LABELS[v])
    ypos = np.arange(len(versions))[::-1]  # first version at the top
    ax.barh(ypos, vals, height=0.46, color=colors, edgecolor=ps.SURFACE, lw=2)
    best = min(vals)
    for i in range(len(versions)):
        val = vals[i]
        if val == best:
            text = f"  {fmt.format(val)}" + "  best"
            ax.text(val, ypos[i], text, va="center", fontsize=11, color=ps.INK, weight="medium")
        else:
            text = f"  {fmt.format(val)}"
            ax.text(val, ypos[i], text, va="center", fontsize=11, color=ps.INK_2, weight="normal")
    ax.set_yticks(ypos, labels, color=ps.INK_2, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.spines["left"].set_visible(False)


def make_benchmark(npz, ood):
    fig = plt.figure(figsize=(17.5, 10.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], hspace=0.42, wspace=0.42,
                          left=0.085, right=0.95, top=0.87, bottom=0.07)
    n = EVAL[HERO]["n_rows"]
    fig.text(0.04, 0.965, "Benchmark: the three Refltron versions on the same held-out curves", fontsize=20,
             weight="medium", va="top")
    fig.text(0.04, 0.925, f"Each version's best checkpoint, scored on all {n:,} validation curves "
             "(rows the networks never trained on). Lower is better everywhere.", fontsize=12.5, color=ps.INK_2,
             va="top")

    hbar_panel(fig.add_subplot(gs[0, 0]), "hybrid_loss", "Hybrid loss (training objective)", "{:.4f}")
    hbar_panel(fig.add_subplot(gs[0, 1]), "profile_l2", "Depth-profile error (L2)", "{:.4f}")
    hbar_panel(fig.add_subplot(gs[0, 2]), "thick", "Film-thickness error (MAE, Å)", "{:.1f} Å")

    # Per-label error relative to v1.
    ax = fig.add_subplot(gs[1, 0])
    labels = list(LABEL_NAMES)
    ypos = np.arange(len(labels))[::-1]
    ax.axvline(100, color=ps.AXIS, lw=1.0)
    for v in RUNS:
        rel = []
        for lab in labels:
            rel.append(100 * EVAL[v]["mae"][lab] / EVAL["v1-relu-rq4"]["mae"][lab])
        ax.scatter(rel, ypos, s=70, color=ps.VERSION_COLORS[v], edgecolor=ps.SURFACE, lw=2, zorder=3,
                   label=ps.VERSION_LABELS[v])
    tick_text = []
    for lab in labels:
        tick_text.append(LABEL_TEXT[lab])
    ax.set_yticks(ypos, tick_text, color=ps.INK_2, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("error relative to v1  [%]")
    ax.set_title("Error on every label, relative to v1")
    handles_and_labels = ax.get_legend_handles_labels()
    fig.legend(handles_and_labels[0], handles_and_labels[1], loc="upper right", bbox_to_anchor=(0.95, 0.975),
               ncol=3, fontsize=11.5, handletextpad=0.3, columnspacing=1.4)
    ax.spines["left"].set_visible(False)

    # Thickness error vs film thickness.
    ax = fig.add_subplot(gs[1, 1])
    edges = np.linspace(20, 800, 27)
    centers = 0.5 * (edges[1:] + edges[:-1])
    t = npz["truth"][:, 0]
    idx = np.digitize(t, edges) - 1  # which thickness bin each curve falls in
    ends = []
    for v in RUNS:
        e = np.abs(npz[v][:, 0] - t)
        mae = np.zeros(len(centers))
        for i in range(len(centers)):
            mae[i] = e[idx == i].mean()
        ax.plot(centers, mae, color=ps.VERSION_COLORS[v], lw=2)
        ends.append((mae[-1], v.split("-")[0], ps.VERSION_COLORS[v]))
    ax.set_xlim(0, 800)
    ax.set_ylim(0, None)
    ax.set_xlabel("true film thickness  [Å]")
    ax.set_ylabel("thickness MAE  [Å]")
    ax.set_title("Thickness error grows with film thickness")
    _right_labels(ax, ends)

    # Out-of-distribution: one row per case, one dot per version.
    ax = fig.add_subplot(gs[1, 2])
    ax.axvspan(TRAIN_MIN, TRAIN_MAX, color=ps.LAYER_SHADES["film"], lw=0, zorder=0)
    ax.text(410, 1.62, "training range 20–800 Å", ha="center", fontsize=10, color=ps.MUTED)
    case_names = ["bare oxide\n(truth 0 Å)", "2000 Å film"]
    case_truths = [0.0, 2000.0]
    case_keys = ["ood0", "ood2000"]
    for row in range(len(case_names)):
        yy = 1 - row
        truth_val = case_truths[row]
        ax.plot([truth_val], [yy], marker="|", ms=22, mew=2.5, color=ps.INK, zorder=4)
        ax.text(truth_val, yy + 0.2, "truth", ha="center", fontsize=10, color=ps.INK)
        versions = list(RUNS)
        for k in range(len(versions)):
            v = versions[k]
            val = ood[v][case_keys[row]]
            ax.scatter([val], [yy - 0.12 + 0.12 * k], s=70, color=ps.VERSION_COLORS[v], edgecolor=ps.SURFACE,
                       lw=2, zorder=3)
            ax.text(val, yy - 0.12 + 0.12 * k, f"  {v.split('-')[0]}  {val:.0f} Å", va="center", fontsize=9.5,
                    color=ps.INK_2)
    ax.set_yticks([1, 0], case_names, color=ps.INK_2, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlim(-80, 2300)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("predicted film thickness  [Å]")
    ax.set_title("Outside the training range")
    save(fig, "benchmark.png")


def _right_labels(ax, ends, log=False):
    """Colour-keyed labels just outside the right edge, at the line-end heights, de-collided.

    ``ends`` holds (y value, text, colour) for each line.
    """
    lo, hi = ax.get_ylim()
    # Convert each y value to a fraction of the axis height (0 = bottom, 1 = top).
    items = []
    for end in ends:
        y = end[0]
        if log:
            fraction = (np.log10(y) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        else:
            fraction = (y - lo) / (hi - lo)
        items.append((fraction, end[1], end[2]))
    items.sort()  # bottom to top
    # Push each label up if it would overlap the one below it.
    placed = []
    for item in items:
        fraction = item[0]
        if placed:
            fraction = max(fraction, placed[-1][0] + 0.06)
        placed.append((fraction, item[1], item[2]))
    for item in placed:
        ax.text(1.015, item[0], item[1], transform=ax.transAxes, color=item[2], fontsize=11, weight="medium",
                va="center", ha="left", clip_on=False)


# ----------------------------------------------------------------------------- training
def version_tag(version, n_versions):
    """Short line label ("v1") when several versions share a panel, else nothing."""
    if n_versions > 1:
        return version.split("-")[0]
    return ""


def mark_resume(ax, resumed):
    """Vertical line at the epoch where v3 was resumed (only when v3 is shown)."""
    if resumed:
        ax.axvline(V3_RESUME_EPOCH, color=ps.AXIS, lw=1.0, zorder=1)


def plot_ood_panel(ax, versions, logs, key, truth, title, resumed):
    """Predicted thickness on one out-of-distribution curve over training (raw faint, smoothed bold)."""
    ax.axhspan(TRAIN_MIN, TRAIN_MAX, color=ps.LAYER_SHADES["film"], lw=0, zorder=0)
    ax.axhline(truth, color=ps.INK, lw=1.2)
    ax.text(8, truth, "truth", va="bottom", fontsize=10, color=ps.INK)
    ends = []
    for v in versions:
        log = logs[v]
        color = ps.VERSION_COLORS[v]
        ax.plot(log["epoch"], log[key], color=color, lw=0.8, alpha=0.22)
        smooth = rolling(log[key])
        ax.plot(log["epoch"], smooth, color=color, lw=2)
        ends.append((smooth[-1], f"{version_tag(v, len(versions))} {smooth[-1]:.0f} Å".strip(), color))
    ax.set_title(title)
    mark_resume(ax, resumed)
    if key == "ood0":
        ax.set_ylim(-40, 600)
    else:
        ax.set_ylim(0, 2200)
        ax.text(590, 400, "training range\n20–800 Å", ha="right", va="center", fontsize=10, color=ps.MUTED)
    _right_labels(ax, ends)


def make_training(versions, logs, name, title):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.8))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.86, bottom=0.08, hspace=0.42, wspace=0.36)
    fig.text(0.04, 0.965, title, fontsize=20, weight="medium", va="top")
    fig.text(0.04, 0.925, "Hybrid loss · Adam 5e-4, cosine decay, 600 epochs, batch 1024 · "
             "OOD panels: 25-epoch moving average over raw "
             "values (faint). Shaded = training range.", fontsize=12.5, color=ps.INK_2, va="top")
    n_versions = len(versions)
    resumed = "v3-gelu-rq4" in versions

    # 1. loss
    ax = axes[0, 0]
    ends = []
    for v in versions:
        log = logs[v]
        color = ps.VERSION_COLORS[v]
        ax.plot(log["epoch"], log["train"], color=color, lw=1.2, alpha=0.5)
        ax.plot(log["epoch"], log["val"], color=color, lw=2)
        ends.append((log["val"][-1], f"{version_tag(v, n_versions)} val".strip(), color))
    ax.set_yscale("log")
    ps.no_minor_grid(ax)
    ax.set_ylim(1.2e-2, 2.2e-1)
    mark_resume(ax, resumed)
    ax.set_title("Loss  (thick = validation, faint = training)")
    _right_labels(ax, ends, log=True)

    # 2. thickness MAE
    ax = axes[0, 1]
    ends = []
    for v in versions:
        log = logs[v]
        ax.plot(log["epoch"], log["mae"], color=ps.VERSION_COLORS[v])
        ends.append((log["mae"][-1], f"{version_tag(v, n_versions)} {log['mae'][-1]:.1f} Å".strip(),
                     ps.VERSION_COLORS[v]))
    ax.set_ylim(20, 60)
    mark_resume(ax, resumed)
    ax.set_title("Validation thickness MAE [Å]  (first epochs off-scale)")
    _right_labels(ax, ends)

    # 3. generalization gap
    ax = axes[0, 2]
    ends = []
    for v in versions:
        log = logs[v]
        gap = log["val"] - log["train"]
        ax.plot(log["epoch"], gap, color=ps.VERSION_COLORS[v], lw=1.6)
        label = version_tag(v, n_versions)
        if label == "":
            label = f"{gap[-1]:.3f}"  # a single version: show the final gap instead
        ends.append((gap[-1], label, ps.VERSION_COLORS[v]))
    ax.axhline(0, color=ps.AXIS, lw=1.0)
    ax.set_ylim(-0.01, 0.03)
    mark_resume(ax, resumed)
    ax.set_title("Overfitting gap  (validation − training loss)")
    _right_labels(ax, ends)

    # 4/5. OOD
    plot_ood_panel(axes[1, 0], versions, logs, "ood0", 0, "Bare oxide (truth 0 Å): predicted film [Å]", resumed)
    plot_ood_panel(axes[1, 1], versions, logs, "ood2000", 2000, "2000 Å film: predicted thickness [Å]", resumed)

    # 6. fringe detail per layer
    ax = axes[1, 2]
    ends = []
    for v in versions:
        d = json.loads((ROOT / "analysis/runs" / RUNS[v][0] / "best.json").read_text())["morph"]
        xs = np.arange(len(d["stages"]))
        ax.plot(xs, d["fringe_r2"], "-o", color=ps.VERSION_COLORS[v], ms=5, mec=ps.SURFACE, mew=1.5,
                label=ps.VERSION_LABELS[v])
        ends.append((d["fringe_r2"][-1], version_tag(v, n_versions), ps.VERSION_COLORS[v]))
        short_names = []
        for stage in d["stages"]:
            short_names.append(stage.split(" ")[0])
        ax.set_xticks(xs, short_names, fontsize=10)
    ax.axhline(0, color=ps.AXIS, lw=1.0)
    ax.set_ylim(-0.35, 1.05)
    ax.set_ylabel("decoder R²")
    ax.set_title("Fringe detail still recoverable after each layer")
    ax.legend(loc="upper right", fontsize=10)

    for a in axes.flat[:5]:
        a.set_xlabel("epoch")
        a.set_xlim(0, 600)
    if resumed:
        axes[0, 0].text(V3_RESUME_EPOCH + 6, 1.9e-1, "v3 resumed", fontsize=9.5, color=ps.MUTED, va="top")
    save(fig, name)


# ----------------------------------------------------------------------------- main
def main():
    npz = np.load(ROOT / "analysis/runs/version_predictions.npz")
    logs = {}
    ood = {}
    for v in RUNS:
        log_name = RUNS[v][1]
        logs[v] = read_log(log_name)
        # Out-of-distribution predictions: mean over the last 50 epochs.
        ood[v] = {"ood0": float(logs[v]["ood0"][-50:].mean()), "ood2000": float(logs[v]["ood2000"][-50:].mean())}
    make_hero(npz)
    make_benchmark(npz, ood)
    make_training(list(RUNS), logs, "training.png", "Training curves of the Refltron versions")
    for v in RUNS:
        make_training([v], logs, f"training_{v}.png", f"Training curves · {ps.VERSION_LABELS[v]}")


if __name__ == "__main__":
    main()
