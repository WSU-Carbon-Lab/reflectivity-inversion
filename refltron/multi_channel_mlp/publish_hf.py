"""Publish trained runs to the Hugging Face Hub as versions (branches) of one model repo.

    uv run python -m refltron.multi_channel_mlp.publish_hf --dry-run   # stage only
    uv run python -m refltron.multi_channel_mlp.publish_hf             # stage + upload
    uv run python -m refltron.multi_channel_mlp.publish_hf --help      # list every option

Each version gets its own branch holding weights, config, training log, diagnostics and
figures. ``main`` holds the overview card and the cross-version comparison.

What happens:
    1. Summarize every version in VERSIONS: its checkpoint, its training log, its
       diagnostics (analysis/diagnostics.py) and its validation scores
       (analysis/eval_versions.py).
    2. Stage the files of each branch under STAGE: weights, config, log, figures and a
       README model card.
    3. Unless --dry-run, upload ``main`` and one branch per version.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from refltron.common.loss import LABEL_NAMES, LabelMSELoss
from refltron.common.paths import PROJECT_ROOT

# =====================================================================================
# DEFAULT SETTINGS: edit these, or override the run options from the command line.
# =====================================================================================

REPO = "carbon-lab/refltron"     # model repository on the Hugging Face Hub
PUBLIC = False                   # True = create the repository public (default private)
DRY_RUN = False                  # True = only stage the files, do not upload

ROOT = PROJECT_ROOT
STAGE = Path("/tmp/hf_stage")    # where the files of every branch are prepared before upload
DATASET = "carbon-lab/reflectivity-1m"

# version branch, checkpoint id, sweep log name, short description
VERSIONS = [
    ("v1-relu-rq4", "d0z13nm4", "relu-rq4-norm", "ReLU, R·q⁴ input regularizer, standardized inputs, 600 epochs"),
    ("v2-relu-log", "pj7cue8j", "relu-log-norm", "ReLU, log10 input regularizer, standardized inputs, 600 epochs"),
    ("v3-gelu-rq4", "al8du562", "gelu-rq4-norm", "GELU, R·q⁴ input regularizer, standardized inputs, 600 epochs"),
]
# Branch from an earlier upload that a newer version replaces (left on the Hub for reference).
SUPERSEDED = {"v3-gelu-rq4-e146": "v3-gelu-rq4"}
# Full-validation-split scores of every version, written by refltron/multi_channel_mlp/analysis/eval_versions.py.
EVAL_PATH = ROOT / "analysis/runs/version_eval.json"
THICKNESS_BINS = ("20-100", "100-400", "400-800")
# One line of the training log, for example:
#   Epoch  12/600 | train(hybrid) 3.1e-01 | val 2.8e-01 | thickness MAE 190.9 Å | OOD 0Å->408.1 Å | OOD 2000Å->408 Å
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+ \| train\(\w+\) ([\d.e+-]+) \| val ([\d.e+-]+) \| thickness MAE ([\d.]+) Å \| "
                      r"OOD 0Å->([-\d.]+) Å \| OOD 2000Å->([-\d.]+) Å")

# The training range of every label, {name: (min, max)}, and the same as a Markdown table.
_label_mse = LabelMSELoss()
_label_min = _label_mse.label_min.tolist()
_label_max = _label_mse.label_max.tolist()
LABEL_BOUNDS = {}
_bound_lines = ["| label | min | max |", "|---|---|---|"]
for _i in range(len(LABEL_NAMES)):
    LABEL_BOUNDS[LABEL_NAMES[_i]] = (float(_label_min[_i]), float(_label_max[_i]))
    _bound_lines.append(f"| {LABEL_NAMES[_i]} | {_label_min[_i]:.4g} | {_label_max[_i]:.4g} |")
BOUNDS_TABLE = "\n".join(_bound_lines)

# How each input regularizer is written in the model card.
REGULARIZER_NAMES = {"log": "log10(R)", "rq4": "R·q⁴", "log_rq4": "log10(R·q⁴)"}

ARCH = """\
| stage | op | shape | activation |
|---|---|---|---|
| input | regularize (s, p) → (S, P), stack with q, per-channel standardize | (B, 3, 256) | – |
| mixer | Conv1d 3→8, kernel 1 (pointwise channel mix at every q) | (B, 8, 256) → flatten 2048 | act |
| L1 | Linear 2048→400 | 400 | **none** |
| L2 | Linear 400→800 | 800 | act |
| L3 | Linear 800→400 | 400 | act |
| L4 | Linear 400→300 | 300 | act |
| L5 | Linear 300→200 | 200 | act |
| L6 | Linear 200→100 | 100 | act |
| L7 | Linear 100→7 | 7 normalized labels | – |
"""

LABELS = ("film_thickness, film_roughness, oxide_thickness, delta_xx, delta_zz, beta_xx, beta_zz "
          "(normalized to [0, 1]: `physical = min + y·(max − min)`, bounds below and in `config.json`)")

HUB_FIGS = ROOT / "analysis/runs/hub"  # written by refltron/multi_channel_mlp/analysis/hub_figures.py
# Every per-version figure and what it shows: {file under figures/: description}.
FIG_DOC = {
    "training.png": "Training curves of this version: loss, thickness MAE, overfitting gap, out-of-distribution "
    "predictions per epoch, and how much fringe detail each layer keeps.",
    "snapshot_best.png": "Training snapshot at the best epoch: s/p reflectivity, Si-aligned depth profile "
    "(height above the Si interface, substrate at the right; solid = truth, dashed = network), and normalized "
    "labels, for one training row, one validation row and two out-of-distribution curves (bare oxide, 2000 Å film).",
    "morph_best.png": "How the reflectivity curve morphs through the network. Each row is one stage; the left three "
    "columns show what a ridge decoder can reconstruct of three validation curves (80 / 300 / 650 Å films) from "
    "that stage's activations, the fourth gives per-q decoder R² (whole curve and high-pass fringe detail), the fifth "
    "a PCA of the stage coloured by film thickness. Bottom row: predicted labels re-simulated back into curves.",
    "layer_health_best.png": "Per-layer health: dead / always-on units, sparsity, gain (RMS out / in), spike ratio, "
    "effective rank, redundancy, linear-probe R² for each label, and weight spectra.",
    "saliency_best.png": "Input sensitivity: |∂label/∂input| × input std across q for each label, and mixer "
    "channel usage.",
    "evolution.png": "Layer health across the periodic checkpoints of the run.",
    "cards/": "One standardized card per layer: receptive fields over q, firing-rate distribution, probe R² per "
    "label, activation PCA, and weight spectrum.",
}
# Analysis figures copied to each branch: {file in analysis/runs/<run>/: file under figures/}.
FIGURE_COPIES = {
    "best_snapshot.png": "snapshot_best.png",
    "best_morph.png": "morph_best.png",
    "best_layer_health.png": "layer_health_best.png",
    "best_saliency.png": "saliency_best.png",
    "evolution.png": "evolution.png",
}
# The three length labels, in table column order.
LENGTH_LABELS = ("film_thickness", "film_roughness", "oxide_thickness")


def main():
    args = parse_args()

    # 1. Summarize every version.
    summaries = []
    for version in VERSIONS:
        summaries.append(summarize(version[0], version[1], version[2], version[3]))

    # 2. Stage the files of every branch.
    main_dir = stage_main(summaries, args.repo)
    version_dirs = []
    for summary in summaries:
        version_dirs.append(stage_version(summary, args.repo))
    print(f"staged under {STAGE}")
    if args.dry_run:
        return

    # 3. Upload main, then one branch per version.
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=main_dir, commit_message="Overview card and version comparison",
                      delete_patterns=["figures/sweep_comparison.png"])  # replaced by benchmark.png / training.png
    for i in range(len(summaries)):
        summary = summaries[i]
        api.create_branch(args.repo, branch=summary["version"], exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=version_dirs[i], revision=summary["version"],
                          commit_message=f"{summary['version']}: {summary['desc']}")
        print(f"uploaded {summary['version']}")
    print(f"https://huggingface.co/{args.repo}")


# =====================================================================================
# Reading the results of one version
# =====================================================================================


def parse_log(path: Path):
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
            "ood0": float(groups[4]),
            "ood2000": float(groups[5]),
        })
    return rows


def last_ckpt(run: Path):
    """The newest numbered checkpoint (epoch_*.pt) of a run, or None if there is none."""
    epoch_files = sorted(run.glob("epoch_*.pt"))
    if len(epoch_files) == 0:
        return None
    return epoch_files[-1]


def tail_mean(rows, key):
    """Average of ``key`` over the last 50 log rows."""
    tail = rows[-50:]
    total = 0
    for row in tail:
        total += row[key]
    return total / len(tail)


def first_stage_value(values_by_stage, prefix):
    """The value of the first stage whose name starts with ``prefix``, or None."""
    for stage in values_by_stage:
        if stage.startswith(prefix):
            return values_by_stage[stage]
    return None


def summarize(version, ckpt_id, log_name, desc):
    """Collect everything the model cards need about one version into a dict."""
    run = ROOT / "checkpoints" / ckpt_id
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    log = parse_log(ROOT / "logs/sweep" / f"{log_name}.log")

    # Diagnostics written by analysis/diagnostics.py (empty if it was not run).
    diag_path = ROOT / "analysis/runs" / ckpt_id / "best.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
    else:
        diag = {}
    val_metrics = diag.get("val_metrics", {})

    # The log row with the lowest validation loss (the first one, if several tie).
    best = log[0]
    for row in log:
        if row["val"] < best["val"]:
            best = row

    # Fringe-detail decoder R² of each network stage: {stage name: R²}.
    morph = diag.get("morph", {})
    stages = morph.get("stages", [])
    fringe_values = morph.get("fringe_r2", [])
    fringe_by_stage = {}
    for i in range(min(len(stages), len(fringe_values))):
        fringe_by_stage[stages[i]] = fringe_values[i]

    # The largest fraction of dead units in any layer (None without diagnostics).
    worst_dead = None
    for layer in diag.get("layers", []):
        if worst_dead is None or layer["dead_frac"] > worst_dead:
            worst_dead = layer["dead_frac"]

    # Validation scores from analysis/eval_versions.py (None if it was not run).
    if EVAL_PATH.exists():
        ev = json.loads(EVAL_PATH.read_text()).get(version)
    else:
        ev = None
    if ev is not None and ev["epoch"] != ck["epoch"]:
        raise SystemExit(f"{EVAL_PATH} is stale for {version}; rerun refltron/multi_channel_mlp/analysis/eval_versions.py")
    if ev:
        label_mse = ev["label_mse"]
        profile_l2 = ev["profile_l2"]
    else:
        label_mse = val_metrics.get("label_mse")
        profile_l2 = val_metrics.get("profile_l2")

    return {
        "eval": ev,
        "version": version,
        "ckpt_id": ckpt_id,
        "log_name": log_name,
        "desc": desc,
        "config": ck["config"],
        "best_epoch": ck["epoch"],
        "best_val": ck["val_loss"],
        "epochs_run": log[-1]["epoch"],
        "epochs_planned": ck["config"]["epochs"],
        "log_best_val": best["val"],
        "log_best_epoch": best["epoch"],
        "final": log[-1],
        "ood0_tail": tail_mean(log, "ood0"),
        "ood2000_tail": tail_mean(log, "ood2000"),
        "label_mse": label_mse,
        "profile_l2": profile_l2,
        "val_mae": val_metrics.get("mae"),
        "worst_dead": worst_dead,
        "fringe_l1": first_stage_value(fringe_by_stage, "L1 "),
        "fringe_mixer": first_stage_value(fringe_by_stage, "mixer"),
        "flags": diag.get("flags", []),
    }


# =====================================================================================
# Text of the model cards
# =====================================================================================


def fmt(x, f="{:.4f}"):
    """Format a number, or "–" when it is missing (None)."""
    if x is None:
        return "–"
    return f.format(x)


def dead_percent(summary):
    """The worst dead-unit fraction of a version in percent (None without diagnostics)."""
    if summary["worst_dead"] is None:
        return None
    return 100 * summary["worst_dead"]


def bold_if(text, is_best):
    """Wrap ``text`` in Markdown bold when it is the best value of its column."""
    if is_best:
        return f"**{text}**"
    return text


def resume_row(c):
    """Extra table row for a run that was stopped and resumed (empty otherwise)."""
    if "resumed_at_epoch" not in c:
        return ""
    return (f"| resumed | training stopped at epoch 146 and was resumed from the epoch-{c['resumed_at_epoch']} "
            "weights with a fresh Adam state; the cosine schedule continued from that epoch |\n")


def per_label_table(s):
    """Table of each label's MAE and R² on the validation split (empty without eval scores)."""
    ev = s["eval"]
    if ev is None:
        return ""
    row_lines = []
    for name in ev["mae"]:
        if name.endswith("thickness") or name.endswith("roughness"):
            unit = " Å"
        else:
            unit = ""
        row_lines.append(f"| {name} | {ev['mae'][name]:.3g}{unit} | {ev['r2'][name]:.3f} |")
    rows = "\n".join(row_lines)
    bin_texts = []
    for b in THICKNESS_BINS:
        bin_texts.append(f"{b} Å films: {ev['thickness_mae_by_bin'][b]:.1f} Å")
    bins = " · ".join(bin_texts)
    return f"""### Per-label accuracy ({ev['n_rows']:,} validation rows, `best.pt`)

| label | MAE | R² |
|---|---|---|
{rows}

Film-thickness MAE by film thickness: {bins}; 90% of films are within {ev['thickness_abs_err_p90']:.0f} Å.
"""


def version_card(s, repo):
    """README.md of one version's branch."""
    c = s["config"]
    if c["activation"] == "gelu":
        act = c["activation"].upper()
    else:
        act = "ReLU"
    reg = REGULARIZER_NAMES[c["regularizer"]]

    if s["epochs_run"] < s["epochs_planned"]:
        partial_note = ("> **Partial run.** Training was stopped at epoch " + str(s["epochs_run"]) + " of "
                        + str(s["epochs_planned"]) + "; the cosine schedule had not decayed, so these numbers are not converged.")
    else:
        partial_note = ""

    fig_lines = []
    for f in FIG_DOC:
        fig_lines.append(f"- `figures/{f}` — {FIG_DOC[f]}")
    figs = "\n".join(fig_lines)

    if len(s["flags"]) == 0:
        flags = "- none"
    else:
        flag_lines = []
        for f in s["flags"]:
            flag_lines.append(f"- {f}")
        flags = "\n".join(flag_lines)

    return f"""---
library_name: pytorch
tags: [refltron, x-ray-reflectivity, polarized-reflectivity, inverse-problem, mlp, variable-q]
datasets: [{DATASET}]
---

# Refltron — `{s['version']}`

{s['desc']}.

Part of [`{repo}`](https://huggingface.co/{repo}) — see `main` for the comparison across versions.
{partial_note}

## Model

*Variable q*: the q grid is fed to the network as a third input channel alongside the s- and p-polarized
reflectivity, so each curve can be sampled on its own grid (256 points).

- activation: **{act}** · input regularizer: **{reg}** · per-channel input standardization (stored in the weights as `input_mean`/`input_std`)
- output: 7 labels — {LABELS}

{ARCH}
Label bounds:

{BOUNDS_TABLE}

## Training

| setting | value |
|---|---|
| data | [`{DATASET}`](https://huggingface.co/datasets/{DATASET}), 1M simulated curves; rows 900k+ held out for validation |
| loss | hybrid: Si-aligned closed-form (erf) depth-profile L2 + normalized label MSE |
| optimizer | Adam, lr {c['lr']}, cosine decay to {c['lr'] * c['lr_min_frac']:.0e} over {c['epochs']} epochs |
| batch / seed | {c['batch_size']} / {c['seed']} |
| epochs run | {s['epochs_run']} / {s['epochs_planned']} |
| checkpoint | `best.pt` = epoch {s['best_epoch']} (lowest val loss); `last.pt` = final periodic checkpoint |
{resume_row(c)}
## Results (validation split)

| metric | value |
|---|---|
| best val loss (hybrid) | {s['best_val']:.4f} (epoch {s['best_epoch']}) |
| final train / val loss | {s['final']['train']:.4f} / {s['final']['val']:.4f} |
| film thickness MAE | {s['final']['mae']:.1f} Å |
| label MSE (normalized) | {fmt(s['label_mse'], '{:.2e}')} |
| Si-aligned depth-profile L2 | {fmt(s['profile_l2'])} |
| OOD bare oxide (truth 0 Å) → predicted film | {s['ood0_tail']:.0f} Å (mean of last 50 epochs) |
| OOD 2000 Å film (training max 800 Å) → predicted | {s['ood2000_tail']:.0f} Å |
| worst dead-unit fraction in any layer | {fmt(dead_percent(s), '{:.0f}%')} |
| fringe-detail decoder R² after mixer / after L1 | {fmt(s['fringe_mixer'], '{:.2f}')} / {fmt(s['fringe_l1'], '{:.2f}')} (≈0 from L2 on) |

{per_label_table(s)}
### Automatic diagnostics flags

{flags}

## Figures

![training](figures/training.png)

![snapshot](figures/snapshot_best.png)

![morph](figures/morph_best.png)

{figs}

## Files

- `model.safetensors` — best weights (state dict, including input standardization buffers)
- `best.pt`, `last.pt` — original PyTorch checkpoints (`model_state_dict`, `epoch`, `val_loss`, `config`)
- `config.json` — training configuration
- `network.py` — model definition
- `training_log.txt` — per-epoch train/val loss, thickness MAE and OOD predictions
- `diagnostics.json` — layer-by-layer statistics behind the figures

```python
import torch
from network import ReflectivityNet
from safetensors.torch import load_file

model = ReflectivityNet(regularizer="{c['regularizer']}", activation="{c['activation']}", normalize_inputs=True)
model.load_state_dict(load_file("model.safetensors"))
labels_norm = model(s, p, q)  # s, p, q: (batch, 256) tensors
```
"""


def selection_table(summaries, repo):
    """Front-page table: every version scored the same way on the full validation split."""
    head = ("| version | best epoch | hybrid loss ↓ | depth-profile L2 ↓ | label MSE ↓ | thickness MAE Å ↓ | "
            "roughness MAE Å ↓ | oxide MAE Å ↓ | thickness R² ↑ | train / val gap | dead units |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")

    # The best (lowest, or for R² highest) value of each column, to print it in bold.
    best = {}
    for key in ("hybrid_loss", "profile_l2", "label_mse"):
        best[key] = summaries[0]["eval"][key]
        for x in summaries:
            if x["eval"][key] < best[key]:
                best[key] = x["eval"][key]
    best_mae = {}
    for key in LENGTH_LABELS:
        best_mae[key] = summaries[0]["eval"]["mae"][key]
        for x in summaries:
            if x["eval"]["mae"][key] < best_mae[key]:
                best_mae[key] = x["eval"]["mae"][key]
    best_r2 = summaries[0]["eval"]["r2"]["film_thickness"]
    for x in summaries:
        if x["eval"]["r2"]["film_thickness"] > best_r2:
            best_r2 = x["eval"]["r2"]["film_thickness"]

    rows = []
    for x in summaries:
        e = x["eval"]
        m = e["mae"]
        cells = [
            f"[`{x['version']}`](https://huggingface.co/{repo}/tree/{x['version']})",
            f"{x['best_epoch']}",
            bold_if(f"{e['hybrid_loss']:.4f}", e["hybrid_loss"] == best["hybrid_loss"]),
            bold_if(f"{e['profile_l2']:.4f}", e["profile_l2"] == best["profile_l2"]),
            bold_if(f"{e['label_mse']:.2e}", e["label_mse"] == best["label_mse"]),
        ]
        for key in LENGTH_LABELS:
            if key == "film_thickness":
                text = f"{m[key]:.1f}"
            else:
                text = f"{m[key]:.2f}"
            cells.append(bold_if(text, m[key] == best_mae[key]))
        cells.append(bold_if(f"{e['r2']['film_thickness']:.3f}", e["r2"]["film_thickness"] == best_r2))
        cells.append(f"{x['final']['train']:.3f} / {x['final']['val']:.3f}")
        cells.append(fmt(dead_percent(x), "{:.0f}%"))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(rows)


def main_card(summaries, repo):
    """README.md of the ``main`` branch: overview and comparison of all versions."""
    bin_rows = []
    for x in summaries:
        cells = [f"`{x['version']}`"]
        for b in THICKNESS_BINS:
            cells.append(f"{x['eval']['thickness_mae_by_bin'][b]:.1f}")
        cells.append(f"{x['eval']['thickness_abs_err_p90']:.0f}")
        cells.append(f"{x['ood0_tail']:.0f}")
        cells.append(f"{x['ood2000_tail']:.0f}")
        bin_rows.append("| " + " | ".join(cells) + " |")
    by_bin = "\n".join(bin_rows)

    superseded_lines = []
    for old in SUPERSEDED:
        superseded_lines.append(f"- `{old}`: early partial upload, superseded by `{SUPERSEDED[old]}`.")
    superseded = "\n".join(superseded_lines)

    n_rows = summaries[0]["eval"]["n_rows"]
    return f"""---
library_name: pytorch
tags: [refltron, x-ray-reflectivity, polarized-reflectivity, inverse-problem, mlp, variable-q, fixed-q]
datasets: [{DATASET}]
---

# Refltron

![Refltron: reflectivity in, depth profile out](figures/hero.png)

The variable-q perceptron: a dense MLP that inverts polarized (s, p) X-ray reflectivity curves into thin-film parameters
(film thickness / roughness, oxide thickness, and the δ, β optical tensor components). *Variable q*: the
q grid is an input channel, so curves need not share a fixed grid. This repo collects the trained
versions; a fixed-q variant is the next step.

Each version lives on its own **branch**, with weights, config, training log, diagnostics and figures.

## Which version to use

**Use `v3-gelu-rq4`.** It is the best on every metric in the validation table below, and the only version without dead units.
The two ReLU versions are effectively tied with each other; pick `v2-relu-log` over `v1` only if bare-oxide
(no film) samples matter, where it predicts the smallest spurious film (see the second table).

```bash
hf download {repo} --revision v3-gelu-rq4 --local-dir refltron-v3
```

### Validation metrics

All versions scored identically with `refltron/multi_channel_mlp/analysis/eval_versions.py`: `best.pt` (lowest validation loss) on the
full held-out split ({n_rows:,} curves, dataset rows 900k+). ↓ lower is better, ↑ higher is better; best in **bold**.

{selection_table(summaries, repo)}

- **hybrid loss**: the training objective = depth-profile L2 + label MSE. The single number to rank versions by.
- **depth-profile L2**: distance between predicted and true electron-density depth profiles (Si-aligned).
- **label MSE**: mean squared error of the 7 labels, each scaled to [0, 1].
- **train / val gap**: final-epoch hybrid loss on training vs validation rows; all versions overfit by a similar
  margin, so the ranking is not an artefact of fitting.

### Thickness error by film thickness, and out-of-distribution behaviour

| version | MAE 20–100 Å films | MAE 100–400 Å | MAE 400–800 Å | 90th pct error Å | bare oxide (0 Å) → predicted | 2000 Å film → predicted |
|---|---|---|---|---|---|---|
{by_bin}

Error grows with thickness for every version (thicker films have denser Kiessig fringes). The two OOD columns are
the mean prediction over the last 50 epochs on curves outside the training range (films are 20–800 Å):
no version extrapolates past ~900 Å, and all predict a spurious film (~145–400 Å) on a bare oxide.

![benchmark](figures/benchmark.png)

## Training

![training curves](figures/training.png)

## Takeaways

- **GELU beats ReLU** at the same input regularizer: about 13% lower hybrid loss and ~4 Å lower thickness MAE,
  and it has no dead units (the ReLU versions lose roughly a third of a layer).
- **log10 vs R·q⁴** makes little difference for ReLU (v1 ≈ v2 on validation).
- Cosine LR decay removes the epoch-to-epoch jumps seen with a constant rate; validation plateaus near
  epoch 450–500 while train loss keeps falling, so longer training does not help this architecture.
- Kiessig-fringe detail is lost in the mixer → L1 → L2 stack (L1 and L2 have no activation between them and
  collapse into one rank-≤400 linear map); fringe decoder R² is ≈0 from L2 on.

Older branches kept for reference:
{superseded}

## Architecture (shared by all versions)

{ARCH}
Labels: {LABELS}.

Training: hybrid loss (Si-aligned depth-profile L2 + normalized label MSE), Adam 5e-4 with cosine decay,
batch 1024, seed 42, on [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) (rows 900k+ held out).

## Figures

- `figures/hero.png` — what the model does: one held-out curve, its recovered depth profile, and predicted vs true
  thickness over all validation curves (v3).
- `figures/benchmark.png` — the versions compared on the same 100k validation curves: losses, per-label error,
  error vs film thickness, out-of-distribution predictions.
- `figures/training.png` — training curves of all versions (v3 was resumed at epoch 143).
- Per-version figures are on each branch under `figures/` (training snapshot, curve-morph, layer health,
  saliency, evolution, per-layer cards).
"""


# =====================================================================================
# Staging the files of each branch
# =====================================================================================


def stage_version(s, repo):
    """Copy everything one version's branch holds into STAGE/<version>/. Returns that folder."""
    d = STAGE / s["version"]
    shutil.rmtree(d, ignore_errors=True)
    (d / "figures").mkdir(parents=True)
    run = ROOT / "checkpoints" / s["ckpt_id"]
    an = ROOT / "analysis/runs" / s["ckpt_id"]

    # Weights as safetensors (each tensor must be contiguous in memory), plus the original checkpoints.
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    weights = {}
    for name in ck["model_state_dict"]:
        weights[name] = ck["model_state_dict"][name].contiguous()
    save_file(weights, d / "model.safetensors", metadata={"epoch": str(ck["epoch"]), "version": s["version"]})
    shutil.copy(run / "best.pt", d / "best.pt")
    last_checkpoint = last_ckpt(run)
    if last_checkpoint is not None:
        shutil.copy(last_checkpoint, d / "last.pt")

    # Training configuration with the label names and ranges added.
    config = ck["config"].copy()
    config["label_names"] = list(LABEL_NAMES)
    config["label_bounds"] = LABEL_BOUNDS
    (d / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy(ROOT / "refltron/multi_channel_mlp/network.py", d / "network.py")

    # Only the epoch lines of the training log.
    log_lines = []
    for line in (ROOT / "logs/sweep" / f"{s['log_name']}.log").read_text(errors="ignore").splitlines():
        if EPOCH_RE.search(line):
            log_lines.append(line)
    (d / "training_log.txt").write_text("\n".join(log_lines) + "\n")

    # Diagnostics and figures, where they exist.
    if (an / "best.json").exists():
        shutil.copy(an / "best.json", d / "diagnostics.json")
    for source in FIGURE_COPIES:
        if (an / source).exists():
            shutil.copy(an / source, d / "figures" / FIGURE_COPIES[source])
    shutil.copy(HUB_FIGS / f"training_{s['version']}.png", d / "figures/training.png")
    if (an / "best_cards").is_dir():
        shutil.copytree(an / "best_cards", d / "figures/cards")
    (d / "README.md").write_text(version_card(s, repo))
    return d


def stage_main(summaries, repo):
    """Copy everything the ``main`` branch holds into STAGE/main/. Returns that folder."""
    d = STAGE / "main"
    shutil.rmtree(d, ignore_errors=True)
    (d / "figures").mkdir(parents=True)
    for name in ("hero.png", "benchmark.png", "training.png"):
        shutil.copy(HUB_FIGS / name, d / "figures" / name)
    shutil.copy(ROOT / "refltron/multi_channel_mlp/network.py", d / "network.py")
    card = main_card(summaries, repo)
    # Keep the single-channel section at the bottom (written by refltron/single_channel_mlp/publish_hf.py).
    single_channel = ROOT / "analysis/runs/single_channel"
    if (single_channel / "card_section.md").exists():
        card = card.rstrip("\n") + "\n\n" + (single_channel / "card_section.md").read_text()
        for name in ("single_channel_inputs.png", "single_channel_training.png", "single_channel_quality.png"):
            shutil.copy(single_channel / name, d / "figures" / name)
    (d / "README.md").write_text(card)
    return d


def parse_args():
    """Read command-line flags. Every flag defaults to the matching setting at the top of this file."""
    parser = argparse.ArgumentParser(
        description="Publish trained Refltron versions to the Hugging Face Hub (one branch per version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # show defaults in --help
    )
    parser.add_argument("--repo", default=REPO, help="model repository on the Hub")
    parser.add_argument("--public", action="store_true", default=PUBLIC, help="create the repo public (default private)")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN, help=f"only stage files under {STAGE}")
    return parser.parse_args()


if __name__ == "__main__":
    main()
