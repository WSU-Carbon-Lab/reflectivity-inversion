"""Publish trained runs to the Hugging Face Hub as versions (branches) of one model repo.

Each version gets its own branch holding weights, config, training log, diagnostics and
figures. ``main`` holds the overview card and the cross-version comparison.

    uv run --with safetensors python scripts/publish_hf.py [--repo carbon-lab/refltron] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss import LABEL_NAMES, LabelMSELoss  # noqa: E402

_crit = LabelMSELoss()
LABEL_BOUNDS = {n: (float(lo), float(hi)) for n, lo, hi in zip(LABEL_NAMES, _crit.label_min.tolist(), _crit.label_max.tolist())}
BOUNDS_TABLE = "| label | min | max |\n|---|---|---|\n" + "\n".join(f"| {n} | {lo:.4g} | {hi:.4g} |" for n, (lo, hi) in LABEL_BOUNDS.items())

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path("/tmp/hf_stage")
DATASET = "carbon-lab/reflectivity-1m"

# version branch, checkpoint id, sweep log name, short description
VERSIONS = [
    ("v1-relu-rq4", "d0z13nm4", "relu-rq4-norm", "ReLU, R·q⁴ input regularizer, standardized inputs, 600 epochs"),
    ("v2-relu-log", "pj7cue8j", "relu-log-norm", "ReLU, log10 input regularizer, standardized inputs, 600 epochs"),
    ("v3-gelu-rq4-e146", "al8du562", "gelu-rq4-norm",
     "GELU, R·q⁴ input regularizer, standardized inputs — stopped at epoch 146 of 600 (partial)"),
]
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+ \| train\(\w+\) ([\d.e+-]+) \| val ([\d.e+-]+) \| thickness MAE ([\d.]+) Å \| "
                      r"OOD 0Å->([-\d.]+) Å \| OOD 2000Å->([-\d.]+) Å")

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

FIG_DOC = [
    ("snapshot_best.png", "Training snapshot at the best epoch: s/p reflectivity, Si-aligned depth profile "
     "(height above the Si interface, substrate at the right; solid = truth, dashed = network), and normalized "
     "labels, for one training row, one validation row and two out-of-distribution curves (bare oxide, 2000 Å film)."),
    ("morph_best.png", "How the reflectivity curve morphs through the network. Each row is one stage; the left three "
     "columns show what a ridge decoder can reconstruct of three validation curves (80 / 300 / 650 Å films) from "
     "that stage's activations, the fourth gives per-q decoder R² (whole curve and high-pass fringe detail), the fifth "
     "a PCA of the stage coloured by film thickness. Bottom row: predicted labels re-simulated back into curves."),
    ("layer_health_best.png", "Per-layer health: dead / always-on units, sparsity, gain (RMS out / in), spike ratio, "
     "effective rank, redundancy, linear-probe R² for each label, and weight spectra."),
    ("saliency_best.png", "Input sensitivity: |∂label/∂input| × input std across q for each label, and mixer "
     "channel usage."),
    ("evolution.png", "Layer health across the periodic checkpoints of the run."),
    ("cards/", "One standardized card per layer: receptive fields over q, firing-rate distribution, probe R² per "
     "label, activation PCA, and weight spectrum."),
]


def parse_log(path: Path):
    rows = [m.groups() for m in map(EPOCH_RE.search, path.read_text(errors="ignore").splitlines()) if m]
    return [dict(epoch=int(r[0]), train=float(r[1]), val=float(r[2]), mae=float(r[3]),
                 ood0=float(r[4]), ood2000=float(r[5])) for r in rows]


def last_ckpt(run: Path):
    eps = sorted(run.glob("epoch_*.pt"))
    return eps[-1] if eps else None


def summarize(version, ckpt_id, log_name, desc):
    run = ROOT / "checkpoints" / ckpt_id
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    log = parse_log(ROOT / "logs/sweep" / f"{log_name}.log")
    diag_p = ROOT / "analysis/runs" / ckpt_id / "best.json"
    diag = json.loads(diag_p.read_text()) if diag_p.exists() else {}
    best = min(log, key=lambda r: r["val"])
    tail = log[-50:]
    mean = lambda k: sum(r[k] for r in tail) / len(tail)  # noqa: E731
    vm = diag.get("val_metrics", {})
    morph = diag.get("morph", {})
    fr = {s: r for s, r in zip(morph.get("stages", []), morph.get("fringe_r2", []))}
    return dict(
        version=version, ckpt_id=ckpt_id, log_name=log_name, desc=desc, config=ck["config"],
        best_epoch=ck["epoch"], best_val=ck["val_loss"], epochs_run=log[-1]["epoch"],
        epochs_planned=ck["config"]["epochs"], log_best_val=best["val"], log_best_epoch=best["epoch"],
        final=log[-1], ood0_tail=mean("ood0"), ood2000_tail=mean("ood2000"),
        label_mse=vm.get("label_mse"), profile_l2=vm.get("profile_l2"), val_mae=vm.get("mae"),
        worst_dead=max((l["dead_frac"] for l in diag.get("layers", [])), default=None),
        fringe_l1=next((v for s, v in fr.items() if s.startswith("L1 ")), None),
        fringe_mixer=next((v for s, v in fr.items() if s.startswith("mixer")), None),
        flags=diag.get("flags", []),
    )


def fmt(x, f="{:.4f}"):
    return "–" if x is None else f.format(x)


def version_card(s, repo):
    c = s["config"]
    act = c["activation"].upper() if c["activation"] == "gelu" else "ReLU"
    reg = {"log": "log10(R)", "rq4": "R·q⁴", "log_rq4": "log10(R·q⁴)"}[c["regularizer"]]
    partial = s["epochs_run"] < s["epochs_planned"]
    figs = "\n".join(f"- `figures/{f}` — {d}" for f, d in FIG_DOC)
    flags = "\n".join(f"- {f}" for f in s["flags"]) or "- none"
    return f"""---
library_name: pytorch
tags: [refltron, x-ray-reflectivity, polarized-reflectivity, inverse-problem, mlp, dynamic-q]
datasets: [{DATASET}]
---

# Refltron — `{s['version']}`

{s['desc']}.

Part of [`{repo}`](https://huggingface.co/{repo}) — see `main` for the comparison across versions.
{"> **Partial run.** Training was stopped at epoch " + str(s['epochs_run']) + " of " + str(s['epochs_planned']) + "; the cosine schedule had not decayed, so these numbers are not converged." if partial else ""}

## Model

*Dynamic q*: the q grid is fed to the network as a third input channel alongside the s- and p-polarized
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
| worst dead-unit fraction in any layer | {fmt(s['worst_dead'] and 100 * s['worst_dead'], '{:.0f}%')} |
| fringe-detail decoder R² after mixer / after L1 | {fmt(s['fringe_mixer'], '{:.2f}')} / {fmt(s['fringe_l1'], '{:.2f}')} (≈0 from L2 on) |

### Automatic diagnostics flags

{flags}

## Figures

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


def main_card(summaries, repo):
    rows = "\n".join(
        f"| [`{s['version']}`](https://huggingface.co/{repo}/tree/{s['version']}) | {s['config']['activation']} | "
        f"{s['config']['regularizer']} | {s['epochs_run']}/{s['epochs_planned']} | {s['best_val']:.4f} | "
        f"{s['final']['mae']:.1f} | {fmt(s['profile_l2'])} | {s['ood0_tail']:.0f} | {s['ood2000_tail']:.0f} | "
        f"{fmt(s['worst_dead'] and 100 * s['worst_dead'], '{:.0f}%')} |"
        for s in summaries)
    return f"""---
library_name: pytorch
tags: [refltron, x-ray-reflectivity, polarized-reflectivity, inverse-problem, mlp, dynamic-q]
datasets: [{DATASET}]
---

# Refltron

The dynamic-q perceptron: a dense MLP that inverts polarized (s, p) X-ray reflectivity curves into thin-film parameters
(film thickness / roughness, oxide thickness, and the δ, β optical tensor components). *Dynamic q*: the
q grid is an input channel, so curves need not share a fixed grid. This repo collects the trained
versions; a fixed-q variant is the next step.

Each version lives on its own **branch**, with weights, config, training log, diagnostics and figures:

| version | activation | input reg. | epochs | best val loss | thickness MAE Å | profile L2 | OOD 0 Å → | OOD 2000 Å → | worst dead units |
|---|---|---|---|---|---|---|---|---|---|
{rows}

```bash
hf download {repo} --revision v2-relu-log --local-dir refltron-v2
```

![comparison](figures/sweep_comparison.png)

## Takeaways

- The two finished ReLU versions are effectively tied on validation; **v2 (log10)** is the better finished
  model, mainly because it handles a bare oxide far better (~145 Å vs ~400 Å predicted for a 0 Å film).
- **v3 (GELU)** beat both by epoch 146 and has no dead units, but was stopped early — not converged.
- Cosine LR decay removes the epoch-to-epoch jumps seen with a constant rate; validation plateaus near
  epoch 450 while train loss keeps falling, so longer training does not help this architecture.
- All versions saturate near the 800 Å training maximum on a 2000 Å film (no extrapolation).
- Kiessig-fringe detail is lost in the mixer → L1 → L2 stack (L1 and L2 have no activation between them and
  collapse into one rank-≤400 linear map); fringe decoder R² is ≈0 from L2 on.

## Architecture (shared by all versions)

{ARCH}
Labels: {LABELS}.

Training: hybrid loss (Si-aligned depth-profile L2 + normalized label MSE), Adam 5e-4 with cosine decay,
batch 1024, seed 42, on [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) (rows 900k+ held out).

## Figures

- `figures/sweep_comparison.png` — loss, thickness MAE, generalization gap, OOD predictions per epoch, and how much
  of the input curve each layer still encodes.
- Per-version figures are on each branch under `figures/` (training snapshot, curve-morph, layer health,
  saliency, evolution, per-layer cards).
"""


def stage_version(s, repo):
    d = STAGE / s["version"]
    shutil.rmtree(d, ignore_errors=True)
    (d / "figures").mkdir(parents=True)
    run = ROOT / "checkpoints" / s["ckpt_id"]
    an = ROOT / "analysis/runs" / s["ckpt_id"]
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    save_file({k: v.contiguous() for k, v in ck["model_state_dict"].items()}, d / "model.safetensors",
              metadata={"epoch": str(ck["epoch"]), "version": s["version"]})
    shutil.copy(run / "best.pt", d / "best.pt")
    if (lc := last_ckpt(run)) is not None:
        shutil.copy(lc, d / "last.pt")
    (d / "config.json").write_text(json.dumps({**ck["config"], "label_names": list(LABEL_NAMES), "label_bounds": LABEL_BOUNDS}, indent=2))
    shutil.copy(ROOT / "scripts/network.py", d / "network.py")
    log = [l for l in (ROOT / "logs/sweep" / f"{s['log_name']}.log").read_text(errors="ignore").splitlines()
           if EPOCH_RE.search(l)]
    (d / "training_log.txt").write_text("\n".join(log) + "\n")
    if (an / "best.json").exists():
        shutil.copy(an / "best.json", d / "diagnostics.json")
    for src, dst in [("best_snapshot.png", "snapshot_best.png"), ("best_morph.png", "morph_best.png"),
                     ("best_layer_health.png", "layer_health_best.png"), ("best_saliency.png", "saliency_best.png"),
                     ("evolution.png", "evolution.png")]:
        if (an / src).exists():
            shutil.copy(an / src, d / "figures" / dst)
    if (an / "best_cards").is_dir():
        shutil.copytree(an / "best_cards", d / "figures/cards")
    (d / "README.md").write_text(version_card(s, repo))
    return d


def stage_main(summaries, repo):
    d = STAGE / "main"
    shutil.rmtree(d, ignore_errors=True)
    (d / "figures").mkdir(parents=True)
    shutil.copy(ROOT / "analysis/runs/sweep_comparison.png", d / "figures/sweep_comparison.png")
    shutil.copy(ROOT / "scripts/network.py", d / "network.py")
    (d / "README.md").write_text(main_card(summaries, repo))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="carbon-lab/refltron")
    ap.add_argument("--public", action="store_true", help="Create the repo public (default private)")
    ap.add_argument("--dry-run", action="store_true", help="Only stage files under /tmp/hf_stage")
    args = ap.parse_args()

    summaries = [summarize(*v) for v in VERSIONS]
    main_dir = stage_main(summaries, args.repo)
    vdirs = [(s, stage_version(s, args.repo)) for s in summaries]
    print(f"staged under {STAGE}")
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=main_dir, commit_message="Overview card and version comparison")
    for s, d in vdirs:
        api.create_branch(args.repo, branch=s["version"], exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=d, revision=s["version"],
                          commit_message=f"{s['version']}: {s['desc']}")
        print(f"uploaded {s['version']}")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
