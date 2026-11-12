"""Score every published Refltron version on the full validation split, on equal terms.

    uv run python -m refltron.multi_channel_mlp.analysis.eval_versions            # writes analysis/runs/version_eval.json

For each checkpoint's ``best.pt`` this reports, over the last 10% of the dataset
(the rows no version ever trained on):
    - the hybrid loss and its two parts (depth-profile L2, normalized label MSE)
    - mean absolute error of every label in physical units, and R² per label
    - film-thickness MAE split by film thickness (thin / medium / thick films)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


from refltron.common.loss import LABEL_NAMES, DepthProfileLoss, LabelMSELoss, build_loss
from refltron.common.paths import PROJECT_ROOT as ROOT
from refltron.multi_channel_mlp.network import ReflectivityNet
from refltron.multi_channel_mlp.train import load_dataset, select_device

# The dataset has 10 shards of 100k rows; training uses the last 10% (= this shard) for
# validation, so this file is exactly the validation split.
VAL_SHARD = ROOT / "data/reflectivity-1m/data/train-00009-of-00010.parquet"

# version name -> checkpoint run id (same list as refltron/multi_channel_mlp/publish_hf.py)
RUNS = {"v1-relu-rq4": "d0z13nm4", "v2-relu-log": "pj7cue8j", "v3-gelu-rq4": "al8du562"}
THICKNESS_BINS = [(20, 100), (100, 400), (400, 800)]  # Å
OUT = ROOT / "analysis/runs/version_eval.json"
# Per-curve predictions in physical units (for the Hub figures): truth + one array per version.
PRED_OUT = ROOT / "analysis/runs/version_predictions.npz"


@torch.no_grad()
def evaluate_checkpoint(path, loader, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ReflectivityNet(
        regularizer=cfg["regularizer"], activation=cfg["activation"],
        normalize_inputs=cfg.get("normalize_inputs", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    hybrid = build_loss("hybrid").to(device)
    profile = DepthProfileLoss().to(device)
    mse = LabelMSELoss().to(device)

    preds = []
    targets = []
    hybrid_sum = 0.0
    profile_sum = 0.0
    mse_sum = 0.0
    n_rows = 0
    for s, p, q, y in loader:
        s = s.to(device)
        p = p.to(device)
        q = q.to(device)
        y = y.to(device)
        out = model(s, p, q)
        n = len(s)
        hybrid_sum += hybrid(out, y, q).item() * n
        profile_sum += profile(out, y, q).item() * n
        mse_sum += mse(out, y, q).item() * n
        preds.append(mse.denormalize(out).cpu().double())
        targets.append(y.cpu().double())
        n_rows += n

    pred = torch.cat(preds)
    true = torch.cat(targets)
    err = (pred - true).abs()
    # R^2 per label: 1 - (squared error) / (spread of the truth around its mean).
    r2 = 1 - ((pred - true) ** 2).sum(0) / ((true - true.mean(0)) ** 2).sum(0)
    thick_err = err[:, 0]
    by_bin = {}
    for bin_range in THICKNESS_BINS:
        lo = bin_range[0]
        hi = bin_range[1]
        # "&" means "and", applied to every row: films with lo <= thickness < hi.
        in_bin = (true[:, 0] >= lo) & (true[:, 0] < hi)
        by_bin[f"{lo}-{hi}"] = float(thick_err[in_bin].mean())

    mae_values = err.mean(0).tolist()
    r2_values = r2.tolist()
    mae = {}
    r2_by_label = {}
    for i in range(len(LABEL_NAMES)):
        mae[LABEL_NAMES[i]] = mae_values[i]
        r2_by_label[LABEL_NAMES[i]] = r2_values[i]
    summary = {
        "checkpoint": str(path.relative_to(ROOT)),
        "epoch": ckpt["epoch"],
        "n_rows": n_rows,
        "hybrid_loss": hybrid_sum / n_rows,
        "profile_l2": profile_sum / n_rows,
        "label_mse": mse_sum / n_rows,
        "mae": mae,
        "r2": r2_by_label,
        "thickness_mae_by_bin": by_bin,
        "thickness_abs_err_p90": float(thick_err.quantile(0.9)),
    }
    return summary, pred.float().numpy(), true.float().numpy()


def main():
    device = select_device("auto")
    val_set = load_dataset(VAL_SHARD)
    loader = DataLoader(val_set, batch_size=4096, shuffle=False)
    print(f"Validation rows: {len(val_set)}")

    results = {}
    arrays = {}  # per-curve predictions: "truth" plus one array per version
    for version in RUNS:
        run = RUNS[version]
        summary, predictions, truth = evaluate_checkpoint(ROOT / "checkpoints" / run / "best.pt", loader, device)
        results[version] = summary
        arrays[version] = predictions
        arrays["truth"] = truth
        r = results[version]
        print(f"{version}: hybrid {r['hybrid_loss']:.4f} | profile {r['profile_l2']:.4f} | "
              f"mse {r['label_mse']:.2e} | thickness MAE {r['mae']['film_thickness']:.1f} Å")
    OUT.write_text(json.dumps(results, indent=2))
    # savez stores each keyword argument as a named array. The "**arrays" turns the
    # dictionary into keyword arguments, so the file gets one array per key (truth, v1-..., ...).
    np.savez_compressed(PRED_OUT, label_names=np.array(LABEL_NAMES), **arrays)
    print(f"wrote {OUT} and {PRED_OUT}")


if __name__ == "__main__":
    main()
