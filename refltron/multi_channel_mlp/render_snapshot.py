"""Re-render the 4-column training snapshot (train / val / OOD bare oxide / OOD 2000 Å) for a checkpoint.

Uses the current plotting style without re-running training or touching W&B:

    uv run python -m refltron.multi_channel_mlp.render_snapshot checkpoints/<run>/best.pt [--out fig.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch

from refltron.common.loss import LABEL_NAMES, DepthProfileLoss
from refltron.common.paths import PROJECT_ROOT
from refltron.multi_channel_mlp.analysis.diagnostics import load_model
from refltron.common.depth_profile import estimate_substrate_sld
from refltron.multi_channel_mlp.metrics import build_ood_test_cases, save_snapshot_figure

SHARDS = PROJECT_ROOT / "data/reflectivity-1m/data"
ROWS_PER_SHARD = 100_000


def load_row(idx: int):
    """Read dataset row ``idx`` and return its (s, p, q, labels) tensors."""
    # Row idx lives in shard idx // 100000, at position idx % 100000 inside it.
    shard = SHARDS / f"train-{idx // ROWS_PER_SHARD:05d}-of-00010.parquet"
    df = pl.read_parquet(shard).slice(idx % ROWS_PER_SHARD, 1)
    s = torch.from_numpy(df["s"].to_numpy().copy())[0]
    p = torch.from_numpy(df["p"].to_numpy().copy())[0]
    q = torch.from_numpy(df["q"].to_numpy().copy())[0]
    labels = torch.from_numpy(df.select(LABEL_NAMES).to_numpy().copy())[0]
    return s, p, q, labels


@torch.no_grad()
def render(ckpt_path: Path, out: Path, train_loss: float = float("nan")):
    model, criterion, meta = load_model(ckpt_path)
    model.eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tr_i = cfg["train_sample_idx"]
    va_i = cfg["val_sample_idx"]
    tr = load_row(tr_i)  # (s, p, q, labels)
    va = load_row(va_i)
    ood = build_ood_test_cases()

    # The same 4 columns as during training: training row, validation row, two OOD cases.
    column_titles = [f"Training (Row #{tr_i})", f"Validation (Row #{va_i})", ood[0]["title"], ood[1]["title"]]
    column_bare_flags = [False, False, ood[0]["bare_oxide"], ood[1]["bare_oxide"]]
    column_slds = [estimate_substrate_sld(tr[2]), estimate_substrate_sld(va[2]), ood[0]["sld"], ood[1]["sld"]]
    sample_s = torch.stack((tr[0], va[0], ood[0]["s"], ood[1]["s"]))
    sample_p = torch.stack((tr[1], va[1], ood[0]["p"], ood[1]["p"]))
    sample_q = torch.stack((tr[2], va[2], ood[0]["q"], ood[1]["q"]))
    sample_targets = torch.stack((tr[3], va[3], ood[0]["target"], ood[1]["target"]))

    pred_norm = model(sample_s, sample_p, sample_q)
    target_norm = criterion.normalize(sample_targets)
    pred_phys = criterion.denormalize(pred_norm)
    profile_losses = DepthProfileLoss().per_sample_loss(
        pred_norm, sample_targets, substrate=torch.tensor(column_slds, dtype=torch.float32)
    ).tolist()
    val_loss = ck.get("val_loss", float("nan"))
    save_snapshot_figure(
        out, ck["epoch"], cfg.get("loss", "profile"), train_loss, val_loss,
        column_titles, column_bare_flags, column_slds,
        sample_s, sample_p, sample_q, sample_targets,
        target_norm, pred_norm, pred_phys, profile_losses,
    )
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--train-loss", type=float, default=float("nan"), help="For the title (not stored in the checkpoint)")
    args = ap.parse_args()
    if args.out is None:
        out = args.checkpoint.with_suffix(".png")  # next to the checkpoint, same name
    else:
        out = args.out
    render(args.checkpoint, out, args.train_loss)


if __name__ == "__main__":
    main()
