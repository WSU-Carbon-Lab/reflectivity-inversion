"""Re-render the 4-column training snapshot (train / val / OOD bare oxide / OOD 2000 Å) for a checkpoint.

Uses the current plotting style without re-running training or touching W&B:

    uv run python scripts/render_snapshot.py checkpoints/<run>/best.pt [--out fig.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from diagnostics import load_model  # noqa: E402
from loss import LABEL_NAMES, DepthProfileLoss  # noqa: E402
from metrics import MetricStore, build_ood_test_cases, estimate_substrate_sld  # noqa: E402

SHARDS = PROJECT_ROOT / "data/reflectivity-1m/data"
ROWS_PER_SHARD = 100_000


def load_row(idx: int):
    shard = SHARDS / f"train-{idx // ROWS_PER_SHARD:05d}-of-00010.parquet"
    df = pl.read_parquet(shard).slice(idx % ROWS_PER_SHARD, 1)
    t = lambda c: torch.from_numpy(df[c].to_numpy().copy())[0]  # noqa: E731
    return t("s"), t("p"), t("q"), torch.from_numpy(df.select(LABEL_NAMES).to_numpy().copy())[0]


@torch.no_grad()
def render(ckpt_path: Path, out: Path, train_loss: float = float("nan")):
    model, criterion, meta = load_model(ckpt_path)
    model.eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    tr_i, va_i = cfg["train_sample_idx"], cfg["val_sample_idx"]
    tr, va = load_row(tr_i), load_row(va_i)
    ood = build_ood_test_cases()

    # A MetricStore shell: only the attributes _save_snapshot_plot reads (no W&B init).
    ms = object.__new__(MetricStore)
    ms.loss_name = cfg.get("loss", "profile")
    ms.column_titles = [f"Training (Row #{tr_i})", f"Validation (Row #{va_i})", ood[0]["title"], ood[1]["title"]]
    ms.column_bare_flags = [False, False, ood[0]["bare_oxide"], ood[1]["bare_oxide"]]
    ms.column_slds = [estimate_substrate_sld(tr[2][-1].item()), estimate_substrate_sld(va[2][-1].item()),
                      ood[0]["sld"], ood[1]["sld"]]
    ms.sample_s = torch.stack((tr[0], va[0], ood[0]["s"], ood[1]["s"]))
    ms.sample_p = torch.stack((tr[1], va[1], ood[0]["p"], ood[1]["p"]))
    ms.sample_q = torch.stack((tr[2], va[2], ood[0]["q"], ood[1]["q"]))
    ms.sample_targets = torch.stack((tr[3], va[3], ood[0]["target"], ood[1]["target"]))

    pred_norm = model(ms.sample_s, ms.sample_p, ms.sample_q)
    target_norm = criterion.normalize(ms.sample_targets)
    pred_phys = criterion.denormalize(pred_norm)
    profile_losses = DepthProfileLoss().per_sample_loss(
        pred_norm, ms.sample_targets, substrate=torch.tensor(ms.column_slds, dtype=torch.float32)
    ).tolist()
    val_loss = ck.get("val_loss", float("nan"))
    ms._save_snapshot_plot(out, ck["epoch"], train_loss, val_loss,
                           target_norm, pred_norm, pred_phys, profile_losses)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--train-loss", type=float, default=float("nan"), help="For the title (not stored in the checkpoint)")
    args = ap.parse_args()
    render(args.checkpoint, args.out or args.checkpoint.with_suffix(".png"), args.train_loss)


if __name__ == "__main__":
    main()
