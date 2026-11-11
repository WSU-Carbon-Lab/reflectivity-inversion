"""Train ReflectivityNet to predict thin-film parameters from (s, p, q) curves."""

import argparse
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from loss import LABEL_NAMES, ReflectivityLoss
from metrics import MetricStore
from network import ReflectivityNet

DEFAULT_LOCAL_PARQUET = Path("data/reflectivity-1m/data")
DEFAULT_HF_PARQUET = "hf://datasets/carbon-lab/reflectivity-1m/data/train-*.parquet"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    # 1. Load the dataset and split post-hoc into training and validation sets.
    dataset = load_dataset(args.data, args.max_samples)
    n_val = int(len(dataset) * args.val_split)
    if n_val == 0:
        # When overfitting 1 sample (or --val-split 0), evaluate on the training set.
        train_set = Subset(dataset, range(len(dataset)))
        val_set = Subset(dataset, range(len(dataset)))
    else:
        n_train = len(dataset) - n_val
        train_set = Subset(dataset, range(0, n_train))
        val_set = Subset(dataset, range(n_train, len(dataset)))

    # DataLoaders group individual rows into mini-batches for training.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 2. Build the neural network, normalized loss function, and Adam optimizer.
    model = ReflectivityNet(mix_channels=args.mix_channels).to(device)
    criterion = ReflectivityLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Set up W&B tracking, checkpoints, and sample snapshot figures.
    tracker = MetricStore(model, criterion, dataset, train_set, val_set, device, args)

    # 4. Training loop: each epoch does one full pass over the training set.
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        tracker.store_metrics(epoch, train_loss, val_loss, val_mae)

    tracker.finish()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the network for one epoch and return the average normalized MSE loss."""
    model.train()
    total_loss = 0.0
    total_rows = 0

    for s, p, q, targets in loader:
        s = s.to(device)
        p = p.to(device)
        q = q.to(device)
        targets = targets.to(device)

        # Forward pass: predict [0, 1] normalized film labels from the 3 input curves.
        predictions = model(s, p, q)
        loss = criterion(predictions, targets)

        # Backward pass: clear old gradients, compute new gradients, update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_rows = s.size(0)
        total_loss += loss.item() * batch_rows
        total_rows += batch_rows

    return total_loss / total_rows


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate average validation loss and physical-unit error without updating weights."""
    model.eval()
    total_loss = 0.0
    total_mae = torch.zeros(len(LABEL_NAMES), device=device)
    total_rows = 0

    for s, p, q, targets in loader:
        s = s.to(device)
        p = p.to(device)
        q = q.to(device)
        targets = targets.to(device)

        predictions = model(s, p, q)

        batch_rows = s.size(0)
        total_loss += criterion(predictions, targets).item() * batch_rows
        total_mae += criterion.physical_mae(predictions, targets) * batch_rows
        total_rows += batch_rows

    avg_loss = total_loss / total_rows
    avg_mae = (total_mae / total_rows).cpu()
    return avg_loss, avg_mae


def load_dataset(data_path, max_samples):
    """Read Parquet shards and pack (s, p, q) curves and labels into PyTorch tensors."""
    if data_path is not None:
        source = data_path
    else:
        local_shards = sorted(DEFAULT_LOCAL_PARQUET.glob("train-*.parquet"))
        source = local_shards if local_shards else DEFAULT_HF_PARQUET

    print(f"Loading dataset from {source} ...")
    df = pl.read_parquet(source, n_rows=max_samples)

    # Copy columns out of Arrow memory into writable float32 PyTorch tensors.
    s = torch.from_numpy(df["s"].to_numpy().copy())
    p = torch.from_numpy(df["p"].to_numpy().copy())
    q = torch.from_numpy(df["q"].to_numpy().copy())
    targets = torch.from_numpy(df.select(LABEL_NAMES).to_numpy().copy())

    return TensorDataset(s, p, q, targets)


def select_device(requested):
    """Pick the best available accelerator (CUDA GPU, Apple MPS, or CPU)."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    """Parse optional command-line settings (all have defaults)."""
    parser = argparse.ArgumentParser(description="Train ReflectivityNet.")
    parser.add_argument("--data", default=None, help="Parquet path or glob")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for quick tests")
    parser.add_argument("--val-split", type=float, default=0.10, help="Validation fraction (default: 0.10)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mix-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--local", action="store_true", help="Run W&B offline and save fewer checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
