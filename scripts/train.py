"""Train the network to predict thin-film parameters from reflectivity curves.

    uv run python scripts/train.py                   # use the defaults below
    uv run python scripts/train.py --epochs 20 --max-samples 50000 --local   # quick test
    uv run python scripts/train.py --help            # list every option

Every option has a default in the DEFAULT SETTINGS block. Edit a value there to change
it permanently, or pass the matching command-line flag to change it for one run.

What happens:
    1. Load the dataset and split it into training and validation rows.
    2. Build the network, the loss function and the optimizer.
    3. Repeat for EPOCHS epochs (one epoch = one pass over all training rows):
         - train: predict, measure the loss, nudge the weights to lower it
         - validate: measure the loss on rows the network never trains on
         - log the numbers and save checkpoints / snapshot figures (metrics.py)
"""

import argparse
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from loss import LABEL_NAMES, LOSS_CHOICES, DepthProfileLoss, LabelMSELoss, build_loss
from metrics import MetricStore
from network import ACTIVATION_CHOICES, REGULARIZER_CHOICES, ReflectivityNet

# =====================================================================================
# DEFAULT SETTINGS: edit these, or override any of them from the command line.
# These defaults are the best finished Refltron configuration (v2-relu-log).
# =====================================================================================

# --- Data ---
DATA_PATH = None            # Parquet file or glob. None = local copy if present, else Hugging Face
LOCAL_DATA_DIR = Path("data/reflectivity-1m/data")
HF_DATA_GLOB = "hf://datasets/carbon-lab/reflectivity-1m/data/train-*.parquet"
MAX_SAMPLES = None          # use only the first N rows (None = all 1M); handy for quick tests
VAL_SPLIT = 0.10            # fraction of rows held out for validation (the last 10%)

# --- Network (see network.py) ---
REGULARIZER = "log"         # input transform for s and p: "log", "rq4" or "log_rq4"
ACTIVATION = "relu"         # "relu" or "gelu"
NORMALIZE_INPUTS = True     # standardize each input channel with training-set statistics
NORMALIZATION_ROWS = 50_000 # how many training rows to measure those statistics on
MIX_CHANNELS = 8            # channels made by the 1x1 convolution mixer

# --- Training ---
LOSS = "hybrid"             # "hybrid", "profile" or "mse" (see loss.py)
EPOCHS = 600
BATCH_SIZE = 1024           # rows per optimizer step
LEARNING_RATE = 5e-4        # Adam step size at the start of training
LR_SCHEDULE = "cosine"      # "cosine" (decay smoothly to LR_MIN_FRACTION * LEARNING_RATE) or "constant"
LR_MIN_FRACTION = 0.01
SEED = 42                   # random seed, so runs are repeatable
DEVICE = "auto"             # "auto" picks CUDA, then Apple MPS, then CPU

# --- Logging and checkpoints (see metrics.py) ---
LOCAL = False               # True = keep W&B offline and save fewer checkpoints
RUN_NAME = None             # optional W&B run name
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_EVERY = None     # save a numbered checkpoint every N epochs (None = 20 with LOCAL, else 5)
MAX_CHECKPOINTS = 3         # keep only the newest N numbered checkpoints (best.pt is always kept)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device} | regularizer={args.regularizer} | activation={args.activation} | loss={args.loss}")

    # 1. Data
    dataset = load_dataset(args.data, args.max_samples)
    train_set, val_set = split_dataset(dataset, args.val_split)
    # A DataLoader cuts the rows into mini-batches. Shuffling the training rows each
    # epoch means every step sees a different random mix of samples.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 2. Network, loss and optimizer
    model = build_model(args, dataset, train_set).to(device)
    criterion = build_loss(args.loss).to(device)
    # Adam is a gradient-descent optimizer that adapts the step size for each weight.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_schedule == "cosine":
        # Lower the learning rate along half a cosine wave over the whole run. Large early
        # steps make fast progress; small late steps settle into a good minimum instead
        # of bouncing around it.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_min_frac
        )

    # Whatever loss we train on, also report the profile distance and the label MSE,
    # so runs trained with different losses can be compared on the same scale.
    profile_metric = DepthProfileLoss().to(device)
    mse_metric = LabelMSELoss().to(device)

    # Logging, checkpoints and snapshot figures.
    tracker = MetricStore(model, criterion, dataset, train_set, val_set, device, args)

    # 3. Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, profile_metric, mse_metric, device)
        tracker.store_metrics(
            epoch, train_loss, val["loss"], val["mae"], val["profile_l2"], val["mse"],
            lr=optimizer.param_groups[0]["lr"],
        )
        if scheduler is not None:
            scheduler.step()

    tracker.finish()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training rows. Returns the average training loss."""
    model.train()  # training mode (matters for layers like dropout; harmless here)
    total_loss, total_rows = 0.0, 0

    for s, p, q, targets in loader:
        s, p, q, targets = s.to(device), p.to(device), q.to(device), targets.to(device)

        # Forward pass: predict the normalized labels and measure how wrong they are.
        predictions = model(s, p, q)
        loss = criterion(predictions, targets, q)

        # Backward pass: compute the gradient of the loss with respect to every weight,
        # then let the optimizer take one step downhill.
        optimizer.zero_grad()  # clear the gradients left over from the previous step
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(s)
        total_rows += len(s)

    return total_loss / total_rows


@torch.no_grad()  # no gradients needed: we only measure, we do not train
def evaluate(model, loader, criterion, profile_metric, mse_metric, device):
    """Measure the model on validation rows.

    Returns a dict with the training loss ("loss"), the two reference metrics
    ("profile_l2", "mse") and the mean absolute error of each label in physical
    units ("mae", shape (7,)).
    """
    model.eval()
    sums = {"loss": 0.0, "profile_l2": 0.0, "mse": 0.0}
    mae_sum = torch.zeros(len(LABEL_NAMES), device=device)
    total_rows = 0

    for s, p, q, targets in loader:
        s, p, q, targets = s.to(device), p.to(device), q.to(device), targets.to(device)
        predictions = model(s, p, q)
        n = len(s)
        sums["loss"] += criterion(predictions, targets, q).item() * n
        sums["profile_l2"] += profile_metric(predictions, targets, q).item() * n
        sums["mse"] += mse_metric(predictions, targets, q).item() * n
        mae_sum += criterion.physical_mae(predictions, targets) * n
        total_rows += n

    results = {name: value / total_rows for name, value in sums.items()}
    results["mae"] = (mae_sum / total_rows).cpu()
    return results


def load_dataset(data_path=None, max_samples=None):
    """Read the Parquet files into one TensorDataset of (s, p, q, labels) rows."""
    if data_path is None:
        local_files = sorted(LOCAL_DATA_DIR.glob("train-*.parquet"))
        if local_files:
            data_path = local_files
            print(f"Loading {len(local_files)} Parquet files from {LOCAL_DATA_DIR} ...")
        else:
            data_path = HF_DATA_GLOB
            print(f"Loading dataset from {HF_DATA_GLOB} ...")
    else:
        print(f"Loading dataset from {data_path} ...")
    df = pl.read_parquet(data_path, n_rows=max_samples)

    # Each of s, p and q is stored as a fixed-length array per row, so to_numpy() gives
    # a 2-D array of shape (rows, 256). .copy() makes it writable for PyTorch.
    s = torch.from_numpy(df["s"].to_numpy().copy())
    p = torch.from_numpy(df["p"].to_numpy().copy())
    q = torch.from_numpy(df["q"].to_numpy().copy())
    labels = torch.from_numpy(df.select(LABEL_NAMES).to_numpy().copy())
    return TensorDataset(s, p, q, labels)


def split_dataset(dataset, val_split):
    """Use the first rows for training and the last ``val_split`` fraction for validation.

    The rows are already in random order, so the split is random too, but always
    the same, which keeps runs comparable. With val_split = 0 (for example when
    over-fitting a single sample on purpose) the training rows are also used to validate.
    """
    n_val = int(len(dataset) * val_split)
    if n_val == 0:
        everything = Subset(dataset, range(len(dataset)))
        return everything, everything
    n_train = len(dataset) - n_val
    return Subset(dataset, range(n_train)), Subset(dataset, range(n_train, len(dataset)))


def build_model(args, dataset, train_set):
    """Create the network and, if requested, measure the input statistics it standardizes with."""
    model = ReflectivityNet(
        mix_channels=args.mix_channels,
        regularizer=args.regularizer,
        activation=args.activation,
        normalize_inputs=args.normalize_inputs,
    )
    if args.normalize_inputs:
        rows = torch.tensor(train_set.indices[:NORMALIZATION_ROWS])  # training rows only
        s, p, q, _ = dataset.tensors
        model.fit_input_normalization(s[rows], p[rows], q[rows])
        print(f"Input mean (S, P, q): {model.input_mean.squeeze().tolist()}")
        print(f"Input std  (S, P, q): {model.input_std.squeeze().tolist()}")
    return model


def select_device(requested):
    """Use the requested device, or with "auto" the fastest one available."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple-silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    """Read command-line flags. Every flag defaults to the matching setting at the top of this file."""
    parser = argparse.ArgumentParser(
        description="Train ReflectivityNet (Refltron).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # show defaults in --help
    )
    on_off = argparse.BooleanOptionalAction  # gives both --flag and --no-flag

    data = parser.add_argument_group("data")
    data.add_argument("--data", default=DATA_PATH, help="Parquet path or glob; None = local copy, else Hugging Face")
    data.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="use only the first N rows")
    data.add_argument("--val-split", type=float, default=VAL_SPLIT, help="validation fraction")

    net = parser.add_argument_group("network")
    net.add_argument("--regularizer", choices=REGULARIZER_CHOICES, default=REGULARIZER, help="input transform for s and p")
    net.add_argument("--activation", choices=ACTIVATION_CHOICES, default=ACTIVATION, help="non-linearity between layers")
    net.add_argument("--normalize-inputs", action=on_off, default=NORMALIZE_INPUTS, help="standardize input channels")
    net.add_argument("--mix-channels", type=int, default=MIX_CHANNELS, help="channels out of the 1x1 conv mixer")

    train = parser.add_argument_group("training")
    train.add_argument("--loss", choices=LOSS_CHOICES, default=LOSS, help="training objective (see loss.py)")
    train.add_argument("--epochs", type=int, default=EPOCHS, help="passes over the training rows")
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="rows per optimizer step")
    train.add_argument("--lr", type=float, default=LEARNING_RATE, help="initial learning rate")
    train.add_argument("--lr-schedule", choices=("cosine", "constant"), default=LR_SCHEDULE, help="how the learning rate changes over the run")
    train.add_argument("--lr-min-frac", type=float, default=LR_MIN_FRACTION, help="cosine floor, as a fraction of --lr")
    train.add_argument("--seed", type=int, default=SEED, help="random seed")
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=DEVICE, help="where to compute")

    log = parser.add_argument_group("logging and checkpoints")
    log.add_argument("--local", action=on_off, default=LOCAL, help="W&B offline, fewer checkpoints")
    log.add_argument("--run-name", default=RUN_NAME, help="W&B run name")
    log.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR, help="where checkpoints are written")
    log.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY, help="epochs between numbered checkpoints")
    log.add_argument("--max-checkpoints", type=int, default=MAX_CHECKPOINTS, help="numbered checkpoints to keep")
    return parser.parse_args()


if __name__ == "__main__":
    main()
