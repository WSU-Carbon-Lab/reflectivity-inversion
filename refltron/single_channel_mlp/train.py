"""Train the single-channel MLP (OldNet) to predict film parameters from one fixed-q curve.

    uv run python -m refltron.single_channel_mlp.train --input s            # log(S) -> 5 labels, isotropic films
    uv run python -m refltron.single_channel_mlp.train --input p            # log(P) -> 7 labels, uniaxial films
    uv run python -m refltron.single_channel_mlp.train --input anisotropy   # A      -> 7 labels, uniaxial films
    uv run python -m refltron.single_channel_mlp.train --input s --epochs 1 --max-samples 50000   # quick test
    uv run python -m refltron.single_channel_mlp.train --help               # list every option

Every option has a default in the DEFAULT SETTINGS block. Edit a value there to change
it permanently, or pass the matching command-line flag to change it for one run.

The three inputs:
    s           log(R_s) on the fixed-q-isotropic data. An isotropic film has one delta and
                one beta, so there are 5 labels instead of 7.
    p           log(R_p) on the fixed-q (uniaxial) data, 7 labels. s-polarized light only
                feels the in-plane (xx) optical constants; p-polarized light feels both
                xx and zz, so P alone might recover both.
    anisotropy  A = (P - S c) / (P + S c) with c = cos^2(2 theta), on the fixed-q data, 7 labels.
                For a film with n close to 1, r_p ~ r_s cos(2 theta), so P ~ S c. A removes
                that purely geometric part and is ~0 for an isotropic film, so what is left
                is the anisotropy. P > 0, so the denominator never reaches 0 and A always
                stays between -1 and +1. Near the Brewster angle (theta ~ 45 deg, the top of
                the grid) c -> 0 and P dips, so A swings toward +1 or -1 there.

What happens (the original Keras recipe, see OldNet in network.py):
    1. Load the dataset, make the input curve, scale every label to [0, 1], and hold out
       the LAST 20% of rows for validation (as Keras' validation_split does).
    2. Build the network, the loss (MSE) and the optimizer (Adam, lr 5e-4).
    3. Repeat for EPOCHS epochs (one epoch = one pass over all training rows):
         - train: predict, measure the loss, nudge the weights to lower it
         - validate: measure the loss on rows the network never trains on
         - log the numbers and save checkpoints / snapshot figures (metrics.py)
"""

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from refltron.common import physics
from refltron.common.loss import LABEL_NAMES, label_range
from refltron.single_channel_mlp.metrics import MetricStore
from refltron.single_channel_mlp.network import OldNet

# =====================================================================================
# DEFAULT SETTINGS: edit these, or override any of them from the command line.
# =====================================================================================

# --- Data ---
INPUT = "s"                 # "s", "p" or "anisotropy" (see the top of this file)
DATA_DIR = Path("data")     # holds reflectivity-1m-fixed-q and reflectivity-1m-fixed-q-isotropic
MAX_SAMPLES = None          # use only the first N rows (None = all 1M); handy for quick tests
VAL_SPLIT = 0.20            # fraction of rows held out for validation (the last 20%)

# --- Training (the original Keras recipe) ---
EPOCHS = 60
BATCH_SIZE = 128            # rows per optimizer step
LEARNING_RATE = 5e-4        # Adam step size
SEED = 42                   # random seed, so runs are repeatable
DEVICE = "auto"             # "auto" picks CUDA, then Apple MPS, then CPU

# --- Logging and checkpoints (see metrics.py) ---
LOCAL = True                # True = keep W&B offline (sync later) and save fewer checkpoints
RUN_NAME = None             # W&B run name (None = "single-<input>")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_EVERY = None     # save a numbered checkpoint every N epochs (None = 20 with LOCAL, else 5)
MAX_CHECKPOINTS = 3         # keep only the newest N numbered checkpoints (best.pt is always kept)

INPUT_CHOICES = ("s", "p", "anisotropy")

# Which dataset and which labels each input uses. The isotropic data stores delta and
# beta in both the xx and zz columns, so the s run only predicts the xx ones.
DATASETS = {
    "s": "reflectivity-1m-fixed-q-isotropic",
    "p": "reflectivity-1m-fixed-q",
    "anisotropy": "reflectivity-1m-fixed-q",
}
ISOTROPIC_LABELS = ("film_thickness", "film_roughness", "oxide_thickness", "delta_xx", "beta_xx")
LABELS = {
    "s": ISOTROPIC_LABELS,
    "p": LABEL_NAMES,
    "anisotropy": LABEL_NAMES,
}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    label_names = LABELS[args.input]
    print(f"Device: {device} | input={args.input} | labels={len(label_names)} | data={DATASETS[args.input]}")

    # 1. Data
    dataset, q_grid = load_dataset(args.input, args.max_samples)
    train_set, val_set = split_dataset(dataset, args.val_split)
    # A DataLoader cuts the rows into mini-batches. Shuffling the training rows each
    # epoch means every step sees a different random mix of samples.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # 2. Network, loss and optimizer
    model = OldNet(n_q=len(q_grid), n_labels=len(label_names)).to(device)
    # Mean squared error between predicted and true labels, both scaled to [0, 1].
    criterion = torch.nn.MSELoss()
    # Adam is a gradient-descent optimizer that adapts the step size for each weight.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # Logging, checkpoints and snapshot figures.
    tracker = MetricStore(model, dataset, train_set, val_set, q_grid, label_names, device, args, optimizer=optimizer)

    # 3. Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, tracker, device)
        tracker.store_metrics(epoch, train_loss, val["loss"], val["mae"], val["profile_l2"], val["dead_outputs"])

    tracker.finish()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training rows. Returns the average training loss."""
    model.train()  # training mode (matters for layers like dropout; harmless here)
    total_loss = 0.0
    total_rows = 0

    for x, targets in loader:
        x = x.to(device)
        targets = targets.to(device)

        # Forward pass: predict the scaled labels and measure how wrong they are.
        predictions = model(x)
        loss = criterion(predictions, targets)

        # Backward pass: compute the gradient of the loss with respect to every weight,
        # then let the optimizer take one step downhill.
        optimizer.zero_grad()  # clear the gradients left over from the previous step
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        total_rows += len(x)

    return total_loss / total_rows


@torch.no_grad()  # no gradients needed: we only measure, we do not train
def evaluate(model, loader, criterion, tracker, device):
    """Measure the model on validation rows.

    Returns a dict with the training loss ("loss"), the depth-profile distance
    ("profile_l2", the same metric the multi-channel MLP reports), the mean absolute
    error of each label in physical units ("mae") and the number of outputs that are 0
    for every row ("dead_outputs": the output ReLU passes no gradient to them).
    """
    model.eval()
    label_scale = torch.tensor(tracker.label_max - tracker.label_min, device=device)
    loss_sum = 0.0
    profile_sum = 0.0
    mae_sum = torch.zeros(len(tracker.label_names), device=device)
    # The largest value each output gives on any row. An output whose largest value is 0
    # is 0 for every row: it is dead.
    largest_output = torch.zeros(len(tracker.label_names), device=device)
    total_rows = 0

    for x, targets in loader:
        x = x.to(device)
        targets = targets.to(device)
        predictions = model(x)
        n = len(x)
        loss_sum += criterion(predictions, targets).item() * n
        profile_sum += tracker.profile_distance(predictions, targets).mean().item() * n
        mae_sum += ((predictions - targets).abs() * label_scale).sum(dim=0)
        largest_output = torch.maximum(largest_output, predictions.max(dim=0).values)
        total_rows += n

    return {
        "loss": loss_sum / total_rows,
        "profile_l2": profile_sum / total_rows,
        "mae": (mae_sum / total_rows).cpu(),
        "dead_outputs": int((largest_output <= 0).sum().item()),
    }


def load_dataset(input_name, max_samples=None):
    """Read the Parquet files into a TensorDataset of (input curve, scaled labels) rows.

    Returns ``(dataset, q_grid)``. The labels are scaled to [0, 1] inside their training
    range: (label - min) / (max - min). q_grid is the fixed q grid shared by every row.
    """
    data_dir = DATA_DIR / DATASETS[input_name] / "data"
    files = sorted(data_dir.glob("train-*.parquet"))
    if len(files) == 0:
        raise FileNotFoundError(f"no Parquet files in {data_dir}")
    print(f"Loading {len(files)} Parquet files from {data_dir} ...")
    label_names = LABELS[input_name]
    df = pl.read_parquet(files, columns=["q", "s", "p"] + list(label_names), n_rows=max_samples)
    q_grid = df["q"].to_numpy()[0].astype(np.float64)  # the same in every row

    if input_name == "s":
        x = np.log(df["s"].to_numpy())
    elif input_name == "p":
        x = np.log(df["p"].to_numpy())
    else:  # "anisotropy"
        s = df["s"].to_numpy().astype(np.float64)
        p = df["p"].to_numpy().astype(np.float64)
        # Grazing angle of each q at the dataset's fixed energy: q = (4 pi / lambda) sin(theta).
        energy_ev = physics.estimate_energy(float(q_grid[0]), float(q_grid[-1]))
        wavelength = physics.HC_EV_ANGSTROM / energy_ev
        theta = np.arcsin(q_grid * wavelength / (4.0 * math.pi))
        c = np.cos(2.0 * theta) ** 2
        x = (p - s * c) / (p + s * c)
        print(f"Anisotropy at {energy_ev:.0f} eV: theta {np.degrees(theta[0]):.2f}-{np.degrees(theta[-1]):.2f} deg, "
              f"A in [{x.min():.3f}, {x.max():.3f}]")

    label_min, label_max = label_range(label_names)
    labels = df.select(label_names).to_numpy().astype(np.float32)
    y = (labels - label_min) / (label_max - label_min)

    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(y))
    return TensorDataset(x, y), q_grid


def split_dataset(dataset, val_split):
    """Use the first rows for training and the last ``val_split`` fraction for validation.

    The rows are already in random order, so the split is random too, but always
    the same, which keeps runs comparable. With val_split = 0 (for example when
    over-fitting a single row on purpose) the training rows are also used to validate.
    """
    n_val = int(len(dataset) * val_split)
    if n_val == 0:
        everything = Subset(dataset, range(len(dataset)))
        return everything, everything
    n_train = len(dataset) - n_val
    return Subset(dataset, range(n_train)), Subset(dataset, range(n_train, len(dataset)))


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
        description="Train OldNet (the single-channel fixed-q MLP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # show defaults in --help
    )
    on_off = argparse.BooleanOptionalAction  # gives both --flag and --no-flag

    data = parser.add_argument_group("data")
    data.add_argument("--input", choices=INPUT_CHOICES, default=INPUT, help="input curve (sets the dataset and labels)")
    data.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="use only the first N rows")
    data.add_argument("--val-split", type=float, default=VAL_SPLIT, help="validation fraction (the last rows)")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=EPOCHS, help="passes over the training rows")
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="rows per optimizer step")
    train.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam learning rate")
    train.add_argument("--seed", type=int, default=SEED, help="random seed")
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=DEVICE, help="where to compute")

    log = parser.add_argument_group("logging and checkpoints")
    log.add_argument("--local", action=on_off, default=LOCAL, help="W&B offline, fewer checkpoints (--no-local: online)")
    log.add_argument("--run-name", default=RUN_NAME, help="W&B run name (default single-<input>)")
    log.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR, help="runs go to <dir>/<W&B run id>/")
    log.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY, help="epochs between numbered checkpoints")
    log.add_argument("--max-checkpoints", type=int, default=MAX_CHECKPOINTS, help="numbered checkpoints to keep")
    return parser.parse_args()


if __name__ == "__main__":
    main()
