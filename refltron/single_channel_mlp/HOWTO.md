# HOWTO: build the single-channel MLP yourself

`README.md` explains what the code in this folder does. This file has you write it. You
will end up with your own working copy of the model, built in the same order someone
would build it for the first time. The finished files in this folder are the answer key.
Check against them after each step, not before.

Pieces marked **fill in** are left for you. Everything else you can type as written.

Setup first (top-level `README.md`): `uv sync`, and the `fixed-q-isotropic` dataset in
`data/reflectivity-1m-fixed-q-isotropic/data/`.

## Step 0: a folder of your own

Make a new package next to this one so nothing you do breaks the real model:

```bash
mkdir refltron/my_mlp
touch refltron/my_mlp/__init__.py
```

You will create three files in it: `network.py`, `train.py`, `metrics.py`. Run them from
the project folder with `uv run python -m refltron.my_mlp.train`.

## Step 1: look at the data before writing any code

Never train on data you have not looked at. In a Python session:

```bash
uv run python
```

```python
import polars as pl

df = pl.read_parquet("data/reflectivity-1m-fixed-q-isotropic/data/train-00000-of-00010.parquet", n_rows=5)
print(df.columns)
print(df.schema)
row = df.row(0, named=True)
print(len(row["q"]), row["q"][:3], row["q"][-3:])
print(row["s"][:3], row["s"][-3:])
print(row["film_thickness"], row["delta_xx"], row["delta_zz"])
```

Things to notice and write down:

- how many `q` points a curve has,
- the first and last `q` values, and whether row 1 has the same ones as row 0,
- how many orders of magnitude `s` spans from first to last point,
- whether `delta_xx` equals `delta_zz` in this dataset.

**Fill in:** plot `s` against `q` for one row with a log y axis. Save it as `plots/my_first_curve.png`.
You will feed the network the log of this curve, and the plot shows you why.

## Step 2: the network

Create `refltron/my_mlp/network.py`. A network in PyTorch is a class that inherits from
`nn.Module`, builds its layers in `__init__`, and says how data flows through them in
`forward`.

```python
import torch
from torch import Tensor, nn


class MyNet(nn.Module):
    """One fixed-q curve in, scaled film parameters out."""

    def __init__(self, n_q: int, n_labels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_q, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            # fill in: 800 -> 400 -> 300 -> 200 -> 100, each followed by a ReLU
            nn.Linear(100, n_labels),
            nn.ReLU(),  # the original keeps a ReLU on the output; we will question this later
        )

    def forward(self, curve: Tensor) -> Tensor:
        return self.layers(curve)


def prepare_input(reflectivity: Tensor) -> Tensor:
    """The only preprocessing: the natural log of the curve."""
    return torch.log(reflectivity)
```

Check it runs before going on:

```python
from refltron.my_mlp.network import MyNet
net = MyNet(n_q=256, n_labels=5)
print(net)
print(sum(p.numel() for p in net.parameters()), "parameters")
out = net(torch.randn(4, 256))
print(out.shape)   # should be (4, 5)
```

**Fill in:** the parameter count by hand for the first layer, `256 -> 400`. Each output
has 256 weights and one bias. Does your number match what `numel` reports for
`net.layers[0]`?

## Step 3: load the data into tensors

Create `refltron/my_mlp/train.py`. Start with the settings block at the top, in capitals.
Every number that you might want to change later goes here, not buried in a function.

```python
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from refltron.common.loss import label_range
from refltron.my_mlp.network import MyNet, prepare_input

# ---- settings ------------------------------------------------------------------
DATA_DIR = Path("data/reflectivity-1m-fixed-q-isotropic/data")
LABELS = ("film_thickness", "film_roughness", "oxide_thickness", "delta_xx", "beta_xx")
MAX_SAMPLES = 50_000      # start small; set to None for all 1M rows
VAL_SPLIT = 0.20
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
SEED = 42
```

Now the loader. The network wants a tensor of inputs and a tensor of targets. The targets
are scaled to 0 to 1 using the label ranges in `refltron/common/loss.py`, so that a
thickness in hundreds of Å and a `delta` of 0.001 count equally in the loss.

```python
def load_dataset(max_samples):
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    df = pl.read_parquet(files, columns=["q", "s"] + list(LABELS), n_rows=max_samples)
    q_grid = df["q"].to_numpy()[0]                      # the same in every row

    x = torch.from_numpy(df["s"].to_numpy().astype(np.float32))
    x = prepare_input(x)

    label_min, label_max = label_range(LABELS)
    labels = df.select(LABELS).to_numpy().astype(np.float32)
    y = # fill in: scale each label to [0, 1] with (labels - min) / (max - min)
    y = torch.from_numpy(y)
    return TensorDataset(x, y), q_grid
```

And the split. The rows are already shuffled on disk, so taking the last 20% is a random
split that stays the same every run.

```python
def split_dataset(dataset, val_split):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    return Subset(dataset, range(n_train)), Subset(dataset, range(n_train, len(dataset)))
```

Test it at the bottom of the file for now:

```python
if __name__ == "__main__":
    dataset, q_grid = load_dataset(MAX_SAMPLES)
    train_set, val_set = split_dataset(dataset, VAL_SPLIT)
    x, y = dataset[0]
    print(len(train_set), len(val_set), x.shape, y)
    print("targets min/max:", dataset.tensors[1].min().item(), dataset.tensors[1].max().item())
```

The targets should lie between 0 and 1. If you see values outside that, your scaling is
upside down.

## Step 4: one training step, then one epoch

This is the heart of it. Add to `train.py`:

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_rows = 0
    for x, targets in loader:
        x = x.to(device)
        targets = targets.to(device)

        predictions = model(x)                    # forward pass
        loss = criterion(predictions, targets)    # how wrong, as one number

        optimizer.zero_grad()                     # forget last step's gradients
        loss.backward()                           # gradient of loss w.r.t. every weight
        optimizer.step()                          # move every weight a little downhill

        total_loss += loss.item() * len(x)
        total_rows += len(x)
    return total_loss / total_rows
```

Read those five lines inside the loop until the order makes sense. Every PyTorch training
loop ever written has them, in that order.

Now evaluation. Same idea without the three optimizer lines, and wrapped in
`torch.no_grad()` so no gradients are stored:

```python
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_rows = 0
    for x, targets in loader:
        # fill in: move to device, predict, accumulate criterion(...) like above
        ...
    return total_loss / total_rows
```

## Step 5: put the loop together

Replace the test block at the bottom with a real `main`:

```python
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset, q_grid = load_dataset(MAX_SAMPLES)
    train_set, val_set = split_dataset(dataset, VAL_SPLIT)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = MyNet(n_q=len(q_grid), n_labels=len(LABELS)).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:3d}/{EPOCHS} | train {train_loss:.2e} | val {val_loss:.2e}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run python -m refltron.my_mlp.train
```

Both numbers should fall every epoch. If the loss sits at exactly the same value from
epoch 2 onwards, every output has died behind the final ReLU. Reduce `LEARNING_RATE` by
ten and try again, then read the note on initialization in the real `network.py`.

## Step 6: make the loss mean something

A validation MSE of `0.008` on scaled labels tells you nothing physical. Add a mean
absolute error per label in real units. Inside `evaluate`, you have `predictions` and
`targets` scaled to 0 to 1; multiplying their difference by `(label_max - label_min)`
gives the error in Å (or in refractive-index units).

```python
    label_min, label_max = label_range(LABELS)
    scale = torch.tensor(label_max - label_min, device=device)
    mae_sum = torch.zeros(len(LABELS), device=device)
    ...
        mae_sum += ((predictions - targets).abs() * scale).sum(dim=0)
    ...
    return total_loss / total_rows, (mae_sum / total_rows).cpu()
```

**Fill in:** change `evaluate` to return both, and print the film thickness MAE in Å on
the epoch line. After five epochs on 50k rows, is it under 100 Å? Under 50?

## Step 7: keep the best weights

The last epoch is not always the best one. Save the weights whenever the validation loss
improves:

```python
    best_val = float("inf")
    for epoch in ...:
        ...
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, "checkpoints/my_mlp_best.pt")
```

(`mkdir -p checkpoints` first.) Later you load it back with
`model.load_state_dict(torch.load(path)["model_state_dict"])`.

## Step 8: the two figures every run should end with

Create `refltron/my_mlp/metrics.py`. The shared module `refltron/common/training_plots.py`
already draws loss curves and parity plots; you only need to collect the inputs.

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from refltron.common.loss import label_range
from refltron.common.training_plots import plot_loss_curves, plot_parity


@torch.no_grad()
def final_report(model, val_set, labels, history, device, out_dir):
    """history: list of {"epoch", "train_loss", "val_loss", "thickness_mae"} per epoch."""
    model.eval()
    label_min, label_max = label_range(labels)
    predicted = []
    true = []
    for x, y in DataLoader(val_set, batch_size=8192):
        predicted.append(model(x.to(device)).cpu().numpy())
        true.append(y.numpy())
    predicted = np.concatenate(predicted) * (label_max - label_min) + label_min
    true = # fill in: same for the targets

    plot_loss_curves(history, out_dir / "loss_curves.png", loss_name="MSE", title="my MLP")
    errors = plot_parity(true, predicted, list(labels), out_dir / "parity.png", title="my MLP · validation rows")
    for name in labels:
        e = errors[name]
        print(f"{name:16s} MAE {e['mae']:.4g}  RMSE {e['rmse']:.4g}  R^2 {e['r2']:+.3f}")
```

**Fill in:** in `train.py`, build the `history` list (append one dict per epoch), reload
the best checkpoint after the loop, and call `final_report`. Open the two PNGs. In the
parity plot, which label has the lowest R²? Why might the s-polarized curve carry little
information about it?

## Step 9: command-line flags

Hard-coded settings are fine until you want to run three variants in a row. Add
`argparse` so every setting at the top of the file can be overridden:

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    # fill in: --batch-size, --val-split, --seed
    return parser.parse_args()
```

Then `main` uses `args.epochs` instead of `EPOCHS` and so on. Now
`uv run python -m refltron.my_mlp.train --help` documents your script for free.

## Step 10: log to Weights & Biases

You do not need a W&B login for this (we train locally, so the run stays offline). Three lines turn the print statements into
a dashboard you can compare across runs:

```python
import wandb

run = wandb.init(project="my-reflectivity", config={"epochs": args.epochs, "lr": args.lr}, mode="offline")
...
run.log({"train/loss": train_loss, "val/loss": val_loss, "val/mae_film_thickness": mae[0].item()}, step=epoch)
...
run.finish()
```

Use `mode="offline"` until the real thing works, then `wandb sync wandb/offline-run-*`.
The real `metrics.py` wraps all of steps 7, 8 and 10 in one `MetricStore` class so that
`train.py` only calls `tracker.store_metrics(...)` once per epoch. Read it now and you
will recognize every line.

## Step 11: compare with the answer key

```bash
diff <(uv run python -m refltron.my_mlp.train --epochs 3 --max-samples 50000 2>&1 | grep Epoch) \
     <(uv run python -m refltron.single_channel_mlp.train --input s --epochs 3 --max-samples 50000 --local 2>&1 | grep Epoch)
```

The numbers will not be identical (the real script uses Keras-style Adam settings and a
different initialization) but they should be the same order of magnitude and fall at the
same rate. Then read the three real files top to bottom. Everything in them is something
you have now written a version of.

## Step 12: overfit one row

Before spending an hour on a real run, prove the whole chain can learn at all. The
cheapest test in machine learning: train on a single row and check the loss goes to
zero. If a network cannot memorize one example, something upstream is broken, and no
amount of data or epochs will fix it.

Your `split_dataset` puts the last 20% of rows in the validation set, which with one row
is an empty set and a division by zero in `evaluate`. Make `val_split = 0` mean "validate
on the training rows too":

```python
def split_dataset(dataset, val_split):
    n_val = int(len(dataset) * val_split)
    if n_val == 0:
        everything = Subset(dataset, range(len(dataset)))
        return everything, everything
    n_train = len(dataset) - n_val
    return Subset(dataset, range(n_train)), Subset(dataset, range(n_train, len(dataset)))
```

Then:

```bash
uv run python -m refltron.my_mlp.train --max-samples 1 --val-split 0 --epochs 300
```

With one row and a batch size of 128 each epoch is one optimizer step, so this takes a
minute. Watch the loss and the thickness MAE. By epoch 100 the thickness error should be
a few Å; by 200 it should be 0.0 Å.

Now look at the loss itself. It will **not** reach zero. On the reference run it flattens
at `1.82e-01` while every label except one is matched to five decimal places. The one
exception is `oxide_thickness`. Print `model(x)` for the row and you will see that
output is exactly `0.0` and never changes. (The real script counts these every epoch and
prints them as `dead outputs`.) That is the output ReLU from step 2 doing exactly what the README warned about: an output
that starts negative gets clipped to 0, the gradient through a ReLU at 0 is 0, and so it
can never move again. Not even one row can teach it. The loss floor is that one label's
squared error divided by the number of labels.

**Fill in:** work out what `(scaled oxide error)² / 5` is from the MAE the final report
prints, and confirm it matches the loss floor. Then remove the output ReLU, rerun, and
check the loss now goes below `1e-5`.

Keep the version without the output ReLU if you prefer. The real `OldNet` keeps it to
match the 2019 original, and avoids the dead-output trap with PyTorch's default
initialization instead. Both are defensible; now you know what the trade is.

## Step 13: the long run

Everything checks out. Put the ReLU question to one side, go back to the full dataset,
and run the real thing:

```bash
uv run python -m refltron.my_mlp.train --max-samples 200000 --epochs 30     # about 20 minutes on a laptop GPU
uv run python -m refltron.my_mlp.train --epochs 60                           # all 1M rows, a few hours
```

Run the short one first and look at `loss_curves.png` before committing to the long one.
If the validation line is still falling at epoch 30, the long run is worth it. If it has
already flattened, more epochs will not help, and the next lever is the input (try `p`
or the anisotropy ratio) rather than the time.

The reference result for 60 epochs on all 1M rows is a film thickness MAE near 13 Å and
an R² of 0.99. Yours should land in the same neighbourhood. If it does, you have trained
a neural network from a blank file.

## Going further

- Switch to the `p` input: change `DATA_DIR` to the `fixed-q` dataset, `"s"` to `"p"`,
  and `LABELS` to all seven. What breaks? (Hint: `label_range` and the parity plot both
  take the label list, so very little should.)
- Build the anisotropy ratio `A` from `s` and `p` as described in the README. You will
  need the grazing angle of each `q`, which needs the photon energy; see
  `refltron.common.physics.estimate_energy`.
- Remove the output ReLU. Does `R²` on `oxide_thickness` improve?

When you are done, delete `refltron/my_mlp/` or keep it as your notebook. It is yours.
