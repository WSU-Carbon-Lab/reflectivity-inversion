# HOWTO: build the multi-channel MLP yourself

This continues from `../single_channel_mlp/HOWTO.md`. It assumes you have a working
`refltron/my_mlp/` with a network, a training loop, a final report and flags. Here you
turn that into Refltron, one change at a time, in the order the changes were actually
made. The finished files in this folder are the answer key.

Pieces marked **fill in** are left for you.

You need the variable-q dataset in `data/reflectivity-1m/data/` (top-level README,
step 4).

## Step 0: copy your single-channel model

```bash
cp -r refltron/my_mlp refltron/my_refltron
```

Work in `refltron/my_refltron/` from now on. Point `DATA_DIR` at `data/reflectivity-1m/data`
and set `LABELS` to all seven names (import `LABEL_NAMES` from `refltron.common.loss`).

## Step 1: look at the new data

Same drill as before. Read five rows and compare `q` between rows:

```python
import polars as pl
df = pl.read_parquet("data/reflectivity-1m/data/train-00000-of-00010.parquet", n_rows=5)
for i in range(3):
    q = df["q"][i].to_list()
    print(f"row {i}: q from {q[0]:.4f} to {q[-1]:.4f}, {len(q)} points")
```

Every row now has its own grid. Write down why the single-channel network cannot be used
on this data at all, in one sentence. If you cannot, reread "What the network sees" in
`../single_channel_mlp/README.md`.

## Step 2: load three channels

The dataset gives the network `s`, `p` and `q`. Change `load_dataset` to keep all three
as separate tensors, and keep the labels in **physical units** this time (the loss will
do the scaling, step 5):

```python
def load_dataset(max_samples):
    files = sorted(DATA_DIR.glob("train-*.parquet"))
    df = pl.read_parquet(files, n_rows=max_samples)
    s = torch.from_numpy(df["s"].to_numpy().copy())
    p = torch.from_numpy(df["p"].to_numpy().copy())
    q = torch.from_numpy(df["q"].to_numpy().copy())
    labels = torch.from_numpy(df.select(LABEL_NAMES).to_numpy().copy())
    return TensorDataset(s, p, q, labels)
```

Every loop that did `for x, targets in loader` now does `for s, p, q, targets in loader`.
**Fill in:** update `train_one_epoch`, `evaluate` and the final report to unpack four
tensors. Do not change anything else yet. Run it; it will fail at the network, which is
the next step.

## Step 3: input transform and channel stacking

In `network.py`, the network now receives `s`, `p` and `q`. First turn them into one
tensor of shape `(batch, 3, n_q)`:

```python
LOG_FLOOR = 1e-30   # a simulated reflectivity can be exactly 0; log(0) is -inf

class MyRefltron(nn.Module):
    def __init__(self, n_q: int = 256, n_labels: int = 7, regularizer: str = "log") -> None:
        super().__init__()
        self.regularizer = regularizer
        # layers come in step 4

    def regularize(self, s, p, q):
        if self.regularizer == "log":
            S = torch.log10(s.clamp_min(LOG_FLOOR))
            P = torch.log10(p.clamp_min(LOG_FLOOR))
        elif self.regularizer == "rq4":
            # fill in: multiply s and p by q to the fourth power (no log)
            ...
        return torch.stack((S, P, q), dim=1)   # (batch, 3, n_q)
```

Why `q⁴`? Reflectivity falls roughly as `1/q⁴` once past the critical angle (the Fresnel
decay). Multiplying by `q⁴` flattens the curve so the fringes, which carry the thickness
information, are the same size everywhere.

Check the shape before going on:

```python
net = MyRefltron()
s, p, q, y = dataset[0]
print(net.regularize(s[None], p[None], q[None]).shape)   # (1, 3, 256)
```

## Step 4: the channel mixer and the MLP

Still in `__init__`. The mixer is a `Conv1d` with kernel size 1: at every `q` point it
takes the three values `(S, P, q)` and makes eight new ones, with the same weights at
every point.

```python
        mix_channels = 8
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(3, mix_channels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),                     # (batch, 8, n_q) -> (batch, 8 * n_q)
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_q * mix_channels, 400),
            nn.Linear(400, 800),
            nn.ReLU(),
            # fill in: 800 -> 400 -> 300 -> 200 -> 100 with ReLUs, then 100 -> n_labels
            # and NO ReLU on the output this time
        )
        nn.init.constant_(self.mlp[-1].bias, 0.5)   # first guess is the middle of every range

    def forward(self, s, p, q):
        x = self.regularize(s, p, q)
        return self.mlp(self.channel_mixer(x))
```

Two deliberate differences from your `MyNet`: no final ReLU (so nothing can die), and
output biases starting at 0.5 instead of 0. Keep a note of both; you will test whether
they matter at the end.

**Fill in:** count the parameters of the mixer by hand (3 inputs, 8 outputs, kernel 1,
plus biases) and check against `sum(p.numel() for p in net.channel_mixer.parameters())`.

## Step 5: a loss in physical units

Your single-channel loop scaled the labels in the loader and used `MSELoss`. Refltron
keeps the labels physical and gives the loss the job of scaling. That is what lets the
loss also reason about physics. `refltron/common/loss.py` already has it:

```python
from refltron.common.loss import build_loss

criterion = build_loss("mse").to(device)          # start with plain MSE
...
loss = criterion(predictions, targets, q)         # predictions scaled, targets physical
```

Note the third argument. The MSE loss ignores `q`; the profile loss needs it to know the
photon energy and thus the substrate's optical constants. Passing it always keeps the
loop identical across losses.

Get the loop training with `"mse"` first. Then switch:

```python
criterion = build_loss("hybrid").to(device)
```

and watch the loss values change scale. Then add the two reference metrics so you can
compare runs trained with different losses:

```python
from refltron.common.loss import DepthProfileLoss, LabelMSELoss
profile_metric = DepthProfileLoss().to(device)
mse_metric = LabelMSELoss().to(device)
```

**Fill in:** in `evaluate`, accumulate `profile_metric(predictions, targets, q)` and
`mse_metric(predictions, targets, q)` alongside the training loss and return all three.
For the per-label MAE in physical units, `criterion.physical_mae(predictions, targets)`
does the denormalizing for you.

Before moving on, read `DepthProfileLoss` in `loss.py` with a pencil. Draw the staircase.
You do not need to follow the closed-form integral in step 4 of its code, but you should
be able to say what the six step heights and positions are.

## Step 6: standardize the inputs

Print the mean and standard deviation of each channel after `regularize` on a few
thousand rows:

```python
x = net.regularize(s[:5000], p[:5000], q[:5000])
print(x.mean(dim=(0, 2)), x.std(dim=(0, 2)))
```

`S` and `P` sit around -4 with a spread near 1.5; `q` sits around 0.2 with a spread of
0.1. A network learns badly from inputs on such different scales. Fix it with two
buffers, which are tensors saved with the weights but never trained:

```python
        self.register_buffer("input_mean", torch.zeros(3, 1))
        self.register_buffer("input_std", torch.ones(3, 1))

    @torch.no_grad()
    def fit_input_normalization(self, s, p, q):
        x = self.regularize(s, p, q).double()
        # fill in: set self.input_mean and self.input_std from x's mean and std over
        # dimensions (0, 2), reshaped to (3, 1). Clamp std away from zero.
        ...

    def forward(self, s, p, q):
        x = self.regularize(s, p, q)
        x = (x - self.input_mean) / self.input_std
        return self.mlp(self.channel_mixer(x))
```

Call `fit_input_normalization` once in `main`, on training rows only, before the loop.
Why only training rows? Because the validation set is supposed to be data the model has
never touched in any way, and a mean computed over it is a (small) leak.

Run five epochs with and without standardization on the same 50k rows and compare the
validation loss. This is the single biggest easy win in the whole model.

## Step 7: a learning-rate schedule

Add a scheduler that lowers the learning rate along a half cosine from `lr` to 1% of `lr`
over the run:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

for epoch in ...:
    train_loss = train_one_epoch(...)
    val = evaluate(...)
    lr_used = optimizer.param_groups[0]["lr"]
    scheduler.step()
    ...
```

**Fill in:** log `lr_used` to W&B as `train/lr` and add it to the epoch print line. Run
20 epochs and confirm it traces a cosine.

## Step 8: curves the network never saw

A good validation score only proves the network handles films like the training films.
Simulate two cases outside the training range and predict them every epoch. The simulator
calls are already written:

```python
from refltron.multi_channel_mlp.metrics import build_ood_test_cases

ood = build_ood_test_cases()        # a list of two dicts: bare oxide, 2000 Å film
for case in ood:
    print(case["title"], case["target"])
```

Each dict has `s`, `p`, `q` tensors (shape `(256,)`) and a `target`. **Fill in:** after
each epoch, stack the two cases into a batch, predict, denormalize with
`criterion.denormalize`, and print the predicted film thickness of each next to the true
one. What does your network say for the bare oxide after 20 epochs? Does the 2000 Å film
come out above 800 Å, or pinned at the top of the training range?

Neither case goes into the loss. They are a thermometer, not a target.

## Step 9: resume a stopped run

A 600-epoch run takes most of a day. Make it survivable. Your checkpoint should hold
everything needed to carry on exactly:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": vars(args),
}, path)
```

**Fill in:** a `--resume PATH` flag. When given, load the file, restore all three state
dicts, and start the loop at `checkpoint["epoch"] + 1`. Test it by running 4 epochs,
killing the script with Ctrl-C during epoch 3, and resuming: the epoch-4 numbers should
be the same as an uninterrupted run (up to GPU non-determinism). The real
`restore_training_state` in `train.py` also handles old checkpoints without optimizer
state; read how.

## Step 10: the final report, and the four-column snapshot

Your `final_report` from the single-channel HOWTO needs one change: predictions are
denormalized with `criterion.denormalize(...)` instead of your own scaling, and the
targets are already physical. Everything else, including the two figures, is the same.
That is on purpose: a parity plot from this model and one from the single-channel model
should be read the same way.

The snapshot figure drawn at each checkpoint grows from two columns to four (training,
validation, bare oxide, thick film), and its top panel shows the raw `s` and `p` curves on
a log axis instead of one input vector. `save_snapshot_figure` in `metrics.py` draws it
if you hand it the pieces. **Fill in:** call it from your checkpoint-saving code. The
docstring lists every argument; `MetricStore.__init__` in `metrics.py` shows how they are
assembled.

## Step 11: compare with the answer key

```bash
uv run python -m refltron.my_refltron.train --epochs 5 --max-samples 50000
uv run python -m refltron.multi_channel_mlp.train --epochs 5 --max-samples 50000 --local
```

The validation hybrid loss after five epochs should agree to within a factor of two. Then
read the three real files. `network.py` is what you wrote with the `log_rq4` option and
some docstrings. `train.py` is yours plus the resume logic. `metrics.py` is your
`final_report`, your OOD check and your checkpointing folded into one `MetricStore`
class so the training loop stays short.

## Step 12: overfit one row

Same sanity check as the single-channel HOWTO, and more important here because there
are more places for a bug to hide: the standardization, the mixer, the physics in the
loss. Train on one row and make sure the loss can reach zero.

```bash
uv run python -m refltron.my_refltron.train --max-samples 1 --val-split 0 --epochs 300
```

Your `split_dataset` should already handle `val_split = 0` from the single-channel
HOWTO. Two things to watch that did not come up before.

**The loss bounces.** With a constant learning rate of `5e-4`, Adam on a single row
overshoots: the loss drops to `1e-2`, climbs back to `1e-1`, drops again. Run it once
with `--lr-schedule constant` (or your equivalent) to see that, then with the cosine
schedule. As the learning rate falls, the bouncing stops and the loss settles. On the
reference run the final hybrid loss is `1.9e-03`, the thickness error `0.1 Å`, and every
label matched to three decimal places. This is the clearest demonstration you will get
of why the schedule is there.

**The loss does reach near zero.** No dead output this time: there is no ReLU on the
output. If your loss floors at a fixed value like the single-channel one did, check that
you left the final ReLU out in step 4.

Also look at the two OOD columns on the one-row run. The network, having only ever seen
one film, predicts roughly that film for the bare oxide and the 2000 Å case too. That is
the expected behaviour of a network with no data, and the baseline against which the OOD
numbers from a real run should be judged.

**Fill in:** `R²` in the final report prints `nan` for a single row. Why? (Look at the
formula in `label_errors` in `refltron/common/training_plots.py`.)

## Step 13: the long run

Now the real thing. Refltron is trained for 600 epochs on all 1M rows, which takes most
of a day on a laptop GPU, so build up:

```bash
uv run python -m refltron.my_refltron.train --max-samples 100000 --epochs 50     # about half an hour
uv run python -m refltron.my_refltron.train --epochs 600                          # the published recipe
```

After the short run, open `loss_curves.png` and `parity.png` and answer three questions
before starting the long one:

1. Is the validation line still falling at the end, or has it flattened?
2. In the parity plot, is `oxide_thickness` a diagonal or a horizontal band? If a band,
   try `--loss hybrid` if you are not already using it.
3. Did the OOD thick-film prediction climb over the run, and where did it stop?

Then start the long run with `--resume` ready, because something will interrupt it.
The published `v2-relu-log` scores a film thickness MAE of about 28 Å over the whole
validation split (3 Å for films under 100 Å, 40 Å for films over 400 Å) and an R² above
0.97 on every optical constant; the full table is on the model page at
`carbon-lab/refltron`. Compare yours when it finishes.

## Experiments worth running

Each of these changes one line. Run them on the same `--max-samples` and compare
`parity.png`:

1. Put the output ReLU back. Watch `oxide_thickness`.
2. Set the output bias to 0 instead of 0.5. Watch how many epochs the profile loss takes
   to leave its plateau.
3. `--loss mse` against `--loss hybrid`. The oxide thickness panel and the OOD bare oxide
   case are where they differ.
4. `--regularizer rq4` against `log`. These are the published `v1` and `v2`.
5. `mix_channels = 1`. If the loss barely changes, what is the mixer actually doing?

Write down your prediction before each run. Being wrong is the useful outcome.
