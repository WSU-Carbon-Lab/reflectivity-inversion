# The multi-channel MLP (Refltron)

Read `../single_channel_mlp/README.md` first. This model, published as **Refltron**,
keeps the same training loop and the same three files, and changes four things:

1. It reads both polarizations and the `q` grid, as three input channels.
2. The `q` grid can be different for every curve, so a measurement at any photon energy
   and any set of angles works. That is what "variable-q" means in this repository.
3. The loss compares physical layer stacks, not just the seven numbers.
4. Training is longer and uses a learning-rate schedule, so there are a few extra
   conveniences: resuming, a check on curves the network never saw, and analysis tools.

Each is explained below in the order you will meet it in the code. `HOWTO.md` in this
folder then has you make those changes yourself, starting from your single-channel copy.

## Three channels instead of one

A real measurement is taken at whatever energy and angles the beamline gives you. The
single-channel network cannot handle that: its weights assume the 256 inputs are always
the same 256 `q` values. The fix is to hand the network `q` itself as an input, next to
the reflectivity, so it can work out where each point sits.

`network.py` defines `ReflectivityNet`. The input is three rows of 256 values:

```
S  = log10(R_s)        s-polarized reflectivity after the input transform
P  = log10(R_p)        p-polarized reflectivity after the input transform
q                      where each point was measured
```

Two things happen before the MLP sees them.

**Regularize.** The `--regularizer` flag picks the input transform: `log` takes
`log10`, `rq4` multiplies by `q⁴` to flatten the steep Fresnel decay, `log_rq4` does
both. Reflectivity drops by eight orders of magnitude across a curve, so none of the
published versions use the raw value.

**Standardize.** Each channel is shifted and scaled to mean 0 and standard deviation 1,
using statistics measured on 50,000 training rows before training starts. After the log
transform `S` sits around -4 and `q` around 0.2; inputs on such different scales are
hard to learn from. The mean and standard deviation are saved with the weights, so a
loaded model always applies the same shift.

Then a **channel mixer**: a `Conv1d` with kernel size 1. That is a small fully connected
layer applied at each `q` point separately, turning the three values `(S, P, q)` there
into eight, with the same weights at every point. The eight channels are flattened into
one vector of 8 × 256 = 2048 numbers, and from there the MLP is the same shape as the
single-channel one:

```
(S, P, q)  -> regularize, standardize   (batch, 3, 256)
           -> Conv1d 3 -> 8, ReLU         (batch, 8, 256)
           -> flatten                     (batch, 2048)
           -> Linear 2048 -> 400
           -> Linear 400 -> 800, ReLU
           -> Linear 800 -> 400, ReLU
           -> Linear 400 -> 300, ReLU
           -> Linear 300 -> 200, ReLU
           -> Linear 200 -> 100, ReLU
           -> Linear 100 -> 7
```

Two differences from `OldNet` worth noticing. There is no ReLU on the output, so
predictions are not clipped and nothing can get stuck at 0. And the output biases start
at 0.5, the middle of the scaled range, so the first guess is a 410 Å film rather than
a 20 Å one; the depth-profile loss learns much faster from there. (The missing ReLU
between the first two Linear layers is a quirk of the original design, kept so old
checkpoints still load.)

## A loss that knows about physics

The single-channel model minimizes the mean squared error of the seven scaled labels.
That treats a 10 Å thickness error and a 10 Å roughness error as equally bad, which they
are not. `../common/loss.py` offers three losses, chosen with `--loss`:

| `--loss` | class | what it compares |
|---|---|---|
| `mse` | `LabelMSELoss` | the seven scaled numbers, as before |
| `profile` | `DepthProfileLoss` | the depth profiles the two sets of labels describe |
| `hybrid` | `HybridLoss` | the sum of both (the default) |

A **depth profile** is a plot of an optical constant against height above the silicon.
It is a staircase: vacuum at the top, a step up into the film, another step at the
oxide, a final step into silicon, with each step blurred by that interface's roughness.
The profile loss draws the staircase for the predicted labels and for the true labels and
measures the area between them. Two stacks that look the same get a small loss even if
the seven numbers differ a bit; a film surface in the wrong place gets a big one. The
integral has a closed form, so there is no depth grid and the gradient is exact.

The profile alone has a blind spot: a parameter the profile barely depends on, like the
oxide thickness under a thick film, gets almost no gradient. Adding the plain MSE back in
fixes that. That sum is the hybrid loss.

Whatever loss you train with, `train.py` also reports the other two on the validation
set every epoch (`val/profile_l2`, `val/label_mse` in W&B), so runs trained with
different losses can still be compared.

## Training

```bash
uv run python -m refltron.multi_channel_mlp.train --epochs 5 --max-samples 50000 --local   # smoke test
uv run python -m refltron.multi_channel_mlp.train                                            # the full run, 600 epochs
uv run python -m refltron.multi_channel_mlp.train --resume checkpoints/<run>/best.pt         # continue a stopped run
uv run python -m refltron.multi_channel_mlp.train --help
```

The defaults at the top of `train.py` are the best published configuration, `v2-relu-log`.
Compared to the single-channel recipe:

- **Batch size 1024, 600 epochs.** Bigger batches make better use of a GPU; more epochs
  because the loss keeps improving for a long time. A full run takes most of a day on a
  laptop GPU. Without a GPU, use `--max-samples` and fewer epochs.
- **Cosine learning-rate schedule.** The learning rate starts at `5e-4` and falls along
  half a cosine wave to 1% of that by the last epoch. Big early steps make quick
  progress; small late steps settle into a good minimum instead of bouncing around it.
  `train/lr` in W&B shows the curve. `--lr-schedule constant` turns it off.
- **Validation split 10%** rather than 20%, because a million rows is plenty.

The terminal line looks like this:

```
Epoch  12/600 | train(hybrid) 8.41e-02 | val 8.02e-02 | thickness MAE 31.4 Å | OOD 0Å->14.2 Å | OOD 2000Å->791 Å
```

The last two columns are the out-of-distribution check, described next.

## Checking the network on curves it never saw

Every training and validation row has a film between 20 and 800 Å. What does the
network do with a curve outside that range? `metrics.py` simulates two test cases once
at the start and predicts them every epoch:

- **A bare oxide:** 10 Å of SiO2 on silicon, no film at all. A good network should say
  "film thickness near 0".
- **A 2000 Å film:** two and a half times thicker than anything in training. Its
  Kiessig fringes are much closer together. Nobody expects 2000 Å back, but the answer
  should at least be "very thick".

Neither case is in the loss; they are only watched. They show up as the two right-hand
columns of every snapshot figure and as `ood/*` in W&B. A network that nails the
validation set but predicts 400 Å for both of these has learned the dataset rather than
the physics.

## What you get at the end

The same files as the single-channel model, in `checkpoints/<run id>/`: `best.pt`,
numbered checkpoints, a snapshot `.png` per checkpoint, `loss_curves.png`,
`parity.png` and `metrics.json`. The loss curves and parity plots are drawn by the same
code (`../common/training_plots.py`), so a parity panel here means exactly what it meant
there, and the MAE, RMSE and R² in `metrics.json` are directly comparable.

The snapshot figures have four columns instead of two: training row, validation row,
bare oxide, 2000 Å film. The top panel shows both polarized curves on a log axis instead
of the single input vector, because here the network sees the raw curves and `q`.

Checkpoints also store the optimizer and schedule state, which is what makes `--resume`
pick up exactly where a run stopped.

## After training

Once you have a `best.pt`, the `analysis/` folder has tools for looking inside it:

| script | what it does |
|---|---|
| `render_snapshot.py` | redraw the four-column snapshot for any checkpoint, no training |
| `analysis/diagnostics.py` | numbers on each layer: dead units, effective rank, gradient flow, which `q` points the network reads |
| `analysis/activations.py` | pictures of a few curves flowing through the network, layer by layer |
| `analysis/perceptron_viz.py` | decode each layer's activations back into a curve, to see what the layer still knows |
| `analysis/eval_versions.py` | score every published version on the same 100k validation rows |
| `analysis/compare_runs.py` | compare training logs from a sweep |
| `analysis/hub_figures.py` | the figures on the Hugging Face model page |
| `publish_hf.py` | upload a version to `carbon-lab/refltron` as its own branch |

Each has a usage line at the top of the file.

## Things to try

- Train with `--loss mse` and `--loss hybrid` on the same `--max-samples` and compare
  the two `parity.png` files. The oxide thickness panel is where the difference shows.
- Try `--regularizer rq4` and `--activation gelu`. These are the published `v1` and `v3`
  variants; their scores are on the model page.
- Watch `ood/thick_2000A_pred_thickness` in W&B over a run. Does it climb toward 800 and
  stop, or go past the training range?
- Set `--mix-channels 1` and see how much the channel mixer is actually doing.

## Files

| file | what it holds |
|---|---|
| `network.py` | `ReflectivityNet`: input transforms, standardization, channel mixer, MLP |
| `train.py` | settings, data loading (local Parquet or straight from Hugging Face), the loop, resuming, flags |
| `metrics.py` | `MetricStore`: W&B logging, checkpoints, the two OOD cases, four-column snapshots, final figures and `metrics.json` |
| `render_snapshot.py` | redraw a snapshot for a saved checkpoint |
| `publish_hf.py` | stage and upload versions to the Hub |
| `analysis/` | the tools in the table above |
