# The single-channel MLP: start here

This folder is the place to begin if you have never trained a neural network. It holds
the smallest model in the repository, a port of the 2019 Keras network from `code-old/`,
and three short files that show the whole training loop with nothing hidden. Once this
folder makes sense, the multi-channel model in `../multi_channel_mlp/` is the same loop
with a few more moving parts.

When you have read this, `HOWTO.md` in the same folder walks you through writing the
model yourself from a blank file.

Before you start, work through the setup in the top-level `README.md` (installing `uv`,
logging in to Hugging Face, downloading the data; a W&B login is optional because we train locally).

## The problem

An X-ray reflectivity measurement shines X-rays onto a thin film at a grazing angle and
records how much is reflected as the angle increases. The result is a curve: reflectivity
against `q`, the momentum transfer, which is just a rescaled angle. The shape of that
curve depends on the film: how thick it is, how rough its surface is, how much the X-rays
slow down and get absorbed inside it. Fitting those parameters by hand is slow and needs
a good starting guess.

The idea here is to turn the fitting around. We simulate a million films whose
parameters we know, compute their reflectivity curves, and train a network to go from
curve to parameters. After training, a new curve goes in and the parameters come out in
a millisecond.

Every simulated sample is the same four-layer stack:

```
vacuum | organic film | native SiO2 oxide | Si substrate
```

The labels the network has to predict are the numbers that describe the film and the
oxide:

| label | meaning | range in the data |
|---|---|---|
| `film_thickness` | how thick the film is | 20 to 800 Å |
| `film_roughness` | how blurred the film's top surface is | 0 to 40 Å |
| `oxide_thickness` | the native oxide under the film | 0 to 15 Å |
| `delta_xx`, `delta_zz` | how much the film slows the X-rays, in-plane and out-of-plane | -0.005 to 0.005 |
| `beta_xx`, `beta_zz` | how much the film absorbs, in-plane and out-of-plane | 0 to 0.005 |

The `xx` / `zz` pairs exist because an organic film can be anisotropic: molecules that
stand up behave differently to light polarized along them than across them. Some of the
datasets are isotropic (the two directions are the same), and then there are only five
distinct labels.

## What the network sees

This model is called *single-channel* and *fixed-q* for two reasons.

**Fixed q.** Every curve in the dataset is measured at the same photon energy, 250 eV,
on the same 256 `q` values. Because the grid never changes, the network does not need to
be told where each point sits in `q`: the first input neuron always gets the reflectivity
at the smallest `q`, the last one always gets the largest. The grid is baked into the
weights.

**Single channel.** The network reads one curve of 256 numbers. Which curve is the
`--input` flag:

| `--input` | curve fed in | labels | dataset |
|---|---|---|---|
| `s` | `ln R_s`, the s-polarized reflectivity | 5 | `reflectivity-1m-fixed-q-isotropic` |
| `p` | `ln R_p`, the p-polarized reflectivity | 7 | `reflectivity-1m-fixed-q` |
| `anisotropy` | `A = (P - S c) / (P + S c)`, with `c = cos² 2θ` | 7 | `reflectivity-1m-fixed-q` |

Reflectivity drops by many orders of magnitude across a curve, so we feed the network the
natural log rather than the raw value. That is the only preprocessing for `s` and `p`.
The anisotropy ratio `A` is a hand-made feature: for a film with refractive index close
to 1, `R_p ≈ R_s cos² 2θ`, so `A` removes that purely geometric difference and what is
left is a direct signature of the anisotropy. It is bounded between -1 and 1 and is close
to 0 for an isotropic film. The point of trying all three is to see how much the choice
of input matters even when the network is identical.

## The network

`network.py` defines `OldNet`. It is a multi-layer perceptron, or MLP: a stack of fully
connected layers, each followed by a ReLU.

```
curve (256 values)
  -> Linear 256 -> 400,  ReLU
  -> Linear 400 -> 800,  ReLU
  -> Linear 800 -> 400,  ReLU
  -> Linear 400 -> 300,  ReLU
  -> Linear 300 -> 200,  ReLU
  -> Linear 200 -> 100,  ReLU
  -> Linear 100 -> n_labels,  ReLU
```

A `Linear(a, b)` layer computes `b` outputs, each a weighted sum of all `a` inputs plus a
bias. Those weights and biases are what training adjusts. ReLU is `max(0, x)`: it
throws away negative values. Without a non-linearity like this between layers, the whole
stack would collapse into one big linear map and could only ever fit straight lines.

The outputs are the labels scaled to the range 0 to 1, where 0 is the bottom of a label's
training range and 1 is the top. Scaling matters because a thickness is hundreds of Å and
a `delta` is about 0.001. Without scaling, the loss would be dominated by thickness errors
and the network would ignore the optical constants.

The final ReLU is a quirk inherited from the original Keras model. It stops predictions
going below 0, which is fine for labels scaled to 0 to 1, but it also means an output that
drifts negative early in training gets stuck at exactly 0 with no gradient to pull it back.
The training log reports how many outputs are stuck as `dead outputs`. If you see a number
other than 0 after the first couple of epochs, the run is in trouble.

## Training, step by step

`train.py` has everything in one place. The settings sit at the top of the file in
capitals. Edit them there if you want a permanent change, or pass the matching
command-line flag for a one-off.

```bash
uv run python -m refltron.single_channel_mlp.train --input s --epochs 2 --max-samples 50000 --local   # 2-minute smoke test
uv run python -m refltron.single_channel_mlp.train --input s                                            # the real thing, 60 epochs
uv run python -m refltron.single_channel_mlp.train --help                                               # every flag
```

What the script does, in the order it does it:

1. **Load the data.** The Parquet files in `data/<dataset>/data/` are read into memory,
   the chosen input curve is computed, and each label is scaled to 0 to 1. The last 20%
   of rows are held out as the *validation set*. The network never trains on them, so its
   error on them is an honest estimate of how it will do on a curve it has never seen.

2. **Build the pieces.** The network, the loss and the optimizer.
   - The *loss* is a single number saying how wrong a batch of predictions is. Here it is
     the mean squared error between predicted and true scaled labels. Lower is better,
     and a perfect model scores 0.
   - The *optimizer* is Adam, which is gradient descent with a per-weight adaptive step
     size. The learning rate, `5e-4`, sets how big those steps are. Too big and training
     bounces around; too small and it crawls.

3. **Loop over epochs.** One epoch is one pass over every training row. Within an epoch
   the rows are shuffled and cut into batches of 128. For each batch:
   - the network predicts the labels (the *forward pass*),
   - the loss is computed,
   - the gradient of the loss with respect to every weight is computed (the *backward
     pass*, `loss.backward()`),
   - the optimizer nudges every weight a little in the direction that lowers the loss.

   After the pass, the network is run on the validation rows without training, and the
   epoch's numbers are logged.

Here is what a healthy run prints:

```
Epoch   1/60 | train(mse) 2.01e-02 | val 1.12e-02 | thickness MAE 48.3 Å | profile L2 1.21e-01 | dead outputs 0
Epoch   2/60 | train(mse) 9.84e-03 | val 8.07e-03 | thickness MAE 36.9 Å | profile L2 1.01e-01 | dead outputs 0
...
Epoch  60/60 | train(mse) 2.52e-03 | val 2.71e-03 | thickness MAE 12.6 Å | profile L2 5.45e-02 | dead outputs 0
```

Both losses should fall together. If the training loss keeps falling while the validation
loss flattens or rises, the network is memorizing the training rows instead of learning
the physics. That is *over-fitting*, and the fix is more data, fewer epochs, or a smaller
network.

The other columns are there to make the loss tangible. `thickness MAE` is the mean
absolute error of the film thickness on validation rows, in Å. A loss of `2.7e-03` is
hard to picture; "off by 13 Å on average" is not. `profile L2` is the depth-profile
distance explained in the multi-channel README; it is reported here so the two models
can be compared on the same scale.

## What you get at the end

Every run writes to `checkpoints/<run id>/`:

| file | what it is |
|---|---|
| `best.pt` | the weights from the epoch with the lowest validation loss |
| `epoch_020.pt`, ... | periodic checkpoints, the newest three are kept |
| `best.png`, `epoch_020.png`, ... | a snapshot figure for each checkpoint, see below |
| `loss_curves.png` | training and validation loss against epoch |
| `parity.png` | predicted against true for every label |
| `metrics.json` | the final MAE, RMSE and R² of every label |

The two figures at the end are the standard way to judge any regression model, and the
multi-channel model produces the same two.

**Loss curves.** Training and validation loss against epoch, on a log axis, with the
best epoch marked. This is the picture to look at first. You want two lines that fall
together and flatten out, not a validation line that bends upwards.

**Parity plots.** One panel per label with the true value along the bottom and the
predicted value up the side. A perfect model puts every point on the dashed diagonal.
A horizontal band means the network is ignoring that label and guessing the average.
The numbers in each panel are:

- **MAE**, mean absolute error, in the label's own units: on average, how far off.
- **RMSE**, root mean squared error: like MAE but big misses count more, so RMSE much
  larger than MAE means a few wild predictions.
- **R²**: 1 is perfect, 0 is no better than always predicting the mean, negative is worse
  than that.

**Snapshot figures.** These are drawn during training, one per checkpoint, so you can
watch the network learn. Two columns, one training row and one validation row, each with
three panels: the input curve exactly as the network sees it, the depth profile the
predicted labels describe against the true one, and a bar chart of true against predicted
labels. Early snapshots show predictions sitting at the middle of every range; by the end
the bars should match.

The same numbers go to Weights & Biases under the keys `train/loss`, `val/loss`,
`val/mae_<label>` and so on, so you can compare runs in the browser. `--local` is the
default here: the run is recorded offline under `./wandb/`, needs no login, and you can
sync it later with `wandb sync`. `--no-local` sends it to the hosted server (login needed).

## Things to try

- Train all three inputs and compare `metrics.json`. Which input recovers `delta_zz`
  best, and why would p-polarized light know more about the out-of-plane direction than
  s-polarized light?
- Change `LEARNING_RATE` to `5e-3` and watch the loss curve. Then try `5e-5`.
- Set `--max-samples 10000` and run 60 epochs. The loss curves should show over-fitting
  clearly.
- Replace the final `nn.ReLU()` in `network.py` with nothing and see whether `dead
  outputs` and the final errors change.

## Files

| file | what it holds |
|---|---|
| `network.py` | `OldNet`, the seven-layer MLP, and `prepare_old_input` (the log transform) |
| `train.py` | settings, data loading, the training loop, command-line flags |
| `metrics.py` | `MetricStore`: W&B logging, checkpoints, snapshot figures, and the final loss curves, parity plots and `metrics.json` |
| `publish_hf.py` | uploads a finished run to the `carbon-lab/refltron` model repository on Hugging Face |

Shared code lives in `../common/`: label names and ranges in `loss.py`, the figure style
in `plot_style.py`, the end-of-run figures in `training_plots.py`.
