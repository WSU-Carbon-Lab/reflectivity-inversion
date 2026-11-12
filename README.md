# ML-reflectivity

Neural networks that turn polarized X-ray reflectivity curves into thin-film parameters.

The main model, **Refltron** (the multi-channel MLP), reads the s- and p-polarized reflectivity of a sample
(vacuum | organic film | SiO2 | Si) together with its q grid, and predicts seven numbers:
the film thickness and roughness, the oxide thickness, and the film's in-plane and
out-of-plane optical constants (delta_xx, delta_zz, beta_xx, beta_zz).

- Trained models: [carbon-lab/refltron](https://huggingface.co/carbon-lab/refltron) (one branch per version)
- Training data: [carbon-lab/reflectivity-1m](https://huggingface.co/datasets/carbon-lab/reflectivity-1m) (1M simulated curves)

## Getting set up

Everything below is done once. If you already have `uv`, a Hugging Face account and a
W&B login, skip to step 4.

### 1. Install uv

`uv` manages Python itself, the virtual environment and the dependencies, all from
`pyproject.toml`. You do not need to install Python separately or activate anything.

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal afterwards so `uv` is on your path.

### 2. Get the code and install the dependencies

```bash
git clone https://github.com/WSU-Carbon-Lab/reflectivity-inversion.git
cd reflectivity-inversion
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, downloads Python 3.13 if you do not
have it, creates `.venv/` and installs the exact package versions the lock file pins
(PyTorch, polars, matplotlib, the `refloxide` simulator, `wandb`, `huggingface-hub`).
Every command in this repository is run as `uv run ...`, which uses that environment
without you having to activate it.

### 3. Accounts and logins

**Weights & Biases** records the loss and the figures of every training run so you can
compare runs in a browser. Our runs go to a hosted W&B instance, not the public one, so
the login needs the host:

```bash
uv run wandb login --host https://api.forge.coreweave.com
```

It will ask for an API key, which you get from that site once you have been added to the
`als-rixs` team. Ask Harlan to add you. If you only want to try things locally, every
training script has a `--local` flag that keeps W&B offline; nothing needs a login then.

**Hugging Face** hosts the training data and the trained models under the `carbon-lab`
organization, which is private. Three steps:

1. Make an account at https://huggingface.co/join if you do not have one.
2. Go to https://huggingface.co/carbon-lab and request to join, or ask Harlan to add you.
3. Create a token at https://huggingface.co/settings/tokens (read access is enough) and
   log in:

```bash
uv run hf auth login
uv run hf auth whoami      # should print your username and carbon-lab under orgs
```

### 4. Download the data

The datasets live in one Hugging Face repository with one branch per variant. Each is a
million simulated curves in ten Parquet files, about 1.5 to 2.2 GB. Download the one you
need into `data/`, where the training scripts look for it:

```bash
# for the single-channel tutorial (--input s)
uv run hf download carbon-lab/reflectivity-1m --repo-type dataset --revision fixed-q-isotropic \
    --include "data/*.parquet" --local-dir data/reflectivity-1m-fixed-q-isotropic

# for --input p and --input anisotropy
uv run hf download carbon-lab/reflectivity-1m --repo-type dataset --revision fixed-q \
    --include "data/*.parquet" --local-dir data/reflectivity-1m-fixed-q

# for the multi-channel model (variable-q)
uv run hf download carbon-lab/reflectivity-1m --repo-type dataset --revision main \
    --include "data/*.parquet" --local-dir data/reflectivity-1m
```

(The multi-channel script can also stream from the Hub directly if the local copy is
missing, but a local copy is much faster.)

### 5. Run something

```bash
uv run python -m refltron.single_channel_mlp.train --input s --epochs 2 --max-samples 50000 --local
```

Two minutes later you should see two epochs of loss values in the terminal and a new
folder under `checkpoints/` with weights, a snapshot figure, `loss_curves.png` and
`parity.png`. If that works, your setup is done.

## Learning path

The code is written to be read as a tutorial, in this order:

1. **`refltron/single_channel_mlp/README.md`**: the smallest model, with the training
   loop, the loss, the optimizer and every figure explained for someone who has never
   trained a network. Then **`HOWTO.md`** next to it: build that model yourself, step
   by step, with gaps left for you to fill in.
2. **`refltron/multi_channel_mlp/README.md`**: the published Refltron model. Same loop,
   plus variable q grids, a physics-based loss, a learning-rate schedule and
   out-of-distribution checks. Its **`HOWTO.md`** turns your single-channel copy into
   Refltron one change at a time.
3. **`refltron/dataset/generate_reflectivity.py`**: how the training data is simulated,
   if you want to change the physics.

Every model folder has the same three files, `network.py`, `train.py` and `metrics.py`,
and every training script has the same main loop. Settings are `CAPITALIZED` defaults at
the top of each script: edit them there, or override any one with a command-line flag
(`--help` lists them). Run everything from the project folder with `uv run python -m`.

## Where things are

All code lives in the `refltron/` package, one folder per piece:

| folder | what it holds |
|---|---|
| `refltron/common/` | shared by every model: `physics.py` (constants, SiO2 / Si substrate, fixed-q energy), `loss.py` (label names and ranges, MSE / depth-profile / hybrid losses), `depth_profile.py` and `snapshot_plots.py` (depth profiles and the panels every snapshot figure shares), `plot_style.py`, `training_plots.py` (the loss curves and parity plots every run ends with), `tracking.py` (W&B settings), `paths.py` |
| `refltron/dataset/` | `generate_reflectivity.py` simulates and uploads the reflectivity-1m datasets (fixed-q / variable-q, uniaxial / isotropic); `hf_dataset.py` writes the dataset card and uploads; `plot_reflectivity.py` draws preview and distribution plots |
| `refltron/multi_channel_mlp/` | Refltron: `README.md` and `HOWTO.md` (tutorial, part 2), `network.py` (ReflectivityNet), `train.py`, `metrics.py` (W&B logging, checkpoints, snapshot figures, final report), `render_snapshot.py`, `publish_hf.py` |
| `refltron/multi_channel_mlp/analysis/` | `diagnostics.py`, `perceptron_viz.py`, `activations.py`, `compare_runs.py`, `eval_versions.py`, `hub_figures.py` |
| `refltron/single_channel_mlp/` | the original 2019 Keras MLP in PyTorch: `README.md` and `HOWTO.md` (tutorial, part 1), `network.py` (OldNet), `train.py` (log S -> 5 labels, log P -> 7, anisotropy ratio -> 7, on the fixed-q datasets), `metrics.py` (W&B logging, checkpoints, snapshot figures, final per-label errors), `publish_hf.py` (card section, figures and one Hub branch per model) |

Generated (not tracked by git): `data/`, `checkpoints/`, `wandb/`, `logs/`, `plots/`, `analysis/runs/`.

## Original 2019 code

The `code-old/` folder keeps the Keras code from the publication below.
[![DOI](https://zenodo.org/badge/212805647.svg)](https://zenodo.org/badge/latestdoi/212805647)

Fast fitting of reflectivity curves using machine learning.

This code was used in the following scientific publication: 

[1] Fast Fitting of Reflectivity Data of Growing Thin Films Using Neural Networks  A. Greco, V. Starostin, C. Karapanagiotis, A. Hinderhofer, A. Gerlach, L. Pithan, S. Liehr, F. Schreiber, & S. Kowarik (2019). _J. Appl. Cryst._, in print.

A co-developed version of this software with a graphical user interface can be found at [kowarik-labs/AI-reflectivity](https://github.com/kowarik-labs/AI-reflectivity).

### Disclaimer
This repository mainly serves as a public archive for the code used in [1] and is not yet optimized for the use by other researchers. Future updates will improve the functionality and usability of the program.

### Main dependencies
To be able to run the code, a python 3.7 installation with the following dependencies is required:
- keras
- tensorflow
- numpy
- tqdm
- csv
- datetime
- configobj
- matplotlib

### Usage
Provided a correct python installation is available, the code can be run from a suitable shell as-is using the configuration and neural network archtitecture used in [1]. The trained network can be tested on the provided real-time X-ray reflectivity dataset "DIP_330K.txt" of a growing diindenoperylene film on a silicon wafer with a native oxide layer (published in [1]).

##### Generate the reflectivity curves for training
`$ python generate_training_data.py`

##### Define neural network and execute training
`$ python training.py`

##### Predict parameters of the test file "DIP_330K.txt" and plot fitted curves
`$ python prediction.py`

### Authors
- Alessandro Greco (Institut für Angewandte Physik, University of Tübingen)
- Vladimir Starostin (Institut für Angewandte Physik, University of Tübingen)
- Christos Karapanagiotis (Institut für Physik, Humboldt Universität zu Berlin)
- Alexander Hinderhofer (Institut für Angewandte Physik, University of Tübingen)
- Alexander Gerlach (Institut für Angewandte Physik, University of Tübingen)
- Linus Pithan (ESRF The European Synchrotron)
- Sascha Liehr (Bundesanstalt für Materialforschung und -prüfung (BAM))
- Frank Schreiber (Institut für Angewandte Physik, University of Tübingen)
- Stefan Kowarik (Bundesanstalt für Materialforschung und -prüfung (BAM) and Institut für Physik, Humboldt Universität zu Berlin)
