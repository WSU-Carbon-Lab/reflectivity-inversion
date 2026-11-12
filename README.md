# ML-reflectivity

Neural networks that turn polarized X-ray reflectivity curves into thin-film parameters.

The main model, **Refltron** (the multi-channel MLP), reads the s- and p-polarized reflectivity of a sample
(vacuum | organic film | SiO2 | Si) together with its q grid, and predicts seven numbers:
the film thickness and roughness, the oxide thickness, and the film's in-plane and
out-of-plane optical constants (delta_xx, delta_zz, beta_xx, beta_zz).

- Trained models: [carbon-lab/refltron](https://huggingface.co/carbon-lab/refltron) (one branch per version)
- Training data: [carbon-lab/reflectivity-1m](https://huggingface.co/datasets/carbon-lab/reflectivity-1m) (1M simulated curves)

## Quick start

```bash
uv sync                                                              # install dependencies
uv run python -m refltron.multi_channel_mlp.train --epochs 5 --max-samples 50000 --local   # short test run
uv run python -m refltron.multi_channel_mlp.train                    # full Refltron run
uv run python -m refltron.single_channel_mlp.train --input s         # original single-channel MLP
```

Run everything from the project folder with `python -m`. Every model folder has the same three files, `network.py`, `train.py` and `metrics.py`, and every
training script has the same main loop. Each script lists its settings
as `CAPITALIZED` defaults at the top: edit them there, or override any one with a
command-line flag (`--help` lists them).

## Where things are

All code lives in the `refltron/` package, one folder per piece:

| folder | what it holds |
|---|---|
| `refltron/common/` | shared by every model: `physics.py` (constants, SiO2 / Si substrate, fixed-q energy), `loss.py` (label names and ranges, MSE / depth-profile / hybrid losses), `depth_profile.py` and `snapshot_plots.py` (depth profiles and the panels every snapshot figure shares), `plot_style.py`, `tracking.py` (W&B settings), `paths.py` |
| `refltron/dataset/` | `generate_reflectivity.py` simulates and uploads the reflectivity-1m datasets (fixed-q / variable-q, uniaxial / isotropic); `hf_dataset.py` writes the dataset card and uploads; `plot_reflectivity.py` draws preview and distribution plots |
| `refltron/multi_channel_mlp/` | Refltron: `network.py` (ReflectivityNet), `train.py`, `metrics.py` (W&B logging, checkpoints, snapshot figures), `render_snapshot.py`, `publish_hf.py` |
| `refltron/multi_channel_mlp/analysis/` | `diagnostics.py`, `perceptron_viz.py`, `activations.py`, `compare_runs.py`, `eval_versions.py`, `hub_figures.py` |
| `refltron/single_channel_mlp/` | the original 2019 Keras MLP in PyTorch: `network.py` (OldNet), `train.py` (log S -> 5 labels, log P -> 7, anisotropy ratio -> 7, on the fixed-q datasets), `metrics.py` (W&B logging, checkpoints, snapshot figures, final per-label errors), `publish_hf.py` (card section, figures and one Hub branch per model) |

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
