# ML-reflectivity

Neural networks that turn polarized X-ray reflectivity curves into thin-film parameters.

The current model, **Refltron**, reads the s- and p-polarized reflectivity of a sample
(vacuum | organic film | SiO2 | Si) together with its q grid, and predicts seven numbers:
the film thickness and roughness, the oxide thickness, and the film's in-plane and
out-of-plane optical constants (delta_xx, delta_zz, beta_xx, beta_zz).

- Trained models: [carbon-lab/refltron](https://huggingface.co/carbon-lab/refltron) (one branch per version)
- Training data: [carbon-lab/reflectivity-1m](https://huggingface.co/datasets/carbon-lab/reflectivity-1m) (1M simulated curves)

## Quick start

```bash
uv sync                                                        # install dependencies
uv run python scripts/train.py --epochs 5 --max-samples 50000 --local   # short test run
uv run python scripts/train.py                                 # full run with the defaults
```

All training settings are listed as `CAPITALIZED` defaults at the top of
`scripts/train.py`: edit them there, or override any one with a command-line flag
(`uv run python scripts/train.py --help` lists them).

## Where things are

The core, in reading order:

| file | what it does |
|---|---|
| `scripts/network.py` | the network: input transform, channel mixer, dense layers |
| `scripts/loss.py` | the losses: label MSE, depth-profile distance, and the hybrid of both |
| `scripts/train.py` | loads the data, builds everything, runs the training loop |
| `scripts/metrics.py` | logging to W&B, checkpoints, and per-checkpoint snapshot figures |
| `scripts/physics.py` | shared physical constants and the SiO2 / Si substrate model |

Tools:

| file | what it does |
|---|---|
| `scripts/generate_reflectivity.py` | simulates the training dataset and uploads it |
| `scripts/render_snapshot.py` | redraws a checkpoint's snapshot figure |
| `scripts/publish_hf.py` | publishes trained runs to `carbon-lab/refltron` |
| `analysis/diagnostics.py` | layer-by-layer health checks for every checkpoint |
| `analysis/perceptron_viz.py` | how a curve morphs through the layers, plus one card per layer |
| `analysis/compare_runs.py` | side-by-side comparison of training runs |

Generated (not tracked by git): `data/`, `checkpoints/`, `wandb/`, `logs/`, `analysis/runs/`.

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
