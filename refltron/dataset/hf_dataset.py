"""Hugging Face dataset card and upload for generate_reflectivity.py.

    write_dataset_card  write README.md (and copy the preview plots) into the dataset folder
    upload_dataset      create the Hugging Face repository if needed and upload the folder
"""

import shutil
from pathlib import Path

from huggingface_hub import HfApi

# Plots shown in the dataset card. They are made by the preview and --dry-run modes.
README_PLOT_FILES = (
    "single_sample.png",
    "dry_run_parameters.png",
    "dry_run_reflectivity_q4.png",
)


def write_dataset_card(
    dataset_dir: Path,
    plot_dir: Path,
    *,
    n_samples: int,
    n_shards: int,
    n_q: int,
    repo_id: str,
    revision: str,
    isotropic: bool,
    fixed_q: bool,
    q_range: tuple[float, float],
    energy_ev: float,
    theta_range_deg: tuple[float, float],
    energy_range_ev: tuple[float, float],
    film_thickness_range: tuple[float, float],
    film_roughness_max: float,
    oxide_thickness_range: tuple[float, float],
    delta_range: tuple[float, float],
    beta_max: float,
    silicon_density: float,
    silicon_roughness: float,
    oxide_density: float,
) -> Path:
    """Write ``dataset_dir / README.md``, the dataset card shown on Hugging Face.

    The figures section is included only when all three plots exist, either already in
    ``dataset_dir / plots`` or in ``plot_dir`` (they are then copied over).
    """
    has_plots = copy_plots(dataset_dir, plot_dir)
    total_points = n_samples * n_q

    # Text that depends on how the curves are measured.
    if fixed_q:
        measurement_text = (
            f"**Fixed-q version.** Every curve is simulated at the same photon energy "
            f"`E = {energy_ev:.0f} eV` on the same `{n_q}`-point, evenly spaced momentum-transfer grid "
            f"`q ∈ [{q_range[0]}, {q_range[1]}] Å⁻¹`. The `q` column is identical in every row; it is "
            "kept so the schema matches the variable-q branches."
        )
        example_text = f"on the fixed {n_q}-point `q` grid at {energy_ev:.0f} eV"
        q_section = (
            "Every curve shares the same evenly spaced `q` grid, so the `q` distribution is flat "
            f"over `[{q_range[0]}, {q_range[1]}] Å⁻¹`."
        )
        q_column_text = "Fixed momentum transfer grid, identical in every row"
    else:
        measurement_text = (
            "**Variable-q version.** Each curve mimics an experimental angle scan at an arbitrary "
            f"soft X-ray photon energy: `{n_q}` random grazing angles "
            f"`θ ∈ [{theta_range_deg[0]}°, {theta_range_deg[1]}°]` are drawn and sorted ascending, "
            f"evaluated at a randomized photon energy `E ∈ [{energy_range_ev[0]:.0f}, {energy_range_ev[1]:.0f}] eV`, "
            "and cast into momentum transfer `q = (4π / λ) sin θ` (`1/Å`). Photon energy is omitted "
            "from the saved labels so downstream models learn an **energy-invariant** mapping "
            "from `(q, R_s, R_p)` to the film parameters."
        )
        example_text = f"sampled at {n_q} random grazing angles and cast to `q`"
        q_section = (
            "Uniform sampling over `(θ, E)` produces dense coverage at low-to-intermediate `q` "
            "with a high-`q` tail, and every curve has its own `q` grid."
        )
        q_column_text = "Ascending momentum transfer grid cast from `(θ, E)`"

    # Text that depends on the film type.
    if isotropic:
        film_name = "isotropic"
        optics_text = (
            "The film is isotropic: one `δ` drawn uniformly from "
            f"`[{delta_range[0]:.1e}, {delta_range[1]:.1e}]` and one `β` from `[0, {beta_max:.1e}]`, "
            "written to both tensor columns (`delta_xx = delta_zz`, `beta_xx = beta_zz`) so the "
            "schema matches the uniaxial branches. Models only need 5 labels: film thickness, "
            "film roughness, oxide thickness, `delta_xx` and `beta_xx`."
        )
        delta_xx_text = "Film dispersion `δ`"
        delta_zz_text = "Same as `delta_xx` (isotropic film)"
        beta_xx_text = "Film absorption `β`"
        beta_zz_text = "Same as `beta_xx` (isotropic film)"
    else:
        film_name = "uniaxial anisotropic"
        optics_text = (
            "Uniaxial optical constants are drawn in polar form (`component_xx = magnitude · cos(angle)`, "
            "`component_zz = magnitude · sin(angle)`) and stored directly as Cartesian tensor "
            "components (`delta_xx`, `delta_zz`, `beta_xx`, `beta_zz`)."
        )
        delta_xx_text = "In-plane (ordinary) dispersion `δ_xx = δ cos α`"
        delta_zz_text = "Out-of-plane (extraordinary) dispersion `δ_zz = δ sin α`"
        beta_xx_text = "In-plane (ordinary) absorption `β_xx = β cos γ`"
        beta_zz_text = "Out-of-plane (extraordinary) absorption `β_zz = β sin γ`"

    figures_section = ""
    if has_plots:
        figures_section = f"""
---

## Example Sample

![Example s/p reflectivity curves and optical-constant depth profile](plots/single_sample.png)

*Top: simulated `s`- and `p`-polarized power reflectivity `R_s(q)` and `R_p(q)`
{example_text}. Bottom: corresponding
erf-smoothed refractive-index depth profile (`1 - n = δ - iβ`) showing
in-plane (`δ_xx`, `β_xx`) and out-of-plane (`δ_zz`, `β_zz`) components across
the `vacuum | film | oxide | silicon` interfaces.*

---

## Dataset Distributions (`{n_samples:,}` Curves · `{total_points:,}` Points)

### Structural & Optical Parameter Priors

Sampling proceeds in two stages so every interface remains strictly inside the
Nevot–Croce validity regime (`thickness ≥ 2π · roughness`): film and native
oxide thicknesses are drawn first, followed by film roughness capped at
`min({film_roughness_max:.0f} Å, t_film / 2π)` and oxide roughness coupled via `σ_oxide = t_oxide / 2π`.
{optics_text}

![Parameter and optical tensor distributions across 1,000,000 samples](plots/dry_run_parameters.png)

### `q` Grid & Fresnel-Compensated `R · q⁴` Densities

{q_section}

![q distribution and Fresnel-compensated R * q^4 joint densities](plots/dry_run_reflectivity_q4.png)
"""

    card = f"""---
license: gpl-3.0
task_categories:
  - tabular-regression
tags:
  - physics
  - x-ray-reflectivity
  - polarized-reflectivity
  - thin-films
  - materials-science
size_categories:
  - {size_category(n_samples)}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# Polarized Soft X-Ray Reflectivity Dataset (`reflectivity-1m`, branch `{revision}`)

Synthetic polarized soft X-ray reflectivity (`s` and `p` channels) simulated with
`refloxide` for {film_name} thin films on native-oxide silicon:

```text
vacuum | {film_name} film | SiO2 oxide | Si substrate
```

{measurement_text}

{optics_text}
{figures_section}
---

## Schema & Storage Layout

- **Total rows**: `{n_samples:,}` across `{n_shards}` Parquet shard(s) (`data/train-*.parquet`). All rows are stored in a single `train` split so train/validation/test splits can be chosen post-hoc.
- **Substrate constants**: `Si` (`ρ = {silicon_density} g/cm³`, `σ = {silicon_roughness} Å`), `SiO2` (`ρ = {oxide_density} g/cm³`, `σ_oxide = t_oxide / 2π`).
- **Optimized datatypes**: Non-nullable `FixedSizeList(float32, {n_q})` curves + `float32` scalars encoded with `BYTE_STREAM_SPLIT` + `ZSTD`—eliminating per-row list-offset and validity-bitmap overhead while preserving ~38 decades of dynamic range (where `float16` underflows below `~6e-8`).

| Column | Arrow Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `q` | `fixed_size_list<float32>[{n_q}]` | `1/Å` | {q_column_text} |
| `s` | `fixed_size_list<float32>[{n_q}]` | power reflectance | `s`-polarized reflectivity `R_s(q)` |
| `p` | `fixed_size_list<float32>[{n_q}]` | power reflectance | `p`-polarized reflectivity `R_p(q)` |
| `film_thickness` | `float32` | `Å`, `[{film_thickness_range[0]}, {film_thickness_range[1]}]` | Film thickness `t_film` |
| `film_roughness` | `float32` | `Å`, `[0.0, min({film_roughness_max}, t_film / 2π)]` | Vacuum/film RMS roughness `σ_film` (Nevot–Croce capped) |
| `oxide_thickness` | `float32` | `Å`, `[{oxide_thickness_range[0]}, {oxide_thickness_range[1]}]` | Native `SiO2` thickness `t_oxide` (`σ_oxide = t_oxide / 2π`) |
| `delta_xx` | `float32` | `[{delta_range[0]:.1e}, {delta_range[1]:.1e}]` | {delta_xx_text} |
| `delta_zz` | `float32` | `[{delta_range[0]:.1e}, {delta_range[1]:.1e}]` | {delta_zz_text} |
| `beta_xx` | `float32` | `[0.0, {beta_max:.1e}]` | {beta_xx_text} |
| `beta_zz` | `float32` | `[0.0, {beta_max:.1e}]` | {beta_zz_text} |

---

## Loading the Data

### Polars -> Contiguous `float32` NumPy / PyTorch Arrays

Because `q`, `s`, and `p` are stored as fixed-size non-nullable `float32` arrays,
Polars materializes them directly into C-contiguous `(N, {n_q})` tensors:

```python
import numpy as np
import polars as pl

LABEL_COLS = [
    "film_thickness",
    "film_roughness",
    "oxide_thickness",
    "delta_xx",
    "delta_zz",
    "beta_xx",
    "beta_zz",
]

df = pl.read_parquet("hf://datasets/{repo_id}@{revision}/data/train-*.parquet")

q = df["q"].to_numpy()  # ({n_samples:_}, {n_q}) float32, C-contiguous
s = df["s"].to_numpy()  # ({n_samples:_}, {n_q}) float32, C-contiguous
p = df["p"].to_numpy()  # ({n_samples:_}, {n_q}) float32, C-contiguous

inputs = np.stack([s, p, q], axis=1)        # ({n_samples:_}, 3, {n_q}) float32
targets = df.select(LABEL_COLS).to_numpy()  # ({n_samples:_}, 7) float32
```

### Hugging Face `datasets` (with Post-Hoc Train/Validation Split)

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", revision="{revision}", split="train").with_format("torch")
splits = ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = splits["train"], splits["test"]

batch = train_ds[:64]
q, s, p = batch["q"], batch["s"], batch["p"]  # each torch.float32 of shape (64, {n_q})
```

### Streaming Batches with PyArrow (Low Memory)

```python
from huggingface_hub import snapshot_download
import pyarrow.dataset as ds

local_dir = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    revision="{revision}",
    allow_patterns="data/train-*.parquet",
)

dataset = ds.dataset(f"{{local_dir}}/data", format="parquet")
for record_batch in dataset.to_batches(batch_size=4096):
    q = record_batch["q"].values.to_numpy().reshape(-1, {n_q})
    s = record_batch["s"].values.to_numpy().reshape(-1, {n_q})
    p = record_batch["p"].values.to_numpy().reshape(-1, {n_q})
```
"""
    readme_path = dataset_dir / "README.md"
    readme_path.write_text(card, encoding="utf-8")
    return readme_path


def upload_dataset(
    dataset_dir: Path, repo_id: str, revision: str, token: str | None, private: bool
) -> str:
    """Upload README.md, plots and Parquet shards in ``dataset_dir`` to branch ``revision``.

    The branch is created (from main) if it does not exist yet. Old shards on that
    branch that are not in this upload are deleted, so it always matches the local
    folder. Returns the commit URL.
    """
    shards = sorted((dataset_dir / "data").glob("train-*.parquet"))
    if len(shards) == 0:
        raise FileNotFoundError(f"no Parquet shards in {dataset_dir / 'data'}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.create_branch(repo_id=repo_id, repo_type="dataset", branch=revision, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        folder_path=dataset_dir,
        allow_patterns=["README.md", "plots/*.png", "data/train-*.parquet"],
        delete_patterns=["data/*.parquet"],
        commit_message=f"Upload polarized reflectivity dataset ({len(shards)} shard(s))",
    )
    return str(commit.commit_url)


def copy_plots(dataset_dir: Path, plot_dir: Path) -> bool:
    """Make sure all README plots are in ``dataset_dir / plots``. Returns False if any are missing."""
    dataset_plots = dataset_dir / "plots"
    already_there = True
    available = True
    for name in README_PLOT_FILES:
        if not (dataset_plots / name).is_file():
            already_there = False
        if not (plot_dir / name).is_file():
            available = False
    if already_there:
        return True
    if not available:
        return False
    dataset_plots.mkdir(parents=True, exist_ok=True)
    for name in README_PLOT_FILES:
        shutil.copy2(plot_dir / name, dataset_plots / name)
    return True


def size_category(n_samples: int) -> str:
    """Hugging Face's size tag for a dataset with n_samples rows."""
    if n_samples < 1_000:
        return "n<1K"
    if n_samples < 10_000:
        return "1K<n<10K"
    if n_samples < 100_000:
        return "10K<n<100K"
    if n_samples < 1_000_000:
        return "100K<n<1M"
    if n_samples < 10_000_000:
        return "1M<n<10M"
    return "10M<n<100M"
