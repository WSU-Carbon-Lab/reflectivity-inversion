"""Script to generate polarized reflectivity data using refloxide.

Data stored here - https://huggingface.co/datasets/carbon-lab/reflectivity-1m

Each sample draws a random photon energy in [200, 500] eV and 256 random
grazing incidence angles theta in [0.5, 65] degrees to mimic an experimental
measurement, then casts (theta, energy) into wavevector transfer q [1/Angstrom]
for the saved dataset. Photon energy is not included in the training labels so
the downstream model learns an energy-invariant representation.

Uniaxial optical constants are sampled in polar form (magnitude + orientation
angle) and stored directly as Cartesian tensor components
(``delta_xx``, ``delta_zz``, ``beta_xx``, ``beta_zz``) rather than angles.

Stack, top to bottom:

    vacuum | anisotropic film | oxide | silicon substrate

Modes:
1. Single-sample preview (default, when ``-n`` and ``--dry-run`` are omitted):
   draws one random sample and plots the s/p reflectivity pair + depth profile.
2. Dry run (``--dry-run [-n 1000000]``):
   streams curves without saving data and writes parameter + ``R*q^4`` / ``q``
   distribution plots to ``plots/``.
3. Full dataset generation + Hugging Face upload (``-n 1000000``):
   bypasses all plotting, streams chunks into Zstd + ``BYTE_STREAM_SPLIT``
   Parquet shards using non-nullable ``FixedSizeList(Float32, 256)`` columns
   and ``Float32`` scalar labels (single ``train`` split for post-hoc splitting),
   and uploads to ``carbon-lab/reflectivity-1m`` (pass ``--no-upload`` to keep
   local only).

Usage:
    uv run python scripts/generate_reflectivity.py [--seed N] [--output PATH]
    uv run python scripts/generate_reflectivity.py --dry-run [-n 1000000]
    uv run python scripts/generate_reflectivity.py -n 1000000
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.parquet as pq
from refloxide import tmm
from refloxide.model import FreeUniTensor, MaterialSLD, ReflectModel

TWO_PI = 2.0 * math.pi
HC_EV_ANGSTROM = 12398.41984  # eV * Angstrom
DEFAULT_HF_REPO_ID = "carbon-lab/reflectivity-1m"

LABEL_NAMES: tuple[str, ...] = (
    "film_thickness",
    "film_roughness",
    "oxide_thickness",
    "delta_xx",
    "delta_zz",
    "beta_xx",
    "beta_zz",
)

README_PLOT_FILES: tuple[str, ...] = (
    "single_sample.png",
    "dry_run_parameters.png",
    "dry_run_reflectivity_q4.png",
)

DeviceChoice = Literal["auto", "cpu", "gpu"]
ResolvedDevice = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class FixedStack:
    """Dataset sampling ranges and fixed substrate/oxide constants."""

    energy_ev_range: tuple[float, float] = (200.0, 500.0)  # eV (not trained)
    theta_range_deg: tuple[float, float] = (0.5, 65.0)  # grazing degrees
    n_q: int = 256

    si_formula: str = "Si"
    si_density: float = 2.33  # g/cm^3
    si_roughness: float = 1.2  # Angstrom, oxide/silicon interface
    oxide_formula: str = "SiO2"
    oxide_density: float = 2.2  # g/cm^3
    # Keeps the oxide at the Nevot-Croce limit boundary; raise it to sit
    # strictly inside t_oxide > 2*pi*sigma_oxide.
    oxide_thickness_to_roughness: float = TWO_PI

    film_thickness_range: tuple[float, float] = (20.0, 800.0)  # Angstrom
    oxide_thickness_range: tuple[float, float] = (0.0, 15.0)  # Angstrom
    film_roughness_range: tuple[float, float] = (0.0, 40.0)  # Angstrom
    delta_magnitude_range: tuple[float, float] = (-5.0e-3, 5.0e-3)
    beta_magnitude_range: tuple[float, float] = (1.0e-6, 5.0e-3)
    angle_range_deg: tuple[float, float] = (0.0, 90.0)


def theta_to_q(
    theta_deg: NDArray[np.float64], energy_ev: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Cast grazing angles ``theta_deg`` (degrees) at ``energy_ev`` to ``q`` (1/Angstrom)."""
    wavelength = HC_EV_ANGSTROM / np.asarray(energy_ev, dtype=np.float64)
    return (4.0 * math.pi / wavelength) * np.sin(np.deg2rad(theta_deg))


def resolve_device(device: DeviceChoice) -> ResolvedDevice:
    """Resolve ``'auto'`` to ``'gpu'`` when wgpu/Metal is available, else ``'cpu'``."""
    if device in ("cpu", "gpu"):
        return device
    try:
        q_probe = np.array([0.05], dtype=np.float64)
        idx_probe = np.array([0], dtype=np.int64)
        layers_probe = np.zeros((1, 2, 4), dtype=np.float64)
        tensors_probe = np.zeros((1, 2, 3, 3), dtype=np.complex128)
        energy_probe = np.array([284.4], dtype=np.float64)
        tmm.uniaxial_reflectivity_points(
            q_probe,
            idx_probe,
            layers_probe,
            tensors_probe,
            energy_probe,
            parallel=False,
            device="gpu",
        )
        return "gpu"
    except RuntimeError:
        return "cpu"


@dataclass(frozen=True)
class Sample:
    """One random draw of experimental conditions and film parameters."""

    energy_ev: float  # eV (randomized, not trained)
    theta_deg: NDArray[np.float64]  # degrees, shape (n_q,), sorted ascending
    film_thickness: float  # Angstrom
    film_roughness: float  # Angstrom, vacuum/film interface
    oxide_thickness: float  # Angstrom
    oxide_roughness: float  # Angstrom, film/oxide interface
    delta_magnitude: float
    delta_angle: float  # radians, 0 = in-plane, pi/2 = out-of-plane
    beta_magnitude: float
    beta_angle: float  # radians

    @property
    def q_values(self) -> NDArray[np.float64]:
        """Wavevector transfer q [1/Angstrom] cast from (theta_deg, energy_ev)."""
        return theta_to_q(self.theta_deg, self.energy_ev)

    @property
    def delta_xx(self) -> float:
        return self.delta_magnitude * math.cos(self.delta_angle)

    @property
    def delta_zz(self) -> float:
        return self.delta_magnitude * math.sin(self.delta_angle)

    @property
    def beta_xx(self) -> float:
        return self.beta_magnitude * math.cos(self.beta_angle)

    @property
    def beta_zz(self) -> float:
        return self.beta_magnitude * math.sin(self.beta_angle)

    @property
    def labels(self) -> NDArray[np.float64]:
        """Target parameter vector for ML training (energy-invariant, Cartesian tensor)."""
        return np.array(
            [
                self.film_thickness,
                self.film_roughness,
                self.oxide_thickness,
                self.delta_xx,
                self.delta_zz,
                self.beta_xx,
                self.beta_zz,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SampleBatch:
    """Vectorized draw of ``n`` samples for fast batched simulation."""

    energy_ev: NDArray[np.float64]  # (n,)
    theta_deg: NDArray[np.float64]  # (n, n_q), sorted ascending along axis 1
    film_thickness: NDArray[np.float64]  # (n,)
    film_roughness: NDArray[np.float64]  # (n,)
    oxide_thickness: NDArray[np.float64]  # (n,)
    oxide_roughness: NDArray[np.float64]  # (n,)
    delta_magnitude: NDArray[np.float64]  # (n,)
    delta_angle: NDArray[np.float64]  # (n,) radians
    beta_magnitude: NDArray[np.float64]  # (n,)
    beta_angle: NDArray[np.float64]  # (n,) radians

    @property
    def size(self) -> int:
        return int(self.energy_ev.shape[0])

    @property
    def q_values(self) -> NDArray[np.float64]:
        return theta_to_q(self.theta_deg, self.energy_ev[:, np.newaxis])

    @property
    def delta_xx(self) -> NDArray[np.float64]:
        return self.delta_magnitude * np.cos(self.delta_angle)

    @property
    def delta_zz(self) -> NDArray[np.float64]:
        return self.delta_magnitude * np.sin(self.delta_angle)

    @property
    def beta_xx(self) -> NDArray[np.float64]:
        return self.beta_magnitude * np.cos(self.beta_angle)

    @property
    def beta_zz(self) -> NDArray[np.float64]:
        return self.beta_magnitude * np.sin(self.beta_angle)

    @property
    def labels(self) -> NDArray[np.float64]:
        return np.column_stack(
            [
                self.film_thickness,
                self.film_roughness,
                self.oxide_thickness,
                self.delta_xx,
                self.delta_zz,
                self.beta_xx,
                self.beta_zz,
            ]
        )


def sample_theta_deg(
    config: FixedStack, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Draw ``n_q`` random grazing angles in degrees, sorted ascending."""
    theta_deg = rng.uniform(*config.theta_range_deg, size=config.n_q)
    theta_deg.sort()
    return theta_deg


def sample_thicknesses(
    config: FixedStack, rng: np.random.Generator
) -> tuple[float, float]:
    """Stage 1: draw the film and oxide thicknesses."""
    film_thickness = rng.uniform(*config.film_thickness_range)
    oxide_thickness = rng.uniform(*config.oxide_thickness_range)
    return film_thickness, oxide_thickness


def sample_roughnesses(
    config: FixedStack,
    rng: np.random.Generator,
    film_thickness: float,
    oxide_thickness: float,
) -> tuple[float, float]:
    """Stage 2: draw the roughness pairings once the thicknesses are known.

    The film roughness is capped at ``t_film / (2*pi)`` so the slab stays
    inside the Nevot-Croce limit ``thickness > 2*pi*roughness``; the oxide
    roughness follows the fixed thickness-to-roughness ratio.
    """
    nevot_croce_cap = film_thickness / TWO_PI
    roughness_hi = min(config.film_roughness_range[1], nevot_croce_cap)
    roughness_hi = max(roughness_hi, config.film_roughness_range[0])
    film_roughness = rng.uniform(config.film_roughness_range[0], roughness_hi)
    oxide_roughness = oxide_thickness / config.oxide_thickness_to_roughness
    return film_roughness, oxide_roughness


def random_sample(config: FixedStack, rng: np.random.Generator) -> Sample:
    """Draw one random sample: energy, theta grid, thicknesses, roughnesses, and optical tensor."""
    energy_ev = rng.uniform(*config.energy_ev_range)
    theta_deg = sample_theta_deg(config, rng)
    film_thickness, oxide_thickness = sample_thicknesses(config, rng)
    film_roughness, oxide_roughness = sample_roughnesses(
        config, rng, film_thickness, oxide_thickness
    )

    angle_lo, angle_hi = np.deg2rad(config.angle_range_deg)
    return Sample(
        energy_ev=energy_ev,
        theta_deg=theta_deg,
        film_thickness=film_thickness,
        film_roughness=film_roughness,
        oxide_thickness=oxide_thickness,
        oxide_roughness=oxide_roughness,
        delta_magnitude=rng.uniform(*config.delta_magnitude_range),
        delta_angle=rng.uniform(angle_lo, angle_hi),
        beta_magnitude=rng.uniform(*config.beta_magnitude_range),
        beta_angle=rng.uniform(angle_lo, angle_hi),
    )


def sample_batch(
    n_samples: int, config: FixedStack, rng: np.random.Generator
) -> SampleBatch:
    """Vectorized draw of ``n_samples`` following the exact staged sampling rules."""
    energy_ev = rng.uniform(*config.energy_ev_range, size=n_samples)
    theta_deg = rng.uniform(*config.theta_range_deg, size=(n_samples, config.n_q))
    theta_deg.sort(axis=1)

    film_thickness = rng.uniform(*config.film_thickness_range, size=n_samples)
    oxide_thickness = rng.uniform(*config.oxide_thickness_range, size=n_samples)

    roughness_lo, roughness_max = config.film_roughness_range
    roughness_hi = np.clip(film_thickness / TWO_PI, roughness_lo, roughness_max)
    film_roughness = rng.uniform(roughness_lo, roughness_hi)
    oxide_roughness = oxide_thickness / config.oxide_thickness_to_roughness

    angle_lo, angle_hi = np.deg2rad(config.angle_range_deg)
    return SampleBatch(
        energy_ev=energy_ev,
        theta_deg=theta_deg,
        film_thickness=film_thickness,
        film_roughness=film_roughness,
        oxide_thickness=oxide_thickness,
        oxide_roughness=oxide_roughness,
        delta_magnitude=rng.uniform(*config.delta_magnitude_range, size=n_samples),
        delta_angle=rng.uniform(angle_lo, angle_hi, size=n_samples),
        beta_magnitude=rng.uniform(*config.beta_magnitude_range, size=n_samples),
        beta_angle=rng.uniform(angle_lo, angle_hi, size=n_samples),
    )


def build_model(sample: Sample, config: FixedStack) -> ReflectModel:
    """Compose vacuum | film | oxide | silicon at ``sample.energy_ev``."""
    vacuum = MaterialSLD("", 0.0, name="vacuum")(0.0, 0.0)

    film = FreeUniTensor(energies=[sample.energy_ev], name="film")
    channel = film.channel_at(sample.energy_ev)
    channel.delta_o.setp(sample.delta_xx)  # ordinary, in-plane
    channel.beta_o.setp(sample.beta_xx)
    channel.delta_e.setp(sample.delta_zz)  # extraordinary, out-of-plane
    channel.beta_e.setp(sample.beta_zz)

    oxide = MaterialSLD(
        config.oxide_formula, config.oxide_density, name="oxide"
    )(sample.oxide_thickness, sample.oxide_roughness)
    silicon = MaterialSLD(
        config.si_formula, config.si_density, name="silicon"
    )(0.0, config.si_roughness)

    stack = (
        vacuum
        | film(sample.film_thickness, sample.film_roughness)
        | oxide
        | silicon
    )
    return ReflectModel(stack, energies=[sample.energy_ev], parallel=False)


def reflectivity(
    sample: Sample, config: FixedStack
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(q, R_s, R_p)`` power reflectivity for one sample."""
    model = build_model(sample, config)
    q = sample.q_values
    result = model(q, sample.energy_ev)
    return q, result.s, result.p


def simulate_batch(
    batch: SampleBatch,
    config: FixedStack,
    oxide_sld: MaterialSLD | None = None,
    si_sld: MaterialSLD | None = None,
    device: ResolvedDevice = "cpu",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate ``(q, R_s, R_p)`` for an entire ``SampleBatch`` via Rust TMM."""
    n = batch.size
    n_q = config.n_q
    if oxide_sld is None:
        oxide_sld = MaterialSLD(
            config.oxide_formula, config.oxide_density, name="oxide"
        )
    if si_sld is None:
        si_sld = MaterialSLD(
            config.si_formula, config.si_density, name="silicon"
        )

    oxide_tensors = oxide_sld.tensor_at_many(batch.energy_ev)
    si_tensors = si_sld.tensor_at_many(batch.energy_ev)

    n_xx = batch.delta_xx + 1j * batch.beta_xx
    n_zz = batch.delta_zz + 1j * batch.beta_zz
    n_avg = (2.0 * n_xx + n_zz) / 3.0

    layers = np.zeros((n, 4, 4), dtype=np.float64)
    tensors = np.zeros((n, 4, 3, 3), dtype=np.complex128)

    # Slab 1: anisotropic film (slab 0 is vacuum: all zeros).
    layers[:, 1, 0] = batch.film_thickness
    layers[:, 1, 1] = n_avg.real
    layers[:, 1, 2] = n_avg.imag
    layers[:, 1, 3] = batch.film_roughness
    tensors[:, 1, 0, 0] = n_xx
    tensors[:, 1, 1, 1] = n_xx
    tensors[:, 1, 2, 2] = n_zz

    # Slab 2: isotropic SiO2 oxide.
    layers[:, 2, 0] = batch.oxide_thickness
    layers[:, 2, 1] = oxide_tensors[:, 0, 0].real
    layers[:, 2, 2] = oxide_tensors[:, 0, 0].imag
    layers[:, 2, 3] = batch.oxide_roughness
    tensors[:, 2] = oxide_tensors

    # Slab 3: isotropic Si backing.
    layers[:, 3, 1] = si_tensors[:, 0, 0].real
    layers[:, 3, 2] = si_tensors[:, 0, 0].imag
    layers[:, 3, 3] = config.si_roughness
    tensors[:, 3] = si_tensors

    q = np.ascontiguousarray(batch.q_values, dtype=np.float64)
    stack_index = np.repeat(np.arange(n, dtype=np.int64), n_q)
    refl, _ = tmm.uniaxial_reflectivity_points(
        q.ravel(),
        stack_index,
        layers,
        tensors,
        batch.energy_ev,
        parallel=True,
        device=device,
    )
    refl = refl.reshape(n, n_q, 2, 2)
    r_p = refl[:, :, 0, 0]
    r_s = refl[:, :, 1, 1]
    return q, r_s, r_p


def parquet_schema(n_q: int) -> pa.Schema:
    """Optimized Arrow schema: non-nullable FixedSizeList(float32, n_q) + float32 scalars.

    ``float32`` preserves the ~12+ decade reflectivity dynamic range (where IEEE
    ``float16`` underflows to zero below ~6e-8) at half the footprint of ``float64``,
    and ``FixedSizeList`` omits per-row 32/64-bit list offset and validity buffers so
    columns load directly as contiguous ``(N, n_q)`` float32 tensors.
    """
    curve_type = pa.list_(pa.field("item", pa.float32(), nullable=False), n_q)
    fields = [
        pa.field("q", curve_type, nullable=False),
        pa.field("s", curve_type, nullable=False),
        pa.field("p", curve_type, nullable=False),
    ] + [pa.field(name, pa.float32(), nullable=False) for name in LABEL_NAMES]
    return pa.schema(fields)


def _batch_to_record_batch(
    q: NDArray[np.float64],
    r_s: NDArray[np.float64],
    r_p: NDArray[np.float64],
    batch: SampleBatch,
    schema: pa.Schema,
) -> pa.RecordBatch:
    """Pack simulated curves and Cartesian optical labels into a zero-offset Arrow RecordBatch."""
    curve_type = schema.field("q").type

    def fixed_f32_list(arr_2d: NDArray[np.float64]) -> pa.FixedSizeListArray:
        flat = np.ascontiguousarray(arr_2d, dtype=np.float32).ravel()
        return pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), type=curve_type
        )

    arrays: list[pa.Array] = [
        fixed_f32_list(q),
        fixed_f32_list(r_s),
        fixed_f32_list(r_p),
        pa.array(batch.film_thickness.astype(np.float32), type=pa.float32()),
        pa.array(batch.film_roughness.astype(np.float32), type=pa.float32()),
        pa.array(batch.oxide_thickness.astype(np.float32), type=pa.float32()),
        pa.array(batch.delta_xx.astype(np.float32), type=pa.float32()),
        pa.array(batch.delta_zz.astype(np.float32), type=pa.float32()),
        pa.array(batch.beta_xx.astype(np.float32), type=pa.float32()),
        pa.array(batch.beta_zz.astype(np.float32), type=pa.float32()),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _hf_size_category(n_samples: int) -> str:
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


def _dataset_readme(
    n_samples: int,
    n_shards: int,
    config: FixedStack,
    *,
    has_plots: bool = True,
) -> str:
    """Render Hugging Face dataset card with YAML metadata, figures, and loading examples."""
    size_cat = _hf_size_category(n_samples)
    total_points = n_samples * config.n_q

    figures_section = ""
    if has_plots:
        figures_section = f"""
---

## Example Sample

![Example s/p reflectivity curves and uniaxial optical-constant depth profile](plots/single_sample.png)

*Top: simulated `s`- and `p`-polarized power reflectivity `R_s(q)` and `R_p(q)`
sampled at {config.n_q} random grazing angles and cast to `q`. Bottom: corresponding
erf-smoothed uniaxial refractive-index depth profile (`1 - n = δ - iβ`) showing
in-plane (`δ_xx`, `β_xx`) and out-of-plane (`δ_zz`, `β_zz`) components across
the `vacuum | film | oxide | silicon` interfaces.*

---

## Dataset Distributions (`{n_samples:,}` Curves · `{total_points:,}` Points)

### Structural & Optical Parameter Priors

Sampling proceeds in two stages so every interface remains strictly inside the
Nevot–Croce validity regime (`thickness ≥ 2π · roughness`): film and native
oxide thicknesses are drawn first, followed by film roughness capped at
`min({config.film_roughness_range[1]:.0f} Å, t_film / 2π)` and oxide roughness coupled via `σ_oxide = t_oxide / 2π`.
Uniaxial optical constants are drawn in polar form (`component_xx = magnitude · cos(angle)`,
`component_zz = magnitude · sin(angle)`) and stored directly as Cartesian tensor
components (`delta_xx`, `delta_zz`, `beta_xx`, `beta_zz`).

![Parameter and optical tensor distributions across 1,000,000 samples](plots/dry_run_parameters.png)

### Sampled `q` Grid & Fresnel-Compensated `R · q⁴` Densities

Uniform sampling over `(θ, E)` produces dense coverage at low-to-intermediate
`q` with a smooth high-`q` tail out to `~0.46 Å⁻¹`, while `p` polarization
exhibits broader dynamic range than `s` polarization due to Brewster-angle and
birefringent contrast.

![Sampled q distribution and Fresnel-compensated R * q^4 joint densities](plots/dry_run_reflectivity_q4.png)
"""

    return f"""---
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
  - {size_cat}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# Polarized Soft X-Ray Reflectivity Dataset (`reflectivity-1m`)

Synthetic polarized soft X-ray reflectivity (`s` and `p` channels) simulated with
`refloxide` for uniaxial anisotropic thin films on native-oxide silicon:

```text
vacuum | anisotropic film | SiO2 oxide | Si substrate
```

Each curve mimics an experimental angle scan at an arbitrary soft X-ray photon
energy: `{config.n_q}` random grazing angles `θ ∈ [{config.theta_range_deg[0]}°, {config.theta_range_deg[1]}°]` are drawn and sorted
ascending, evaluated at a randomized photon energy `E ∈ [{config.energy_ev_range[0]:.0f}, {config.energy_ev_range[1]:.0f}] eV`, and cast
into momentum transfer `q = (4π / λ) sin θ` (`1/Å`). Photon energy is omitted
from the saved labels so downstream models learn an **energy-invariant** mapping
from `(q, R_s, R_p)` to structural and Cartesian uniaxial optical tensor parameters
(`delta_xx`, `delta_zz`, `beta_xx`, `beta_zz`).
{figures_section}
---

## Schema & Storage Layout

- **Total rows**: `{n_samples:,}` across `{n_shards}` Parquet shard(s) (`data/train-*.parquet`). All rows are stored in a single `train` split so train/validation/test splits can be chosen post-hoc.
- **Substrate constants**: `Si` (`ρ = {config.si_density} g/cm³`, `σ = {config.si_roughness} Å`), `SiO2` (`ρ = {config.oxide_density} g/cm³`, `σ_oxide = t_oxide / 2π`).
- **Optimized datatypes**: Non-nullable `FixedSizeList(float32, {config.n_q})` curves + `float32` scalars encoded with `BYTE_STREAM_SPLIT` + `ZSTD`—eliminating per-row list-offset and validity-bitmap overhead while preserving ~38 decades of dynamic range (where `float16` underflows below `~6e-8`).

| Column | Arrow Type | Units / Range | Description |
| :--- | :--- | :--- | :--- |
| `q` | `fixed_size_list<float32>[{config.n_q}]` | `1/Å` | Ascending momentum transfer grid cast from `(θ, E)` |
| `s` | `fixed_size_list<float32>[{config.n_q}]` | power reflectance | `s`-polarized reflectivity `R_s(q)` |
| `p` | `fixed_size_list<float32>[{config.n_q}]` | power reflectance | `p`-polarized reflectivity `R_p(q)` |
| `film_thickness` | `float32` | `Å`, `[{config.film_thickness_range[0]}, {config.film_thickness_range[1]}]` | Anisotropic film thickness `t_film` |
| `film_roughness` | `float32` | `Å`, `[0.0, min({config.film_roughness_range[1]}, t_film / 2π)]` | Vacuum/film RMS roughness `σ_film` (Nevot–Croce capped) |
| `oxide_thickness` | `float32` | `Å`, `[{config.oxide_thickness_range[0]}, {config.oxide_thickness_range[1]}]` | Native `SiO2` thickness `t_oxide` (`σ_oxide = t_oxide / 2π`) |
| `delta_xx` | `float32` | `[{config.delta_magnitude_range[0]:.1e}, {config.delta_magnitude_range[1]:.1e}]` | In-plane (ordinary) dispersion `δ_xx = δ cos α` |
| `delta_zz` | `float32` | `[{config.delta_magnitude_range[0]:.1e}, {config.delta_magnitude_range[1]:.1e}]` | Out-of-plane (extraordinary) dispersion `δ_zz = δ sin α` |
| `beta_xx` | `float32` | `[0.0, {config.beta_magnitude_range[1]:.1e}]` | In-plane (ordinary) absorption `β_xx = β cos γ` |
| `beta_zz` | `float32` | `[0.0, {config.beta_magnitude_range[1]:.1e}]` | Out-of-plane (extraordinary) absorption `β_zz = β sin γ` |

---

## Loading the Data

### Polars -> Contiguous `float32` NumPy / PyTorch Arrays

Because `q`, `s`, and `p` are stored as fixed-size non-nullable `float32` arrays,
Polars materializes them directly into C-contiguous `(N, {config.n_q})` tensors:

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

df = pl.read_parquet("hf://datasets/{DEFAULT_HF_REPO_ID}/data/train-*.parquet")

q = df["q"].to_numpy()  # ({n_samples:_}, {config.n_q}) float32, C-contiguous
s = df["s"].to_numpy()  # ({n_samples:_}, {config.n_q}) float32, C-contiguous
p = df["p"].to_numpy()  # ({n_samples:_}, {config.n_q}) float32, C-contiguous

inputs = np.stack([s, p, q], axis=1)        # ({n_samples:_}, 3, {config.n_q}) float32
targets = df.select(LABEL_COLS).to_numpy()  # ({n_samples:_}, 7) float32
```

### Hugging Face `datasets` (with Post-Hoc Train/Validation Split)

```python
from datasets import load_dataset

ds = load_dataset("{DEFAULT_HF_REPO_ID}", split="train").with_format("torch")
splits = ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = splits["train"], splits["test"]

batch = train_ds[:64]
q, s, p = batch["q"], batch["s"], batch["p"]  # each torch.float32 of shape (64, {config.n_q})
```

### Streaming Batches with PyArrow (Low Memory)

```python
from huggingface_hub import snapshot_download
import pyarrow.dataset as ds

local_dir = snapshot_download(
    repo_id="{DEFAULT_HF_REPO_ID}",
    repo_type="dataset",
    allow_patterns="data/train-*.parquet",
)

dataset = ds.dataset(f"{{local_dir}}/data", format="parquet")
for record_batch in dataset.to_batches(batch_size=4096):
    q = record_batch["q"].values.to_numpy().reshape(-1, {config.n_q})
    s = record_batch["s"].values.to_numpy().reshape(-1, {config.n_q})
    p = record_batch["p"].values.to_numpy().reshape(-1, {config.n_q})
```
"""


def _sync_readme_plots(dataset_dir: Path, plot_dir: Path) -> bool:
    """Copy existing plot PNGs into ``dataset_dir / 'plots'`` if all three are available."""
    dataset_plots = dataset_dir / "plots"
    if all((dataset_plots / name).is_file() for name in README_PLOT_FILES):
        return True
    if all((plot_dir / name).is_file() for name in README_PLOT_FILES):
        dataset_plots.mkdir(parents=True, exist_ok=True)
        for name in README_PLOT_FILES:
            shutil.copy2(plot_dir / name, dataset_plots / name)
        return True
    return False


def generate_parquet_dataset(
    n_samples: int,
    config: FixedStack,
    rng: np.random.Generator,
    dataset_dir: Path,
    *,
    shard_size: int = 100_000,
    chunk_size: int = 25_000,
    device: ResolvedDevice = "gpu",
    plot_dir: Path = Path("plots"),
) -> list[Path]:
    """Stream ``n_samples`` curves directly into sharded, Zstd+BSS Parquet files."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")

    t0 = time.perf_counter()
    dataset_dir = Path(dataset_dir)
    data_subdir = dataset_dir / "data"
    data_subdir.mkdir(parents=True, exist_ok=True)

    # Remove stale shards from earlier runs so shard counts never mix.
    for old_shard in data_subdir.glob("train-*.parquet"):
        old_shard.unlink()

    schema = parquet_schema(config.n_q)
    n_shards = max(1, math.ceil(n_samples / shard_size))
    oxide_sld = MaterialSLD(
        config.oxide_formula, config.oxide_density, name="oxide"
    )
    si_sld = MaterialSLD(config.si_formula, config.si_density, name="silicon")

    written_shards: list[Path] = []
    total_bytes = 0

    for shard_idx in range(n_shards):
        shard_start = shard_idx * shard_size
        shard_end = min(shard_start + shard_size, n_samples)
        shard_path = data_subdir / f"train-{shard_idx:05d}-of-{n_shards:05d}.parquet"

        with pq.ParquetWriter(
            shard_path,
            schema,
            compression="zstd",
            compression_level=3,
            use_byte_stream_split=True,
            use_dictionary=False,
        ) as writer:
            for start in range(shard_start, shard_end, chunk_size):
                end = min(start + chunk_size, shard_end)
                batch = sample_batch(end - start, config, rng)
                q, r_s, r_p = simulate_batch(
                    batch, config, oxide_sld, si_sld, device=device
                )
                record_batch = _batch_to_record_batch(q, r_s, r_p, batch, schema)
                writer.write_batch(record_batch)

        shard_bytes = shard_path.stat().st_size
        total_bytes += shard_bytes
        written_shards.append(shard_path)
        elapsed = time.perf_counter() - t0
        print(
            f"[{shard_idx + 1}/{n_shards}] wrote {shard_path.name} "
            f"({shard_end:,}/{n_samples:,} rows, {shard_bytes / 1e6:.1f} MB, "
            f"elapsed {elapsed:.1f}s)",
            flush=True,
        )

    has_plots = _sync_readme_plots(dataset_dir, Path(plot_dir))
    readme_path = dataset_dir / "README.md"
    readme_path.write_text(
        _dataset_readme(
            n_samples=n_samples,
            n_shards=n_shards,
            config=config,
            has_plots=has_plots,
        ),
        encoding="utf-8",
    )
    print(
        f"finished local dataset in {dataset_dir} "
        f"({n_samples:,} rows across {n_shards} shard(s), "
        f"{total_bytes / 1e9:.2f} GB total)",
        flush=True,
    )
    return written_shards


def upload_dataset_to_hub(
    dataset_dir: Path,
    repo_id: str = DEFAULT_HF_REPO_ID,
    *,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Create/verify the Hugging Face dataset repo and stream-upload ``dataset_dir``."""
    dataset_dir = Path(dataset_dir)
    shards = sorted((dataset_dir / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No Parquet shards matching 'data/train-*.parquet' found in {dataset_dir}"
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        allow_patterns=["README.md", "plots/*.png", "data/train-*.parquet"],
        delete_patterns=["data/*.parquet"],
        commit_message=f"Upload polarized reflectivity dataset ({len(shards)} shard(s))",
    )
    return str(commit_info.commit_url)


def _accumulate_uniform_2d_hist(
    target: NDArray[np.int64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> None:
    """Fast in-place uniform 2D histogram accumulation via ``np.bincount``."""
    nx, ny = target.shape
    ix = np.clip(
        ((x - x_lo) * (nx / (x_hi - x_lo))).astype(np.int64), 0, nx - 1
    )
    iy = np.clip(
        ((y - y_lo) * (ny / (y_hi - y_lo))).astype(np.int64), 0, ny - 1
    )
    counts = np.bincount(ix * ny + iy, minlength=nx * ny)
    target += counts.reshape(nx, ny)


def run_dry_run(
    n_samples: int,
    config: FixedStack,
    rng: np.random.Generator,
    plot_dir: Path,
    chunk_size: int = 50_000,
    device: ResolvedDevice = "gpu",
) -> tuple[Path, Path]:
    """Stream ``n_samples`` curves without storing them and plot parameter & R*q^4 distributions."""
    t0 = time.perf_counter()

    # Parameter arrays (only ~8 scalars/sample -> ~32 MB at 1M samples).
    film_t = np.empty(n_samples, dtype=np.float32)
    film_sigma = np.empty(n_samples, dtype=np.float32)
    oxide_t = np.empty(n_samples, dtype=np.float32)
    delta_mag = np.empty(n_samples, dtype=np.float32)
    delta_deg = np.empty(n_samples, dtype=np.float32)
    beta_mag = np.empty(n_samples, dtype=np.float32)
    beta_deg = np.empty(n_samples, dtype=np.float32)
    energy_ev = np.empty(n_samples, dtype=np.float32)

    # Uniform 1D/2D streaming histograms for all (n_samples * n_q) reflectivity points.
    q_max_phys = (
        4.0
        * math.pi
        * config.energy_ev_range[1]
        / HC_EV_ANGSTROM
        * math.sin(math.radians(config.theta_range_deg[1]))
    )
    q_lo, q_hi, n_q_bins = 0.0, q_max_phys * 1.02, 220
    y_lo, y_hi, n_y_bins = -15.5, -4.5, 220
    q_edges = np.linspace(q_lo, q_hi, n_q_bins + 1)
    log_rq4_edges = np.linspace(y_lo, y_hi, n_y_bins + 1)
    rq4_floor = 10.0**y_lo

    rs_q4_2d = np.zeros((n_q_bins, n_y_bins), dtype=np.int64)
    rp_q4_2d = np.zeros((n_q_bins, n_y_bins), dtype=np.int64)

    oxide_sld = MaterialSLD(
        config.oxide_formula, config.oxide_density, name="oxide"
    )
    si_sld = MaterialSLD(config.si_formula, config.si_density, name="silicon")

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch = sample_batch(end - start, config, rng)
        q, r_s, r_p = simulate_batch(
            batch, config, oxide_sld, si_sld, device=device
        )

        film_t[start:end] = batch.film_thickness
        film_sigma[start:end] = batch.film_roughness
        oxide_t[start:end] = batch.oxide_thickness
        delta_mag[start:end] = batch.delta_magnitude
        delta_deg[start:end] = np.rad2deg(batch.delta_angle)
        beta_mag[start:end] = batch.beta_magnitude
        beta_deg[start:end] = np.rad2deg(batch.beta_angle)
        energy_ev[start:end] = batch.energy_ev

        q_flat = q.ravel()
        q4 = q_flat * q_flat
        q4 *= q4
        log_rs_q4 = np.log10(np.maximum(r_s.ravel() * q4, rq4_floor))
        log_rp_q4 = np.log10(np.maximum(r_p.ravel() * q4, rq4_floor))

        _accumulate_uniform_2d_hist(
            rs_q4_2d, q_flat, log_rs_q4, q_lo, q_hi, y_lo, y_hi
        )
        _accumulate_uniform_2d_hist(
            rp_q4_2d, q_flat, log_rp_q4, q_lo, q_hi, y_lo, y_hi
        )

        elapsed = time.perf_counter() - t0
        print(
            f"dry-run progress: {end:,}/{n_samples:,} samples "
            f"({end * config.n_q:,} q-points) in {elapsed:.1f}s",
            flush=True,
        )

    q_hist = rs_q4_2d.sum(axis=1)
    rs_q4_hist = rs_q4_2d.sum(axis=0)
    rp_q4_hist = rp_q4_2d.sum(axis=0)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    param_plot = _plot_parameter_distributions(
        n_samples=n_samples,
        config=config,
        film_t=film_t,
        film_sigma=film_sigma,
        oxide_t=oxide_t,
        delta_mag=delta_mag,
        delta_deg=delta_deg,
        beta_mag=beta_mag,
        beta_deg=beta_deg,
        energy_ev=energy_ev,
        output=plot_dir / "dry_run_parameters.png",
    )
    rq4_plot = _plot_rq4_distributions(
        n_samples=n_samples,
        n_q=config.n_q,
        q_edges=q_edges,
        log_rq4_edges=log_rq4_edges,
        q_hist=q_hist,
        rs_q4_hist=rs_q4_hist,
        rp_q4_hist=rp_q4_hist,
        rs_q4_2d=rs_q4_2d,
        rp_q4_2d=rp_q4_2d,
        output=plot_dir / "dry_run_reflectivity_q4.png",
    )
    return param_plot, rq4_plot


def _hist_quantiles_2d(
    hist_2d: NDArray[np.int64],
    y_edges: NDArray[np.float64],
    quantiles: Sequence[float],
) -> NDArray[np.float64]:
    """Compute approximate y-quantiles per x-bin from a 2D histogram."""
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    totals = hist_2d.sum(axis=1)
    out = np.full((len(quantiles), hist_2d.shape[0]), np.nan, dtype=np.float64)
    cdf = np.cumsum(hist_2d, axis=1)
    for ix in range(hist_2d.shape[0]):
        if totals[ix] == 0:
            continue
        norm_cdf = cdf[ix] / totals[ix]
        out[:, ix] = np.interp(quantiles, norm_cdf, y_centers)
    return out


def _plot_parameter_distributions(
    *,
    n_samples: int,
    config: FixedStack,
    film_t: NDArray[np.float32],
    film_sigma: NDArray[np.float32],
    oxide_t: NDArray[np.float32],
    delta_mag: NDArray[np.float32],
    delta_deg: NDArray[np.float32],
    beta_mag: NDArray[np.float32],
    beta_deg: NDArray[np.float32],
    energy_ev: NDArray[np.float32],
    output: Path,
) -> Path:
    """Plot 1D and coupled 2D parameter distributions for the dry run."""
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), layout="constrained")
    bins = 80

    ax = axes[0, 0]
    ax.hist(film_t, bins=bins, density=True, color="tab:blue", alpha=0.75)
    ax.set_xlabel(r"film thickness $t_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_ylabel("density")
    ax.set_title("film thickness")

    ax = axes[0, 1]
    ax.hist(film_sigma, bins=bins, density=True, color="tab:cyan", alpha=0.75)
    ax.set_xlabel(r"film roughness $\sigma_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_ylabel("density")
    ax.set_title("film roughness (Nevot–Croce capped)")

    ax = axes[0, 2]
    h2d, xedges, yedges = np.histogram2d(film_t, film_sigma, bins=80)
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_less_equal(h2d.T, 0),
        cmap="viridis",
    )
    t_line = np.linspace(*config.film_thickness_range, 200)
    ax.plot(
        t_line,
        np.minimum(t_line / TWO_PI, config.film_roughness_range[1]),
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=r"$\min(t / 2\pi,\; \sigma_{\max})$",
    )
    fig.colorbar(mesh, ax=ax, label="count")
    ax.set_xlabel(r"$t_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_ylabel(r"$\sigma_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_title("staged thickness–roughness coupling")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    ax.hist(oxide_t, bins=bins, density=True, color="tab:olive", alpha=0.75)
    ax.set_xlabel(r"oxide thickness $t_{\mathrm{oxide}}$ [$\mathrm{\AA}$]")
    ax.set_ylabel("density")
    ax.set_title(r"oxide thickness ($\sigma_{\mathrm{ox}} = t_{\mathrm{ox}} / 2\pi$)")

    ax = axes[1, 1]
    ax.hist(
        delta_mag * 1e3, bins=bins, density=True, color="tab:purple", alpha=0.75
    )
    ax.set_xlabel(r"$\delta$ magnitude [$\times 10^{-3}$]")
    ax.set_ylabel("density")
    ax.set_title(r"$\delta$ signed magnitude")

    ax = axes[1, 2]
    ax.hist(
        beta_mag * 1e3, bins=bins, density=True, color="tab:orange", alpha=0.75
    )
    ax.set_xlabel(r"$|\beta|$ magnitude [$\times 10^{-3}$]")
    ax.set_ylabel("density")
    ax.set_title(r"$\beta$ magnitude")

    ax = axes[2, 0]
    ax.hist(
        delta_deg,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        color="tab:purple",
        label=r"$\delta$ angle",
    )
    ax.hist(
        beta_deg,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.6,
        color="tab:orange",
        linestyle="--",
        label=r"$\beta$ angle",
    )
    ax.set_xlabel("tensor angle [deg]")
    ax.set_ylabel("density")
    ax.set_title(r"$\delta$ and $\beta$ orientation angles")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2, 1]
    d_rad = np.deg2rad(delta_deg)
    b_rad = np.deg2rad(beta_deg)
    ax.hist(
        (delta_mag * np.cos(d_rad)) * 1e3,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="tab:blue",
        label=r"$\delta_{xx}$",
    )
    ax.hist(
        (delta_mag * np.sin(d_rad)) * 1e3,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color="tab:red",
        label=r"$\delta_{zz}$",
    )
    ax.hist(
        (beta_mag * np.cos(b_rad)) * 1e3,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        linestyle="--",
        color="tab:blue",
        label=r"$\beta_{xx}$",
    )
    ax.hist(
        (beta_mag * np.sin(b_rad)) * 1e3,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        linestyle="--",
        color="tab:red",
        label=r"$\beta_{zz}$",
    )
    ax.set_xlabel(r"tensor component [$\times 10^{-3}$]")
    ax.set_ylabel("density")
    ax.set_title("stored uniaxial tensor components")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[2, 2]
    ax.hist(energy_ev, bins=bins, density=True, color="0.45", alpha=0.8)
    ax.set_xlabel("photon energy [eV]")
    ax.set_ylabel("density")
    ax.set_title("photon energy (not trained)")

    for row in axes:
        for subplot in row:
            subplot.grid(alpha=0.25)

    fig.suptitle(
        f"Dry-Run Parameter Distributions ({n_samples:,} samples)",
        fontsize=13,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_rq4_distributions(
    *,
    n_samples: int,
    n_q: int,
    q_edges: NDArray[np.float64],
    log_rq4_edges: NDArray[np.float64],
    q_hist: NDArray[np.int64],
    rs_q4_hist: NDArray[np.int64],
    rp_q4_hist: NDArray[np.int64],
    rs_q4_2d: NDArray[np.int64],
    rp_q4_2d: NDArray[np.int64],
    output: Path,
) -> Path:
    """Plot q, R_s * q^4, and R_p * q^4 1D/2D distributions across all points."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), layout="constrained")
    total_points = n_samples * n_q

    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    dq = np.diff(q_edges)
    q_density = q_hist / (total_points * dq)

    ax = axes[0, 0]
    ax.fill_between(q_centers, q_density, step="mid", color="tab:gray", alpha=0.35)
    ax.plot(q_centers, q_density, color="0.25", linewidth=1.5)
    ax.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
    ax.set_ylabel("probability density")
    ax.set_title(
        r"sampled $q$ point distribution ($\theta \in [0.5^\circ, 65^\circ]$, $E \in [200, 500]$ eV)"
    )
    ax.grid(alpha=0.3)

    log_centers = 0.5 * (log_rq4_edges[:-1] + log_rq4_edges[1:])
    dlog = np.diff(log_rq4_edges)
    rs_density = rs_q4_hist / (total_points * dlog)
    rp_density = rp_q4_hist / (total_points * dlog)

    ax = axes[0, 1]
    ax.plot(
        log_centers,
        rs_density,
        color="tab:blue",
        linewidth=1.8,
        label=r"$R_s \cdot q^4$",
    )
    ax.fill_between(log_centers, rs_density, color="tab:blue", alpha=0.2)
    ax.plot(
        log_centers,
        rp_density,
        color="tab:red",
        linewidth=1.8,
        label=r"$R_p \cdot q^4$",
    )
    ax.fill_between(log_centers, rp_density, color="tab:red", alpha=0.2)
    ax.set_xlabel(r"$\log_{10}(R \cdot q^4\ [\mathrm{\AA}^{-4}])$")
    ax.set_ylabel("probability density")
    ax.set_title(r"$R_s \cdot q^4$ and $R_p \cdot q^4$ marginal distributions")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    vmax = float(max(rs_q4_2d.max(), rp_q4_2d.max(), 1))
    norm = LogNorm(vmin=1.0, vmax=vmax)

    for ax_2d, hist_2d, title, line_color in (
        (axes[1, 0], rs_q4_2d, r"$R_s \cdot q^4$ vs $q$ joint density", "cyan"),
        (axes[1, 1], rp_q4_2d, r"$R_p \cdot q^4$ vs $q$ joint density", "gold"),
    ):
        mesh = ax_2d.pcolormesh(
            q_edges,
            log_rq4_edges,
            np.ma.masked_less_equal(hist_2d.T, 0),
            cmap="magma",
            norm=norm,
        )
        q05, q50, q95 = _hist_quantiles_2d(
            hist_2d, log_rq4_edges, (0.05, 0.50, 0.95)
        )
        ax_2d.plot(q_centers, q50, color=line_color, linewidth=1.6, label="median")
        ax_2d.plot(
            q_centers,
            q05,
            color=line_color,
            linewidth=1.0,
            linestyle="--",
            label="5% / 95%",
        )
        ax_2d.plot(q_centers, q95, color=line_color, linewidth=1.0, linestyle="--")
        fig.colorbar(mesh, ax=ax_2d, label="point count (log scale)")
        ax_2d.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
        ax_2d.set_ylabel(r"$\log_{10}(R \cdot q^4\ [\mathrm{\AA}^{-4}])$")
        ax_2d.set_title(title)
        ax_2d.legend(loc="lower left", fontsize=8)
        ax_2d.grid(alpha=0.25)

    fig.suptitle(
        f"Dry-Run Reflectivity × $q^4$ & $q$ Distributions "
        f"({n_samples:,} curves, {total_points:,} points)",
        fontsize=13,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


@dataclass(frozen=True)
class DepthProfile:
    """Erf-smoothed optical-constant depth profile of the composed stack."""

    z: NDArray[np.float64]  # Angstrom, 0 at the film surface
    boundaries: NDArray[np.float64]  # interface depths
    interface_roughness: NDArray[np.float64]  # RMS roughness per interface
    layer_values: dict[str, NDArray[np.float64]]  # per-layer tensor components
    curves: dict[str, NDArray[np.float64]]  # smeared tensor components vs z


def depth_profile(
    model: ReflectModel,
    sample: Sample,
    n_points: int = 2000,
    pad_above: float = 15.0,
    pad_below: float = 15.0,
) -> DepthProfile:
    """Erf-smoothed depth profile straight from the composed structure."""
    rows, tensors = model.structure.materialize_at(sample.energy_ev)

    thicknesses = rows[1:, 0]
    boundaries = np.concatenate([[0.0], np.cumsum(thicknesses)])[:-1]
    interface_roughness = rows[1:, 3]

    layer_values = {
        "delta_xx": tensors[:, 0, 0].real,
        "delta_zz": tensors[:, 2, 2].real,
        "beta_xx": tensors[:, 0, 0].imag,
        "beta_zz": tensors[:, 2, 2].imag,
    }

    z = np.linspace(-pad_above, boundaries[-1] + pad_below, n_points)
    erf = np.vectorize(math.erf, otypes=[np.float64])
    curves: dict[str, NDArray[np.float64]] = {}
    for key, values in layer_values.items():
        profile = np.full_like(z, values[0])
        for i, (boundary, roughness) in enumerate(
            zip(boundaries, interface_roughness)
        ):
            if roughness > 0.0:
                step = 0.5 * (
                    1.0 + erf((z - boundary) / (math.sqrt(2.0) * roughness))
                )
            else:
                step = (z >= boundary).astype(np.float64)
            profile = profile + (values[i + 1] - values[i]) * step
        curves[key] = profile
    return DepthProfile(
        z=z,
        boundaries=boundaries,
        interface_roughness=interface_roughness,
        layer_values=layer_values,
        curves=curves,
    )


def plot_sample(
    sample: Sample,
    q: NDArray[np.float64],
    r_s: NDArray[np.float64],
    r_p: NDArray[np.float64],
    profile: DepthProfile,
    output: Path,
) -> Path:
    """Plot the s/p reflectivity pair and the optical-constant depth profile."""
    figure, (ax_reflectivity, ax_depth) = plt.subplots(
        2, 1, figsize=(9.0, 9.0), layout="constrained"
    )

    ax_reflectivity.plot(
        q,
        r_s,
        label="s polarization",
        color="tab:blue",
        linestyle="",
        marker="o",
        markersize=4,
        markerfacecolor="none",
    )
    ax_reflectivity.plot(
        q,
        r_p,
        label="p polarization",
        color="tab:red",
        linestyle="",
        marker="o",
        markersize=4,
        markerfacecolor="none",
    )
    ax_reflectivity.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
    ax_reflectivity.set_yscale("log")
    ax_reflectivity.set_ylabel("reflectivity")
    ax_reflectivity.set_title(
        f"s and p reflectivity at {sample.energy_ev:.1f} eV "
        f"($\\theta \\in [{sample.theta_deg[0]:.1f}^\\circ, {sample.theta_deg[-1]:.1f}^\\circ]$)"
    )
    ax_reflectivity.grid(which="both", alpha=0.3)
    ax_reflectivity.legend(loc="upper right")

    z = profile.z
    colors = {"xx": "tab:blue", "zz": "tab:red"}
    ax_depth.axhline(0.0, color="0.6", linewidth=0.6, linestyle=":")
    for component in ("xx", "zz"):
        for quantity, linestyle in (("delta", "-"), ("beta", "--")):
            ax_depth.plot(
                z,
                profile.curves[f"{quantity}_{component}"],
                color=colors[component],
                linestyle=linestyle,
                label=rf"$\{quantity}_{{{component}}}$",
            )

    boundaries = profile.boundaries
    regions = ["vacuum", "film", "oxide", "silicon"]
    edges = np.concatenate([[z[0]], boundaries, [z[-1]]])
    shades = ("0.97", "0.92", "0.84", "0.76")
    xaxis_transform = ax_depth.get_xaxis_transform()
    for i, (region, shade) in enumerate(zip(regions, shades)):
        ax_depth.axvspan(edges[i], edges[i + 1], color=shade, zorder=0)
        y_pos = 0.90 if region == "oxide" else 0.96
        ax_depth.text(
            0.5 * (edges[i] + edges[i + 1]),
            y_pos,
            region,
            ha="center",
            va="top",
            transform=xaxis_transform,
        )
    for i, (boundary, roughness) in enumerate(
        zip(boundaries, profile.interface_roughness)
    ):
        ax_depth.axvline(boundary, color="0.5", linewidth=0.8)
        align_left = i == 1
        ax_depth.annotate(
            f"σ {roughness:.1f} Å",
            (boundary, 0.02),
            xycoords=xaxis_transform,
            xytext=(-3 if align_left else 3, 2),
            textcoords="offset points",
            ha="right" if align_left else "left",
            fontsize=8,
            color="0.35",
        )
    ax_depth.set_xlabel(r"depth [$\mathrm{\AA}$]")
    ax_depth.set_ylabel(r"$\delta$, $\beta$")
    ax_depth.set_title("optical constant depth profile (tensor components)")
    ax_depth.margins(y=0.12)
    ax_depth.grid(which="both", alpha=0.3)
    ax_depth.legend(loc="lower right")

    figure.suptitle(
        f"E {sample.energy_ev:.1f} eV  |  "
        f"film {sample.film_thickness:.0f} Å (σ {sample.film_roughness:.1f} Å)  |  "
        f"δ ({sample.delta_xx:+.2e}, {sample.delta_zz:+.2e})  |  "
        f"β ({sample.beta_xx:.2e}, {sample.beta_zz:.2e})  |  "
        f"oxide {sample.oxide_thickness:.1f} Å (σ {sample.oxide_roughness:.1f} Å)"
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def describe_sample(sample: Sample, config: FixedStack) -> str:
    """Human-readable summary of one random draw and the fixed stack."""
    q = sample.q_values
    return (
        f"random sample (staged: thicknesses first, roughnesses second):\n"
        f"  photon energy      {sample.energy_ev:8.2f} eV  (not trained)\n"
        f"  theta grid ({config.n_q})   [{sample.theta_deg[0]:.2f}°, {sample.theta_deg[-1]:.2f}°]"
        f"  ->  q in [{q[0]:.4f}, {q[-1]:.4f}] 1/Å\n"
        f"  film thickness     {sample.film_thickness:8.1f} Å\n"
        f"  film roughness     {sample.film_roughness:8.1f} Å"
        f"  (NC limit {sample.film_thickness / TWO_PI:.1f} Å)\n"
        f"  oxide thickness    {sample.oxide_thickness:8.1f} Å\n"
        f"  oxide roughness    {sample.oxide_roughness:8.1f} Å"
        f"  (NC limit {sample.oxide_thickness / TWO_PI:.1f} Å)\n"
        f"  delta (stored)     δ_xx {sample.delta_xx:+.3e}, δ_zz {sample.delta_zz:+.3e}"
        f"  (polar {sample.delta_magnitude:+.3e} @ {np.rad2deg(sample.delta_angle):5.1f}°)\n"
        f"  beta  (stored)     β_xx {sample.beta_xx:.3e}, β_zz {sample.beta_zz:.3e}"
        f"  (polar {sample.beta_magnitude:.3e} @ {np.rad2deg(sample.beta_angle):5.1f}°)\n"
        f"fixed substrate: Si (ρ {config.si_density}, σ {config.si_roughness} Å),"
        f" SiO2 (ρ {config.oxide_density}),"
        f" oxide t/σ ratio {config.oxide_thickness_to_roughness:.2f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a single preview plot, dry-run distributions, or Parquet dataset + HF upload."""
    parser = argparse.ArgumentParser(
        description="Generate anisotropic reflectivity samples in theta space and cast to q."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / "single_sample.png",
        help="output plot path for single-sample preview mode",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plots"),
        help="output directory for --dry-run distribution plots",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only generate parameter and R*q^4/q distribution plots without saving data",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="number of samples to generate; without --dry-run, bypasses plots, writes Parquet shards, and uploads to Hugging Face",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "reflectivity-1m",
        help="local directory for generated Parquet shards and dataset card",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="rows per Parquet shard (default: 100,000 -> 10 shards for 1M rows)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help=f"Hugging Face dataset repository ID (default: {DEFAULT_HF_REPO_ID})",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (defaults to cached login or HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the Hugging Face dataset repository as private if it does not exist",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="generate local Parquet shards without uploading to Hugging Face",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="upload existing Parquet shards in --dataset-dir to Hugging Face without regenerating",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="TMM execution device for batched runs (default: auto -> gpu if available)",
    )
    args = parser.parse_args(argv)

    config = FixedStack()
    rng = np.random.default_rng(args.seed)

    if args.dry_run:
        device = resolve_device(args.device)
        n_samples = 1_000_000 if args.num_samples is None else args.num_samples
        param_plot, rq4_plot = run_dry_run(
            n_samples=n_samples,
            config=config,
            rng=rng,
            plot_dir=args.plot_dir,
            device=device,
        )
        print(f"saved {param_plot}")
        print(f"saved {rq4_plot}")
        return 0

    if args.num_samples is not None or args.upload_only:
        if not args.upload_only:
            assert args.num_samples is not None
            device = resolve_device(args.device)
            generate_parquet_dataset(
                n_samples=args.num_samples,
                config=config,
                rng=rng,
                dataset_dir=args.dataset_dir,
                shard_size=args.shard_size,
                device=device,
                plot_dir=args.plot_dir,
            )

        if args.no_upload:
            return 0

        print(
            f"uploading {args.dataset_dir} to https://huggingface.co/datasets/{args.repo_id} ...",
            flush=True,
        )
        try:
            commit_url = upload_dataset_to_hub(
                dataset_dir=args.dataset_dir,
                repo_id=args.repo_id,
                token=args.hf_token,
                private=args.private,
            )
            print(f"uploaded dataset: {commit_url}")
            return 0
        except (LocalTokenNotFoundError, HfHubHTTPError, FileNotFoundError) as exc:
            print(
                f"Hugging Face upload failed ({exc}).\n"
                f"Local Parquet shards are preserved in '{args.dataset_dir}'.\n"
                f"Authenticate with `uv run hf auth login` (or set HF_TOKEN) and resume via:\n"
                f"  uv run python scripts/generate_reflectivity.py --upload-only --dataset-dir {args.dataset_dir} --repo-id {args.repo_id}"
            )
            return 1

    sample = random_sample(config, rng)
    print(describe_sample(sample, config))
    q, r_s, r_p = reflectivity(sample, config)
    model = build_model(sample, config)
    profile = depth_profile(model, sample)
    output = plot_sample(sample, q, r_s, r_p, profile, args.output)
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
