"""Generate simulated polarized soft X-ray reflectivity curves for training.

    uv run python scripts/generate_reflectivity.py                   # preview one random sample
    uv run python scripts/generate_reflectivity.py --dry-run         # distribution plots, nothing saved
    uv run python scripts/generate_reflectivity.py -n 1000000        # write the dataset + upload it
    uv run python scripts/generate_reflectivity.py -n 1000 --no-upload   # small local dataset
    uv run python scripts/generate_reflectivity.py --help            # list every option

Every option has a default in the DEFAULT SETTINGS block. Edit a value there to change
it permanently, or pass the matching command-line flag to change it for one run.

Each simulated sample is a stack of four layers, from top to bottom:

    vacuum | film | native SiO2 oxide | Si substrate

How one batch of samples is made:
    1. Measurement: every sample uses the same photon energy and the same grid of
       N_Q momentum-transfer values q (``sample_measurement``).
    2. Sample: pick the film thickness, roughness and optical constants, plus the
       oxide thickness (``sample_parameters``).
    3. Stack: put every layer's numbers into the arrays the simulator reads
       (``build_stack``).
    4. Simulate: compute the s- and p-polarized reflectivity at every q
       (``simulate_reflectivity``).

The film is uniaxial: it has one set of optical constants in the plane of the film
(``xx``) and another perpendicular to it (``zz``). Each pair is drawn in polar form
(a magnitude and an angle) and stored as Cartesian components
(``delta_xx``, ``delta_zz``, ``beta_xx``, ``beta_zz``). The refractive index of a
material is n = 1 - delta + i * beta.

This is the fixed-q version: every curve is measured at ENERGY_EV on the same Q_GRID.

Plots live in plot_reflectivity.py and the Hugging Face upload in hf_dataset.py.
Data is published at https://huggingface.co/datasets/carbon-lab/reflectivity-1m (branch fixed-q)
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError
from numpy.typing import NDArray
from refloxide import tmm
from refloxide.model import MaterialSLD

import hf_dataset
import plot_reflectivity

# Type aliases used throughout: a numpy array of float64 values, and a dictionary
# that maps a parameter name (e.g. "film_thickness") to one value per sample.
FloatArray = NDArray[np.float64]
Parameters = dict[str, FloatArray]

# =====================================================================================
# DEFAULT SETTINGS: edit these, or override the run options from the command line.
# =====================================================================================

# --- Measurement: where the reflectivity is "measured" ---
# Every sample uses the same photon energy and the same evenly spaced q grid.
# Q_RANGE spans q(0.5 deg, 500 eV) to q(65 deg, 200 eV): the range the earlier
# random-energy dataset could reach at every energy. At 250 eV it is theta ~0.2-46 deg.
ENERGY_EV = 250.0                   # photon energy [eV]
Q_RANGE = (0.0044, 0.184)           # first and last q of the grid [1/Å]
N_Q = 256                           # q points per curve
Q_GRID = np.linspace(Q_RANGE[0], Q_RANGE[1], N_Q)

# --- Film parameter ranges (every value is drawn uniformly inside its range) ---
FILM_THICKNESS_RANGE = (20.0, 800.0)    # [Å]
FILM_ROUGHNESS_RANGE = (0.0, 40.0)      # [Å], also capped at film_thickness / (2 pi)
DELTA_MAGNITUDE_RANGE = (-5.0e-3, 5.0e-3)
BETA_MAGNITUDE_RANGE = (1.0e-6, 5.0e-3)
TENSOR_ANGLE_RANGE_DEG = (0.0, 90.0)    # 0 = all in-plane (xx), 90 = all out-of-plane (zz)

# --- Substrate: native oxide on silicon ---
OXIDE_THICKNESS_RANGE = (0.0, 15.0)     # [Å]
OXIDE_FORMULA = "SiO2"
OXIDE_DENSITY = 2.2                     # [g/cm^3]
SILICON_FORMULA = "Si"
SILICON_DENSITY = 2.33                  # [g/cm^3]
SILICON_ROUGHNESS = 1.2                 # [Å], oxide | silicon interface

# Rough interfaces are only modelled correctly when thickness > 2 pi * roughness
# (the Nevot-Croce limit). The film roughness is capped at that limit, and the oxide
# roughness is tied to it: oxide_roughness = oxide_thickness / OXIDE_THICKNESS_TO_ROUGHNESS.
OXIDE_THICKNESS_TO_ROUGHNESS = 2.0 * math.pi

# The parameters saved as training labels, in column order.
LABEL_NAMES = (
    "film_thickness",
    "film_roughness",
    "oxide_thickness",
    "delta_xx",
    "delta_zz",
    "beta_xx",
    "beta_zz",
)

# --- Run options ---
SEED = None                     # random seed (None = different samples every run)
DEVICE = "auto"                 # "auto" uses the GPU when available, otherwise "cpu"
DRY_RUN_SAMPLES = 1_000_000     # samples streamed by --dry-run when -n is not given
CHUNK_SIZE = 25_000             # samples simulated at once (bigger = faster, more memory)

# --- Output ---
PREVIEW_PLOT = Path("plots") / "fixed-q" / "single_sample.png"
PLOT_DIR = Path("plots") / "fixed-q"
DATASET_DIR = Path("data") / "reflectivity-1m-fixed-q"
SHARD_SIZE = 100_000            # rows per Parquet file
HF_REPO_ID = "carbon-lab/reflectivity-1m"
HF_REVISION = "fixed-q"         # branch of the dataset repository to upload to
UPLOAD = True                   # upload to Hugging Face after writing the dataset

# Photon energy E [eV] times wavelength [Å] is a constant: E * lambda = h * c.
HC_EV_ANGSTROM = 12398.41984


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.upload_only:
        return upload(args)
    if args.dry_run:
        n_samples = DRY_RUN_SAMPLES if args.num_samples is None else args.num_samples
        run_dry_run(n_samples, rng, resolve_device(args.device), args.chunk_size, args.plot_dir)
        return 0
    if args.num_samples is not None:
        write_dataset(
            args.num_samples, rng, resolve_device(args.device),
            args.chunk_size, args.shard_size, args.dataset_dir, args.plot_dir,
        )
        if not args.upload:
            return 0
        return upload(args)
    run_preview(rng, args.output)
    return 0


# =====================================================================================
# The three run modes
# =====================================================================================


def run_preview(rng: np.random.Generator, output: Path) -> None:
    """Simulate one random sample, print its parameters and plot it."""
    oxide = MaterialSLD(OXIDE_FORMULA, OXIDE_DENSITY, name="oxide")
    silicon = MaterialSLD(SILICON_FORMULA, SILICON_DENSITY, name="silicon")

    energy_ev, q = sample_measurement(1, rng)
    params = sample_parameters(1, rng)
    layers, tensors = build_stack(energy_ev, params, oxide, silicon)
    r_s, r_p = simulate_reflectivity(q, energy_ev, layers, tensors, device="cpu")

    print("random sample:")
    print(f"  photon energy    {energy_ev[0]:8.2f} eV  (not a label)")
    print(f"  q range          [{q[0, 0]:.4f}, {q[0, -1]:.4f}] 1/Å over {N_Q} points")
    print(f"  film thickness   {params['film_thickness'][0]:8.1f} Å")
    print(f"  film roughness   {params['film_roughness'][0]:8.1f} Å"
          f"  (limit {params['film_thickness'][0] / (2.0 * math.pi):.1f} Å)")
    print(f"  oxide thickness  {params['oxide_thickness'][0]:8.1f} Å")
    print(f"  oxide roughness  {params['oxide_roughness'][0]:8.1f} Å")
    print(f"  delta  xx {params['delta_xx'][0]:+.3e}   zz {params['delta_zz'][0]:+.3e}")
    print(f"  beta   xx {params['beta_xx'][0]:+.3e}   zz {params['beta_zz'][0]:+.3e}")

    plot_reflectivity.plot_sample(
        energy_ev[0], q[0], r_s[0], r_p[0], layers[0], tensors[0], params, output
    )
    print(f"saved {output}")


def run_dry_run(
    n_samples: int,
    rng: np.random.Generator,
    device: str,
    chunk_size: int,
    plot_dir: Path,
) -> None:
    """Simulate n_samples curves without saving them, then plot their distributions."""
    oxide = MaterialSLD(OXIDE_FORMULA, OXIDE_DENSITY, name="oxide")
    silicon = MaterialSLD(SILICON_FORMULA, SILICON_DENSITY, name="silicon")

    # The parameters are small (a few numbers per sample), so we keep all of them.
    # The curves are too big to keep, so we only add them to running histograms.
    all_params: dict[str, list[FloatArray]] = {}
    all_energies: list[FloatArray] = []
    histogram_s = plot_reflectivity.RQ4Histogram(q_max=Q_RANGE[1])
    histogram_p = plot_reflectivity.RQ4Histogram(q_max=Q_RANGE[1])

    start_time = time.perf_counter()
    for start in range(0, n_samples, chunk_size):
        n_chunk = min(chunk_size, n_samples - start)
        energy_ev, q = sample_measurement(n_chunk, rng)
        params = sample_parameters(n_chunk, rng)
        layers, tensors = build_stack(energy_ev, params, oxide, silicon)
        r_s, r_p = simulate_reflectivity(q, energy_ev, layers, tensors, device)

        all_energies.append(energy_ev)
        for name in params:
            if name not in all_params:
                all_params[name] = []
            all_params[name].append(params[name])
        histogram_s.add(q, r_s)
        histogram_p.add(q, r_p)

        done = start + n_chunk
        print(f"dry run: {done:,}/{n_samples:,} samples in {time.perf_counter() - start_time:.1f}s", flush=True)

    params = {}
    for name in all_params:
        params[name] = np.concatenate(all_params[name])

    plot_dir.mkdir(parents=True, exist_ok=True)
    parameter_plot = plot_dir / "dry_run_parameters.png"
    plot_reflectivity.plot_parameter_distributions(
        params, np.concatenate(all_energies),
        FILM_THICKNESS_RANGE, FILM_ROUGHNESS_RANGE[1], parameter_plot,
    )
    print(f"saved {parameter_plot}")

    rq4_plot = plot_dir / "dry_run_reflectivity_q4.png"
    plot_reflectivity.plot_rq4_distributions(histogram_s, histogram_p, n_samples, N_Q, rq4_plot)
    print(f"saved {rq4_plot}")


def write_dataset(
    n_samples: int,
    rng: np.random.Generator,
    device: str,
    chunk_size: int,
    shard_size: int,
    dataset_dir: Path,
    plot_dir: Path,
) -> None:
    """Simulate n_samples curves and write them to Parquet files plus a dataset card.

    Storage choices, all for speed and size:
      - float32 keeps reflectivity's ~12 decades of range at half the size of float64
        (float16 would round everything below ~6e-8 to zero).
      - Each curve is a fixed-size list of N_Q floats, so it loads straight into an
        (n_samples, N_Q) array without per-row bookkeeping.
      - Zstd compression with byte-stream-split encoding compresses floats well.
      - Rows are split into shards of shard_size rows so files stay a manageable size.
    """
    if n_samples <= 0 or shard_size <= 0:
        raise ValueError("n_samples and shard_size must be positive")

    oxide = MaterialSLD(OXIDE_FORMULA, OXIDE_DENSITY, name="oxide")
    silicon = MaterialSLD(SILICON_FORMULA, SILICON_DENSITY, name="silicon")

    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in data_dir.glob("train-*.parquet"):
        old_shard.unlink()  # never mix shards from an earlier run

    # Column layout: three curves (q, s, p) followed by one float per label.
    curve_type = pa.list_(pa.field("item", pa.float32(), nullable=False), N_Q)
    fields = [
        pa.field("q", curve_type, nullable=False),
        pa.field("s", curve_type, nullable=False),
        pa.field("p", curve_type, nullable=False),
    ]
    for name in LABEL_NAMES:
        fields.append(pa.field(name, pa.float32(), nullable=False))
    schema = pa.schema(fields)

    n_shards = math.ceil(n_samples / shard_size)
    start_time = time.perf_counter()
    total_bytes = 0
    for shard_index in range(n_shards):
        shard_start = shard_index * shard_size
        shard_end = min(shard_start + shard_size, n_samples)
        shard_path = data_dir / f"train-{shard_index:05d}-of-{n_shards:05d}.parquet"

        writer = pq.ParquetWriter(
            shard_path, schema,
            compression="zstd", compression_level=3,
            use_byte_stream_split=True, use_dictionary=False,
        )
        for start in range(shard_start, shard_end, chunk_size):
            n_chunk = min(chunk_size, shard_end - start)
            energy_ev, q = sample_measurement(n_chunk, rng)
            params = sample_parameters(n_chunk, rng)
            layers, tensors = build_stack(energy_ev, params, oxide, silicon)
            r_s, r_p = simulate_reflectivity(q, energy_ev, layers, tensors, device)

            columns = []
            for curve in (q, r_s, r_p):
                flat = pa.array(curve.astype(np.float32).ravel(), type=pa.float32())
                columns.append(pa.FixedSizeListArray.from_arrays(flat, type=curve_type))
            for name in LABEL_NAMES:
                columns.append(pa.array(params[name].astype(np.float32), type=pa.float32()))
            writer.write_batch(pa.RecordBatch.from_arrays(columns, schema=schema))
        writer.close()

        shard_bytes = shard_path.stat().st_size
        total_bytes += shard_bytes
        print(
            f"[{shard_index + 1}/{n_shards}] wrote {shard_path.name} "
            f"({shard_end:,}/{n_samples:,} rows, {shard_bytes / 1e6:.1f} MB, "
            f"{time.perf_counter() - start_time:.1f}s)",
            flush=True,
        )

    hf_dataset.write_dataset_card(
        dataset_dir, plot_dir,
        n_samples=n_samples, n_shards=n_shards, n_q=N_Q,
        repo_id=HF_REPO_ID, revision=HF_REVISION, q_range=Q_RANGE, energy_ev=ENERGY_EV,
        film_thickness_range=FILM_THICKNESS_RANGE, film_roughness_max=FILM_ROUGHNESS_RANGE[1],
        oxide_thickness_range=OXIDE_THICKNESS_RANGE,
        delta_range=DELTA_MAGNITUDE_RANGE, beta_max=BETA_MAGNITUDE_RANGE[1],
        silicon_density=SILICON_DENSITY, silicon_roughness=SILICON_ROUGHNESS,
        oxide_density=OXIDE_DENSITY,
    )
    print(f"finished {dataset_dir}: {n_samples:,} rows in {n_shards} shard(s), {total_bytes / 1e9:.2f} GB")


def upload(args: argparse.Namespace) -> int:
    """Upload the dataset folder to Hugging Face. Returns the program's exit code."""
    print(f"uploading {args.dataset_dir} to https://huggingface.co/datasets/{args.repo_id} (branch {args.revision}) ...", flush=True)
    try:
        url = hf_dataset.upload_dataset(
            args.dataset_dir, args.repo_id, args.revision, args.hf_token, args.private
        )
    except (LocalTokenNotFoundError, HfHubHTTPError, FileNotFoundError) as error:
        print(
            f"Hugging Face upload failed ({error}).\n"
            f"The local Parquet files are kept in '{args.dataset_dir}'.\n"
            f"Log in with `uv run hf auth login` (or set HF_TOKEN), then retry with:\n"
            f"  uv run python scripts/generate_reflectivity.py --upload-only "
            f"--dataset-dir {args.dataset_dir} --repo-id {args.repo_id} --revision {args.revision}"
        )
        return 1
    print(f"uploaded dataset: {url}")
    return 0


# =====================================================================================
# Sampling and simulation
# =====================================================================================


def sample_measurement(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Return where each curve is measured: one photon energy and N_Q q values per sample.

    Returns ``(energy_ev, q)`` with shapes (n,) and (n, N_Q). Every sample shares the
    fixed ENERGY_EV and Q_GRID, so nothing random is drawn here (``rng`` is kept so the
    function can go back to random measurements without changing its callers).
    """
    energy_ev = np.full(n, ENERGY_EV)
    q = np.tile(Q_GRID, (n, 1))  # the same row of q values, repeated n times
    return energy_ev, q


def sample_parameters(n: int, rng: np.random.Generator) -> Parameters:
    """Draw the film and oxide parameters for n samples.

    Thicknesses are drawn first, because each roughness is limited by its layer's
    thickness. Draws happen in a fixed order so a given seed always gives the same data.
    """
    film_thickness = rng.uniform(FILM_THICKNESS_RANGE[0], FILM_THICKNESS_RANGE[1], size=n)
    oxide_thickness = rng.uniform(OXIDE_THICKNESS_RANGE[0], OXIDE_THICKNESS_RANGE[1], size=n)

    # Film roughness: uniform between the minimum and min(maximum, thickness / 2 pi).
    roughness_min, roughness_max = FILM_ROUGHNESS_RANGE
    roughness_limit = np.clip(film_thickness / (2.0 * math.pi), roughness_min, roughness_max)
    film_roughness = rng.uniform(roughness_min, roughness_limit)
    oxide_roughness = oxide_thickness / OXIDE_THICKNESS_TO_ROUGHNESS

    # Optical constants in polar form: magnitude * (cos(angle), sin(angle)) = (xx, zz).
    delta_magnitude = rng.uniform(DELTA_MAGNITUDE_RANGE[0], DELTA_MAGNITUDE_RANGE[1], size=n)
    delta_angle_deg = rng.uniform(TENSOR_ANGLE_RANGE_DEG[0], TENSOR_ANGLE_RANGE_DEG[1], size=n)
    beta_magnitude = rng.uniform(BETA_MAGNITUDE_RANGE[0], BETA_MAGNITUDE_RANGE[1], size=n)
    beta_angle_deg = rng.uniform(TENSOR_ANGLE_RANGE_DEG[0], TENSOR_ANGLE_RANGE_DEG[1], size=n)
    delta_angle = np.deg2rad(delta_angle_deg)
    beta_angle = np.deg2rad(beta_angle_deg)

    return {
        "film_thickness": film_thickness,
        "film_roughness": film_roughness,
        "oxide_thickness": oxide_thickness,
        "oxide_roughness": oxide_roughness,
        "delta_xx": delta_magnitude * np.cos(delta_angle),
        "delta_zz": delta_magnitude * np.sin(delta_angle),
        "beta_xx": beta_magnitude * np.cos(beta_angle),
        "beta_zz": beta_magnitude * np.sin(beta_angle),
        # Polar form, kept only for the dry-run plots.
        "delta_magnitude": delta_magnitude,
        "delta_angle_deg": delta_angle_deg,
        "beta_magnitude": beta_magnitude,
        "beta_angle_deg": beta_angle_deg,
    }


def build_stack(
    energy_ev: FloatArray,
    params: Parameters,
    oxide: MaterialSLD,
    silicon: MaterialSLD,
) -> tuple[FloatArray, NDArray[np.complex128]]:
    """Fill the two arrays that describe every sample's layer stack.

    Returns ``(layers, tensors)``, one entry per sample and per layer
    (0 vacuum, 1 film, 2 oxide, 3 silicon):
      - layers  (n, 4, 4):    [thickness, delta, beta, roughness of the layer's top interface]
      - tensors (n, 4, 3, 3): optical tensor delta + i * beta (diagonal: xx, yy, zz)
    The vacuum layer stays all zeros.
    """
    n = energy_ev.shape[0]
    layers = np.zeros((n, 4, 4), dtype=np.float64)
    tensors = np.zeros((n, 4, 3, 3), dtype=np.complex128)

    # Layer 1: the uniaxial film. xx and yy (in-plane) are equal; zz is out-of-plane.
    n_xx = params["delta_xx"] + 1j * params["beta_xx"]
    n_zz = params["delta_zz"] + 1j * params["beta_zz"]
    n_average = (2.0 * n_xx + n_zz) / 3.0
    layers[:, 1, 0] = params["film_thickness"]
    layers[:, 1, 1] = n_average.real
    layers[:, 1, 2] = n_average.imag
    layers[:, 1, 3] = params["film_roughness"]
    tensors[:, 1, 0, 0] = n_xx
    tensors[:, 1, 1, 1] = n_xx
    tensors[:, 1, 2, 2] = n_zz

    # Layer 2: SiO2 oxide (isotropic, optical constants depend on photon energy).
    oxide_tensors = oxide.tensor_at_many(energy_ev)
    layers[:, 2, 0] = params["oxide_thickness"]
    layers[:, 2, 1] = oxide_tensors[:, 0, 0].real
    layers[:, 2, 2] = oxide_tensors[:, 0, 0].imag
    layers[:, 2, 3] = params["oxide_roughness"]
    tensors[:, 2] = oxide_tensors

    # Layer 3: silicon substrate (infinitely thick, so thickness stays 0).
    silicon_tensors = silicon.tensor_at_many(energy_ev)
    layers[:, 3, 1] = silicon_tensors[:, 0, 0].real
    layers[:, 3, 2] = silicon_tensors[:, 0, 0].imag
    layers[:, 3, 3] = SILICON_ROUGHNESS
    tensors[:, 3] = silicon_tensors

    return layers, tensors


def simulate_reflectivity(
    q: FloatArray,
    energy_ev: FloatArray,
    layers: FloatArray,
    tensors: NDArray[np.complex128],
    device: str,
) -> tuple[FloatArray, FloatArray]:
    """Compute ``(R_s, R_p)``, each (n, n_q), with refloxide's transfer-matrix solver.

    The solver takes every q point of every sample in one flat list, plus
    ``stack_index`` saying which sample (row of ``layers``) each q point belongs to.
    """
    n, n_q = q.shape
    q_flat = np.ascontiguousarray(q, dtype=np.float64).ravel()
    stack_index = np.repeat(np.arange(n, dtype=np.int64), n_q)
    reflectivity, _ = tmm.uniaxial_reflectivity_points(
        q_flat, stack_index, layers, tensors, energy_ev, parallel=True, device=device,
    )
    # One 2x2 polarization matrix per point: [0, 0] is p -> p and [1, 1] is s -> s.
    reflectivity = reflectivity.reshape(n, n_q, 2, 2)
    return reflectivity[:, :, 1, 1], reflectivity[:, :, 0, 0]


def resolve_device(device: str) -> str:
    """Return "cpu" or "gpu". With "auto", try a tiny GPU calculation and fall back to CPU."""
    if device != "auto":
        return device
    try:
        tmm.uniaxial_reflectivity_points(
            np.array([0.05]),
            np.array([0], dtype=np.int64),
            np.zeros((1, 2, 4)),
            np.zeros((1, 2, 3, 3), dtype=np.complex128),
            np.array([284.4]),
            parallel=False,
            device="gpu",
        )
        return "gpu"
    except RuntimeError:
        return "cpu"


def parse_args() -> argparse.Namespace:
    """Read command-line flags. Every flag defaults to the matching setting at the top of this file."""
    parser = argparse.ArgumentParser(
        description="Generate polarized reflectivity curves (preview, dry run, or full dataset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # show defaults in --help
    )
    on_off = argparse.BooleanOptionalAction  # gives both --flag and --no-flag

    mode = parser.add_argument_group("mode (default: preview one random sample)")
    mode.add_argument("-n", "--num-samples", type=int, default=None, help="write a dataset of N samples (with --dry-run: plot N samples)")
    mode.add_argument("--dry-run", action="store_true", help=f"only plot distributions, save no data (default N = {DRY_RUN_SAMPLES:,})")
    mode.add_argument("--upload-only", action="store_true", help="upload an existing --dataset-dir without generating")

    run = parser.add_argument_group("run")
    run.add_argument("--seed", type=int, default=SEED, help="random seed")
    run.add_argument("--device", choices=("auto", "cpu", "gpu"), default=DEVICE, help="where to run the simulation")
    run.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="samples simulated at once")

    output = parser.add_argument_group("output")
    output.add_argument("--output", type=Path, default=PREVIEW_PLOT, help="preview plot path")
    output.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="folder for dry-run plots")
    output.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="folder for Parquet files and dataset card")
    output.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="rows per Parquet file")

    hub = parser.add_argument_group("hugging face")
    hub.add_argument("--upload", action=on_off, default=UPLOAD, help="upload after writing the dataset")
    hub.add_argument("--repo-id", default=HF_REPO_ID, help="dataset repository")
    hub.add_argument("--revision", default=HF_REVISION, help="branch of the repository to upload to")
    hub.add_argument("--hf-token", default=None, help="token (default: cached login or HF_TOKEN)")
    hub.add_argument("--private", action="store_true", help="create the repository as private")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
