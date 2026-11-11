"""Plots for generate_reflectivity.py.

    plot_sample                   one curve pair (s and p) and its depth profile
    RQ4Histogram                  running histogram of R * q^4 vs q over many curves
    plot_parameter_distributions  histograms of the sampled parameters (dry run)
    plot_rq4_distributions        q and R * q^4 distributions (dry run)

Nothing here simulates anything: every function takes plain numpy arrays.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LAYER_NAMES = ("vacuum", "film", "oxide", "silicon")


# =====================================================================================
# Single sample
# =====================================================================================


def depth_profile(
    layers: FloatArray,
    tensors: NDArray[np.complex128],
    n_points: int = 2000,
    padding: float = 15.0,
) -> tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """Optical constants as a function of depth for one stack, with smooth interfaces.

    ``layers`` (n_layers, 4) and ``tensors`` (n_layers, 3, 3) are one sample's rows from
    ``build_stack``. A rough interface is drawn as an error-function step whose width
    is the interface roughness.

    Returns ``(z, boundaries, curves)``: depth [Å] (0 at the film surface), the depth
    of each interface, and a dictionary with a curve for each of
    delta_xx, delta_zz, beta_xx and beta_zz.
    """
    thicknesses = layers[1:, 0]
    roughnesses = layers[1:, 3]  # roughness of the interface on top of each layer
    boundaries = np.concatenate([[0.0], np.cumsum(thicknesses)])[:-1]
    z = np.linspace(-padding, boundaries[-1] + padding, n_points)

    layer_values = {
        "delta_xx": tensors[:, 0, 0].real,
        "delta_zz": tensors[:, 2, 2].real,
        "beta_xx": tensors[:, 0, 0].imag,
        "beta_zz": tensors[:, 2, 2].imag,
    }
    curves = {}
    for name in layer_values:
        values = layer_values[name]
        curve = np.full_like(z, values[0])
        for i in range(len(boundaries)):
            if roughnesses[i] > 0.0:
                scaled = (z - boundaries[i]) / (math.sqrt(2.0) * roughnesses[i])
                step = np.zeros_like(z)
                for j in range(len(z)):
                    step[j] = 0.5 * (1.0 + math.erf(scaled[j]))
            else:
                step = (z >= boundaries[i]).astype(np.float64)
            curve = curve + (values[i + 1] - values[i]) * step
        curves[name] = curve
    return z, boundaries, curves


def plot_sample(
    energy_ev: float,
    q: FloatArray,
    r_s: FloatArray,
    r_p: FloatArray,
    layers: FloatArray,
    tensors: NDArray[np.complex128],
    params: dict[str, FloatArray],
    output: Path,
) -> None:
    """Plot the s/p reflectivity of one sample (top) and its depth profile (bottom).

    ``params`` is the dictionary from ``sample_parameters`` for a batch of one sample.
    """
    figure, (ax_curve, ax_depth) = plt.subplots(2, 1, figsize=(9.0, 9.0), layout="constrained")

    # --- Reflectivity ---
    marker_style = {"linestyle": "", "marker": "o", "markersize": 4, "markerfacecolor": "none"}
    ax_curve.plot(q, r_s, color="tab:blue", label="s polarization", **marker_style)
    ax_curve.plot(q, r_p, color="tab:red", label="p polarization", **marker_style)
    ax_curve.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
    ax_curve.set_ylabel("reflectivity")
    ax_curve.set_yscale("log")
    ax_curve.set_title(f"s and p reflectivity at {energy_ev:.1f} eV")
    ax_curve.grid(which="both", alpha=0.3)
    ax_curve.legend(loc="upper right")

    # --- Depth profile ---
    z, boundaries, curves = depth_profile(layers, tensors)
    roughnesses = layers[1:, 3]
    ax_depth.axhline(0.0, color="0.6", linewidth=0.6, linestyle=":")
    ax_depth.plot(z, curves["delta_xx"], color="tab:blue", linestyle="-", label=r"$\delta_{xx}$")
    ax_depth.plot(z, curves["beta_xx"], color="tab:blue", linestyle="--", label=r"$\beta_{xx}$")
    ax_depth.plot(z, curves["delta_zz"], color="tab:red", linestyle="-", label=r"$\delta_{zz}$")
    ax_depth.plot(z, curves["beta_zz"], color="tab:red", linestyle="--", label=r"$\beta_{zz}$")

    # Shade each layer and label it; mark each interface with its roughness.
    edges = np.concatenate([[z[0]], boundaries, [z[-1]]])
    shades = ("0.97", "0.92", "0.84", "0.76")
    x_in_data_y_in_axes = ax_depth.get_xaxis_transform()
    for i in range(len(LAYER_NAMES)):
        ax_depth.axvspan(edges[i], edges[i + 1], color=shades[i], zorder=0)
        label_height = 0.90 if LAYER_NAMES[i] == "oxide" else 0.96  # oxide is thin; avoid overlap
        ax_depth.text(
            0.5 * (edges[i] + edges[i + 1]), label_height, LAYER_NAMES[i],
            ha="center", va="top", transform=x_in_data_y_in_axes,
        )
    for i in range(len(boundaries)):
        ax_depth.axvline(boundaries[i], color="0.5", linewidth=0.8)
        on_left = i == 1  # put the film | oxide label left of its line so it doesn't collide
        ax_depth.annotate(
            f"σ {roughnesses[i]:.1f} Å", (boundaries[i], 0.02),
            xycoords=x_in_data_y_in_axes,
            xytext=(-3 if on_left else 3, 2), textcoords="offset points",
            ha="right" if on_left else "left", fontsize=8, color="0.35",
        )
    ax_depth.set_xlabel(r"depth [$\mathrm{\AA}$]")
    ax_depth.set_ylabel(r"$\delta$, $\beta$")
    ax_depth.set_title("optical constant depth profile (tensor components)")
    ax_depth.margins(y=0.12)
    ax_depth.grid(which="both", alpha=0.3)
    ax_depth.legend(loc="lower right")

    figure.suptitle(
        f"E {energy_ev:.1f} eV  |  "
        f"film {params['film_thickness'][0]:.0f} Å (σ {params['film_roughness'][0]:.1f} Å)  |  "
        f"δ ({params['delta_xx'][0]:+.2e}, {params['delta_zz'][0]:+.2e})  |  "
        f"β ({params['beta_xx'][0]:.2e}, {params['beta_zz'][0]:.2e})  |  "
        f"oxide {params['oxide_thickness'][0]:.1f} Å (σ {params['oxide_roughness'][0]:.1f} Å)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


# =====================================================================================
# Dry run: distributions over many samples
# =====================================================================================


class RQ4Histogram:
    """Counts how many (q, log10(R * q^4)) points fall in each cell of a fixed 2D grid.

    Call ``add`` once per batch of curves. Only the counts are kept, so memory does
    not grow with the number of curves. Multiplying by q^4 removes the steep q^-4
    fall-off of reflectivity, so curves of every q are on a similar scale.
    """

    def __init__(self, q_max: float, n_q_bins: int = 220, n_y_bins: int = 220):
        self.q_min = 0.0
        self.q_max = q_max * 1.02
        self.y_min = -15.5  # log10(R * q^4) range; values outside land in the edge bins
        self.y_max = -4.5
        self.q_edges = np.linspace(self.q_min, self.q_max, n_q_bins + 1)
        self.y_edges = np.linspace(self.y_min, self.y_max, n_y_bins + 1)
        self.counts = np.zeros((n_q_bins, n_y_bins), dtype=np.int64)

    def add(self, q: FloatArray, reflectivity: FloatArray) -> None:
        """Add every point of a batch of curves (both arrays shaped (n, n_q))."""
        q_flat = q.ravel()
        rq4 = reflectivity.ravel() * q_flat**4
        y = np.log10(np.maximum(rq4, 10.0**self.y_min))

        # Find each point's bin. This is what np.histogram2d does, but much faster
        # for evenly spaced bins: scale to [0, n_bins) and round down.
        n_q_bins, n_y_bins = self.counts.shape
        q_bin = ((q_flat - self.q_min) * (n_q_bins / (self.q_max - self.q_min))).astype(np.int64)
        y_bin = ((y - self.y_min) * (n_y_bins / (self.y_max - self.y_min))).astype(np.int64)
        q_bin = np.clip(q_bin, 0, n_q_bins - 1)
        y_bin = np.clip(y_bin, 0, n_y_bins - 1)
        # Number each cell 0 .. n_q_bins * n_y_bins - 1 and count with bincount.
        cell_counts = np.bincount(q_bin * n_y_bins + y_bin, minlength=n_q_bins * n_y_bins)
        self.counts += cell_counts.reshape(n_q_bins, n_y_bins)

    def quantiles(self, fractions: tuple[float, ...]) -> FloatArray:
        """Approximate quantiles of log10(R * q^4) in each q bin, shape (len(fractions), n_q_bins)."""
        y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        result = np.full((len(fractions), self.counts.shape[0]), np.nan)
        for i in range(self.counts.shape[0]):
            total = self.counts[i].sum()
            if total == 0:
                continue
            cumulative = np.cumsum(self.counts[i]) / total
            result[:, i] = np.interp(fractions, cumulative, y_centers)
        return result


def plot_parameter_distributions(
    params: dict[str, FloatArray],
    energy_ev: FloatArray,
    film_thickness_range: tuple[float, float],
    film_roughness_max: float,
    output: Path,
) -> None:
    """3x3 grid of histograms of the sampled parameters."""
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), layout="constrained")
    bins = 80
    n_samples = len(energy_ev)

    ax = axes[0, 0]
    ax.hist(params["film_thickness"], bins=bins, density=True, color="tab:blue", alpha=0.75)
    ax.set_xlabel(r"film thickness $t_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_title("film thickness")

    ax = axes[0, 1]
    ax.hist(params["film_roughness"], bins=bins, density=True, color="tab:cyan", alpha=0.75)
    ax.set_xlabel(r"film roughness $\sigma_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_title("film roughness (Nevot–Croce capped)")

    ax = axes[0, 2]
    counts, x_edges, y_edges = np.histogram2d(params["film_thickness"], params["film_roughness"], bins=80)
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_less_equal(counts.T, 0), cmap="viridis")
    thickness = np.linspace(film_thickness_range[0], film_thickness_range[1], 200)
    roughness_limit = np.minimum(thickness / (2.0 * math.pi), film_roughness_max)
    ax.plot(thickness, roughness_limit, color="tab:red", linestyle="--", linewidth=1.5,
            label=r"$\min(t / 2\pi,\; \sigma_{\max})$")
    fig.colorbar(mesh, ax=ax, label="count")
    ax.set_xlabel(r"$t_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_ylabel(r"$\sigma_{\mathrm{film}}$ [$\mathrm{\AA}$]")
    ax.set_title("thickness–roughness coupling")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    ax.hist(params["oxide_thickness"], bins=bins, density=True, color="tab:olive", alpha=0.75)
    ax.set_xlabel(r"oxide thickness $t_{\mathrm{oxide}}$ [$\mathrm{\AA}$]")
    ax.set_title(r"oxide thickness ($\sigma_{\mathrm{ox}} = t_{\mathrm{ox}} / 2\pi$)")

    ax = axes[1, 1]
    ax.hist(params["delta_magnitude"] * 1e3, bins=bins, density=True, color="tab:purple", alpha=0.75)
    ax.set_xlabel(r"$\delta$ magnitude [$\times 10^{-3}$]")
    ax.set_title(r"$\delta$ signed magnitude")

    ax = axes[1, 2]
    ax.hist(params["beta_magnitude"] * 1e3, bins=bins, density=True, color="tab:orange", alpha=0.75)
    ax.set_xlabel(r"$\beta$ magnitude [$\times 10^{-3}$]")
    ax.set_title(r"$\beta$ magnitude")

    ax = axes[2, 0]
    ax.hist(params["delta_angle_deg"], bins=bins, density=True, histtype="step", linewidth=1.6,
            color="tab:purple", label=r"$\delta$ angle")
    ax.hist(params["beta_angle_deg"], bins=bins, density=True, histtype="step", linewidth=1.6,
            color="tab:orange", linestyle="--", label=r"$\beta$ angle")
    ax.set_xlabel("tensor angle [deg]")
    ax.set_title(r"$\delta$ and $\beta$ orientation angles")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2, 1]
    step_style = {"bins": bins, "density": True, "histtype": "step", "linewidth": 1.5}
    ax.hist(params["delta_xx"] * 1e3, color="tab:blue", label=r"$\delta_{xx}$", **step_style)
    ax.hist(params["delta_zz"] * 1e3, color="tab:red", label=r"$\delta_{zz}$", **step_style)
    ax.hist(params["beta_xx"] * 1e3, color="tab:blue", linestyle="--", label=r"$\beta_{xx}$", **step_style)
    ax.hist(params["beta_zz"] * 1e3, color="tab:red", linestyle="--", label=r"$\beta_{zz}$", **step_style)
    ax.set_xlabel(r"tensor component [$\times 10^{-3}$]")
    ax.set_title("stored uniaxial tensor components")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax = axes[2, 2]
    ax.hist(energy_ev, bins=bins, density=True, color="0.45", alpha=0.8)
    ax.set_xlabel("photon energy [eV]")
    ax.set_title("photon energy (not a label)")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        if ax is not axes[0, 2]:
            ax.set_ylabel("density")
    fig.suptitle(f"Dry-Run Parameter Distributions ({n_samples:,} samples)", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rq4_distributions(
    histogram_s: RQ4Histogram,
    histogram_p: RQ4Histogram,
    n_samples: int,
    n_q: int,
    output: Path,
) -> None:
    """Plot the q distribution, the R * q^4 distributions, and R * q^4 vs q for s and p."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), layout="constrained")
    total_points = n_samples * n_q

    # Top left: how the q points are spread (every point counts once, s or p alike).
    q_edges = histogram_s.q_edges
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    q_density = histogram_s.counts.sum(axis=1) / (total_points * np.diff(q_edges))
    ax = axes[0, 0]
    ax.fill_between(q_centers, q_density, step="mid", color="tab:gray", alpha=0.35)
    ax.plot(q_centers, q_density, color="0.25", linewidth=1.5)
    ax.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
    ax.set_ylabel("probability density")
    ax.set_title(r"sampled $q$ point distribution")
    ax.grid(alpha=0.3)

    # Top right: distribution of log10(R * q^4) over all points.
    y_edges = histogram_s.y_edges
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    s_density = histogram_s.counts.sum(axis=0) / (total_points * np.diff(y_edges))
    p_density = histogram_p.counts.sum(axis=0) / (total_points * np.diff(y_edges))
    ax = axes[0, 1]
    ax.plot(y_centers, s_density, color="tab:blue", linewidth=1.8, label=r"$R_s \cdot q^4$")
    ax.fill_between(y_centers, s_density, color="tab:blue", alpha=0.2)
    ax.plot(y_centers, p_density, color="tab:red", linewidth=1.8, label=r"$R_p \cdot q^4$")
    ax.fill_between(y_centers, p_density, color="tab:red", alpha=0.2)
    ax.set_xlabel(r"$\log_{10}(R \cdot q^4\ [\mathrm{\AA}^{-4}])$")
    ax.set_ylabel("probability density")
    ax.set_title(r"$R_s \cdot q^4$ and $R_p \cdot q^4$ marginal distributions")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Bottom: 2D density of log10(R * q^4) vs q, with median and 5% / 95% lines.
    largest_count = max(histogram_s.counts.max(), histogram_p.counts.max(), 1)
    norm = LogNorm(vmin=1.0, vmax=float(largest_count))
    panels = (
        (axes[1, 0], histogram_s, r"$R_s \cdot q^4$ vs $q$ joint density", "cyan"),
        (axes[1, 1], histogram_p, r"$R_p \cdot q^4$ vs $q$ joint density", "gold"),
    )
    for ax, histogram, title, line_color in panels:
        mesh = ax.pcolormesh(q_edges, y_edges, np.ma.masked_less_equal(histogram.counts.T, 0),
                             cmap="magma", norm=norm)
        low, median, high = histogram.quantiles((0.05, 0.50, 0.95))
        ax.plot(q_centers, median, color=line_color, linewidth=1.6, label="median")
        ax.plot(q_centers, low, color=line_color, linewidth=1.0, linestyle="--", label="5% / 95%")
        ax.plot(q_centers, high, color=line_color, linewidth=1.0, linestyle="--")
        fig.colorbar(mesh, ax=ax, label="point count (log scale)")
        ax.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
        ax.set_ylabel(r"$\log_{10}(R \cdot q^4\ [\mathrm{\AA}^{-4}])$")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Dry-Run Reflectivity × $q^4$ & $q$ Distributions ({n_samples:,} curves, {total_points:,} points)",
        fontsize=13,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
