"""One plotting style for every Refltron figure (training snapshots, Hub figures).

Colours follow a validated categorical palette: the first three slots pass the
colour-vision-deficiency checks for every pair, slots 1-4 for adjacent pairs (lines).
A colour always follows the same thing: version v1/v2/v3 keep their colour in every
figure, and so do the s/p polarizations and the four optical components.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# --- Palette -------------------------------------------------------------------------
SURFACE = "#fcfcfb"     # chart background
INK = "#0b0b0b"         # titles, values
INK_2 = "#52514e"       # secondary text, axis labels
MUTED = "#898781"       # tick labels, annotations, "truth" marks
GRID = "#e1e0d9"        # hairline gridlines
AXIS = "#c3c2b7"        # axis lines / baselines
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # blue, orange, aqua, yellow
TRUTH = "#a8a69e"       # neutral gray for ground truth next to a coloured prediction

# Things that keep one colour everywhere.
VERSION_COLORS = {"v1-relu-rq4": SLOTS[0], "v2-relu-log": SLOTS[1], "v3-gelu-rq4": SLOTS[2]}
VERSION_LABELS = {"v1-relu-rq4": "v1 · ReLU · R·q⁴", "v2-relu-log": "v2 · ReLU · log R", "v3-gelu-rq4": "v3 · GELU · R·q⁴"}
POLARIZATION_COLORS = {"s": SLOTS[0], "p": SLOTS[1]}
COMPONENT_COLORS = {  # optical components of the depth profile
    "delta_xx": (SLOTS[0], r"$\delta_{xx}$"),
    "delta_zz": (SLOTS[1], r"$\delta_{zz}$"),
    "beta_xx": (SLOTS[2], r"$\beta_{xx}$"),
    "beta_zz": (SLOTS[3], r"$\beta_{zz}$"),
}
# Depth-profile layers, light (vacuum) to dark (silicon): one neutral ramp.
LAYER_SHADES = {"vacuum": "#fcfcfb", "film": "#f3f2ee", "oxide": "#e6e5df", "Si": "#d6d5ce"}

FIG_DPI = 220


def apply():
    """Set matplotlib defaults: system sans, recessive hairline chrome, readable sizes."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "text.color": INK,
    })


def no_minor_grid(ax):
    """Log axes add a minor gridline per decade step; keep only the major ones."""
    ax.grid(False, which="minor")
    ax.grid(True, which="major")
