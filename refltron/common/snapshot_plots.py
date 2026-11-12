"""Panels shared by every model's snapshot figures: depth profile and label bars."""

import math

import numpy as np
from matplotlib.lines import Line2D

from refltron.common import plot_style as ps
from refltron.common.depth_profile import compute_depth_profile

# Colour and axis label of each optical-constant curve: {name: (colour, label)}.
COMPONENT_COLORS = ps.COMPONENT_COLORS

LAYER_NAMES = ("vacuum", "film", "oxide", "Si")


def plot_depth_profile_panel(ax_depth, true_phys, pred_phys, substrate_sld, profile_l2):
    """Draw truth vs network depth profiles on one axis.

    ``true_phys`` and ``pred_phys`` are the 7 physical labels (LABEL_NAMES order).
    Both stacks are pinned at the silicon interface (h = 0, right edge) so their film
    surfaces can be compared; the truth's layers are shaded.
    """
    # Height range: from just below the silicon interface to just above the higher of the
    # two film surfaces. A wildly wrong prediction is capped so the truth stays readable.
    true_top = max(float(true_phys[0]) + float(true_phys[2]), 15.0)
    pred_cap = max(2500.0, true_top * 1.35)
    pred_top = min(max(float(pred_phys[0]) + max(float(pred_phys[2]), 0.0), 0.0), pred_cap)
    max_h = max(true_top, pred_top)
    pad = max(15.0, 0.06 * max_h)
    si_pad = max(10.0, 0.04 * max_h)
    h = np.linspace(-si_pad, max_h + pad, 2000)

    true_heights, _, true_curves = compute_depth_profile(true_phys, h, substrate_sld)
    pred_heights, _, pred_curves = compute_depth_profile(pred_phys, h, substrate_sld)

    # Shade vacuum | film | oxide | silicon using the true interface heights.
    edges = [h[-1], true_heights[0], true_heights[1], true_heights[2], h[0]]
    xaxis_transform = ax_depth.get_xaxis_transform()
    h_span = h[-1] - h[0]
    for i in range(len(LAYER_NAMES)):
        region = LAYER_NAMES[i]
        ax_depth.axvspan(edges[i + 1], edges[i], color=ps.LAYER_SHADES[region], lw=0, zorder=0)
        slab_width = edges[i] - edges[i + 1]
        if slab_width < 0.03 * h_span:
            continue  # too thin to label
        if slab_width < 0.075 * h_span:
            rotation = 90  # thin layer: write its name vertically
        else:
            rotation = 0
        ax_depth.text(
            0.5 * (edges[i] + edges[i + 1]), 0.97, region,
            ha="center", va="top", rotation=rotation,
            fontsize=10.5, color=ps.MUTED, transform=xaxis_transform,
        )
    # Predicted film surface: a dashed vertical line, to compare with the shaded truth.
    ax_depth.axvline(pred_heights[0], color=ps.INK_2, ls=(0, (4, 3)), lw=1.0)

    # Truth as faint wide lines, network as dashed lines of the same colour.
    for key in COMPONENT_COLORS:
        color = COMPONENT_COLORS[key][0]
        ax_depth.plot(h, true_curves[key], color=color, lw=5, alpha=0.28, solid_capstyle="butt")
        ax_depth.plot(h, pred_curves[key], color=color, lw=1.6, ls=(0, (5, 2.2)))

    # Legend: one entry per optical constant, then "truth" and "network".
    legend_handles = []
    for key in COMPONENT_COLORS:
        color = COMPONENT_COLORS[key][0]
        label = COMPONENT_COLORS[key][1]
        legend_handles.append(Line2D([0], [0], color=color, lw=2.0, label=label))
    legend_handles.append(Line2D([0], [0], color=ps.MUTED, lw=5, alpha=0.35, label="truth"))
    legend_handles.append(Line2D([0], [0], color=ps.INK_2, lw=1.6, ls=(0, (5, 2.2)), label="network"))

    # Height decreases left -> right so the silicon substrate is always at the right edge.
    ax_depth.set_xlim(h[-1], h[0])

    # y range: the lowest and highest value of any curve, with room for the legend below.
    lowest = math.inf
    highest = -math.inf
    for key in true_curves:
        lowest = min(lowest, true_curves[key].min(), pred_curves[key].min())
        highest = max(highest, true_curves[key].max(), pred_curves[key].max())
    if highest > lowest:
        span = highest - lowest
    else:
        span = 1e-3
    ax_depth.set_ylim(lowest - 0.30 * span, highest + 0.16 * span)

    ax_depth.set_xlabel(r"height above the Si interface  [$\mathrm{\AA}$]")
    ax_depth.set_ylabel(r"$\delta$, $\beta$")
    ax_depth.set_title(
        f"Depth profile · film top {true_heights[0]:.0f} Å, network {pred_heights[0]:.0f} Å · L2 {profile_l2:.1e}"
    )
    ax_depth.legend(handles=legend_handles, loc="lower left", ncol=6, fontsize=10, handlelength=1.6,
                    columnspacing=1.0, handletextpad=0.4)


def plot_label_bars(ax_bar, true_norm, pred_norm, tick_labels, true_phys=None, pred_phys=None, bare_oxide=False):
    """Bar chart of true vs predicted labels, normalized so 0 and 1 are the training range.

    ``tick_labels`` names each bar. With ``bare_oxide`` (no film) only film thickness
    (index 0) and oxide thickness (index 2) mean anything, so the other bars are faded;
    ``true_phys`` / ``pred_phys`` are then used for the oxide thickness in the title.
    """
    x = np.arange(len(tick_labels))
    width = 0.38
    ax_bar.axhline(0.0, color=ps.AXIS, lw=1.0, zorder=1)
    ax_bar.axhline(1.0, color=ps.AXIS, lw=1.0, zorder=1)
    ax_bar.text(len(x) - 0.45, 1.0, "training range ", ha="right", va="bottom", fontsize=9.5, color=ps.MUTED)
    true_norm = np.asarray(true_norm)
    pred_norm = np.asarray(pred_norm)

    if bare_oxide:
        # With no film, only film_thickness and oxide_thickness mean anything; fade the rest.
        active = [0, 2]
        unused = []
        for i in range(len(tick_labels)):
            if i not in active:
                unused.append(i)
        ax_bar.bar(x[active] - width / 2, true_norm[active], width=width, color=ps.TRUTH, label="truth",
                   edgecolor=ps.SURFACE, linewidth=1.5, zorder=2)
        ax_bar.bar(x[active] + width / 2, pred_norm[active], width=width, color=ps.SLOTS[0], label="network",
                   edgecolor=ps.SURFACE, linewidth=1.5, zorder=2)
        ax_bar.bar(x[unused] + width / 2, pred_norm[unused], width=width, color=ps.SLOTS[0], alpha=0.25,
                   edgecolor=ps.SURFACE, linewidth=1.5, zorder=2)
        bar_title = f"Labels · faded = undefined without a film · oxide {true_phys[2]:.1f} vs {pred_phys[2]:.1f} Å"
    else:
        ax_bar.bar(x - width / 2, true_norm, width=width, color=ps.TRUTH, label="truth",
                   edgecolor=ps.SURFACE, linewidth=1.5, zorder=2)
        ax_bar.bar(x + width / 2, pred_norm, width=width, color=ps.SLOTS[0], label="network",
                   edgecolor=ps.SURFACE, linewidth=1.5, zorder=2)
        bar_title = "Labels, normalized to the training range"

    ax_bar.set_xticks(x, tick_labels, fontsize=10)
    ax_bar.tick_params(axis="x", length=0)
    ax_bar.set_ylabel("normalized value")
    ax_bar.set_title(bar_title)
    y_lo = min(-0.12, float(min(true_norm.min(), pred_norm.min())) - 0.08)
    y_hi = max(1.28, float(max(true_norm.max(), pred_norm.max())) + 0.15)
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.grid(axis="x", visible=False)
    ax_bar.legend(loc="best", ncol=2, fontsize=10.5)
