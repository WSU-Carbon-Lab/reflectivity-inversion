"""The two figures every training run ends with: loss curves and parity plots.

Both models' ``metrics.py`` call these from ``finish()`` so the single-channel and the
multi-channel MLP are judged with the same pictures and the same numbers.

    loss curves   training and validation loss against epoch (log y axis). The two
                  lines should fall together. Validation flattening while training keeps
                  falling is over-fitting.
    parity plots  predicted value against true value, one panel per label. A perfect
                  model puts every point on the diagonal. Each panel prints the label's
                  MAE, RMSE and R^2 (see ``label_errors``).
"""

import math

import matplotlib

# Save figures directly to PNG files without opening an interactive window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from refltron.common import plot_style as ps

ps.apply()  # shared figure style (colours, fonts, hairline grid)

# Parity plots draw at most this many validation rows (drawing 100k+ points is slow and
# just makes a blob). The rows are picked at random with a fixed seed.
MAX_PARITY_POINTS = 20_000

# Unit printed next to each label's errors.
LABEL_UNITS = {
    "film_thickness": "Å",
    "film_roughness": "Å",
    "oxide_thickness": "Å",
    "delta_xx": "",
    "delta_zz": "",
    "beta_xx": "",
    "beta_zz": "",
}
# Pretty panel titles.
LABEL_TITLES = {
    "film_thickness": "film thickness",
    "film_roughness": "film roughness",
    "oxide_thickness": "oxide thickness",
    "delta_xx": r"$\delta_{xx}$",
    "delta_zz": r"$\delta_{zz}$",
    "beta_xx": r"$\beta_{xx}$",
    "beta_zz": r"$\beta_{zz}$",
}


def label_errors(true, predicted):
    """MAE, RMSE and R^2 of one label. ``true`` and ``predicted`` are 1-D arrays.

        MAE   mean absolute error: on average, how far off is a prediction (in the
              label's own units, so Å for thicknesses).
        RMSE  root mean squared error: like MAE but large misses count more.
        R^2   1 - (mean squared error) / (variance of the truth). 1 is perfect, 0 is no
              better than always guessing the average, below 0 is worse than that.
    """
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - true
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt((error ** 2).mean()))
    variance = float(true.var())
    if variance > 0.0:
        r2 = 1.0 - float((error ** 2).mean()) / variance
    else:
        r2 = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_loss_curves(history, path, loss_name, title):
    """Training and validation loss per epoch (left) and film thickness MAE (right).

    ``history`` is a list with one dict per epoch holding "epoch", "train_loss",
    "val_loss" and "thickness_mae" (in Å).
    """
    epochs = []
    train_loss = []
    val_loss = []
    thickness_mae = []
    for row in history:
        epochs.append(row["epoch"])
        train_loss.append(row["train_loss"])
        val_loss.append(row["val_loss"])
        thickness_mae.append(row["thickness_mae"])

    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(14.0, 5.0), layout="constrained")

    ax_loss.plot(epochs, train_loss, color=ps.SLOTS[0], lw=2.0, label="training")
    ax_loss.plot(epochs, val_loss, color=ps.SLOTS[1], lw=2.0, label="validation")
    # Mark the epoch with the lowest validation loss: that is the checkpoint we keep.
    best = int(np.argmin(val_loss))
    ax_loss.plot(epochs[best], val_loss[best], marker="o", ms=7, color=ps.SLOTS[1], ls="")
    ax_loss.annotate(
        f" best: epoch {epochs[best]}, {val_loss[best]:.2e}", (epochs[best], val_loss[best]),
        color=ps.INK_2, fontsize=10.5, va="bottom",
    )
    ax_loss.set_yscale("log")
    ps.no_minor_grid(ax_loss)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel(f"{loss_name} loss")
    ax_loss.set_title(f"Loss per epoch ({loss_name})")
    ax_loss.legend(loc="upper right")

    ax_mae.plot(epochs, thickness_mae, color=ps.SLOTS[2], lw=2.0)
    ax_mae.set_xlabel("epoch")
    ax_mae.set_ylabel("MAE  [Å]")
    ax_mae.set_title("Film thickness error on validation rows")
    ax_mae.set_ylim(bottom=0.0)
    for ax in (ax_loss, ax_mae):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epochs are whole numbers

    fig.suptitle(title, fontsize=14, x=0.008, ha="left", color=ps.INK)
    fig.savefig(path, dpi=ps.FIG_DPI)
    plt.close(fig)


def plot_parity(true, predicted, label_names, path, title, seed=0):
    """Predicted vs true, one panel per label, with MAE / RMSE / R^2 printed in each.

    ``true`` and ``predicted`` are arrays of shape (rows, len(label_names)) in
    physical units. Returns {label: {"mae", "rmse", "r2"}} computed on ALL rows (the
    drawing uses at most MAX_PARITY_POINTS of them).
    """
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    n_labels = len(label_names)

    report = {}
    for i in range(n_labels):
        report[label_names[i]] = label_errors(true[:, i], predicted[:, i])

    # Which rows to draw.
    rng = np.random.default_rng(seed)
    if len(true) > MAX_PARITY_POINTS:
        rows = rng.choice(len(true), size=MAX_PARITY_POINTS, replace=False)
    else:
        rows = np.arange(len(true))

    n_cols = min(n_labels, 4)
    n_rows = math.ceil(n_labels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.6 * n_rows), layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_labels):
        ax = axes[i]
        name = label_names[i]
        t = true[rows, i]
        p = predicted[rows, i]
        ax.scatter(t, p, s=3, alpha=0.18, color=ps.SLOTS[0], edgecolors="none", rasterized=True)
        # The diagonal: where a perfect prediction would sit. Axes share the truth's range.
        low = float(min(t.min(), p.min()))
        high = float(max(t.max(), p.max()))
        pad = 0.04 * (high - low)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=ps.INK_2, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal")
        unit = LABEL_UNITS[name]
        errors = report[name]
        ax.text(
            0.04, 0.96,
            f"MAE {errors['mae']:.3g} {unit}\nRMSE {errors['rmse']:.3g} {unit}\n$R^2$ {errors['r2']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5, color=ps.INK_2,
        )
        if unit:
            ax.set_xlabel(f"true  [{unit}]")
            ax.set_ylabel(f"predicted  [{unit}]")
        else:
            ax.set_xlabel("true")
            ax.set_ylabel("predicted")
        ax.set_title(LABEL_TITLES[name])

    # Hide any unused panels (7 labels on a 2 x 4 grid leaves one empty).
    for j in range(n_labels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, x=0.008, ha="left", color=ps.INK)
    fig.savefig(path, dpi=ps.FIG_DPI)
    plt.close(fig)
    return report
