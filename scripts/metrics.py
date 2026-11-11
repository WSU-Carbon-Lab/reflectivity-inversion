"""W&B metric logging, model checkpointing, and single-sample snapshot plots."""

import math
from pathlib import Path

import matplotlib

# Save figures directly to PNG files without opening an interactive window.
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from refloxide.model import MaterialSLD
import torch
import wandb

from loss import LABEL_NAMES

# W&B project: https://forge.coreweave.com/wandb/als-rixs/reflectivity-inversion
# (Training runs 100% on local hardware; W&B is only used to log metrics & checkpoints.)
WANDB_BASE_URL = "https://api.forge.coreweave.com"
WANDB_ENTITY = "als-rixs"
WANDB_PROJECT = "reflectivity-inversion"

COMPONENT_COLORS = {
    "delta_xx": ("tab:blue", r"$\delta_{xx}$"),
    "delta_zz": ("tab:red", r"$\delta_{zz}$"),
    "beta_xx": ("tab:green", r"$\beta_{xx}$"),
    "beta_zz": ("tab:orange", r"$\beta_{zz}$"),
}


def pick_sample_near_thickness(
    dataset, indices, target_thickness=200.0, exclude_idx=None
):
    """Pick the sample index in ``indices`` with film_thickness closest to ~200 Å."""
    candidates = list(indices[:500])
    best_idx = candidates[0]
    best_diff = float("inf")

    for idx in candidates:
        if idx == exclude_idx and len(candidates) > 1:
            continue
        film_thickness = dataset[idx][3][0].item()
        diff = abs(film_thickness - target_thickness)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx

    return best_idx


def estimate_substrate_sld(q_max):
    """Estimate photon energy from max(q) to get SiO2 and Si optical constants."""
    sin_theta_max = math.sin(math.radians(64.75))
    energy_ev = (float(q_max) * 12398.42) / (4.0 * math.pi * sin_theta_max)
    energy_ev = min(max(energy_ev, 200.0), 500.0)

    ox_tensor = MaterialSLD("SiO2", 2.2).tensor_at_many(np.array([energy_ev]))[0]
    si_tensor = MaterialSLD("Si", 2.33).tensor_at_many(np.array([energy_ev]))[0]
    return (
        float(ox_tensor[0, 0].real),
        float(ox_tensor[0, 0].imag),
        float(si_tensor[0, 0].real),
        float(si_tensor[0, 0].imag),
    )


def compute_depth_profile(labels, z, substrate_sld):
    """Compute erf-smoothed (delta_xx, delta_zz, beta_xx, beta_zz) vs depth z [Å]."""
    ox_delta, ox_beta, si_delta, si_beta = substrate_sld

    film_t = max(float(labels[0]), 0.0)
    film_sigma = max(float(labels[1]), 0.0)
    oxide_t = max(float(labels[2]), 0.0)
    delta_xx = float(labels[3])
    delta_zz = float(labels[4])
    beta_xx = float(labels[5])
    beta_zz = float(labels[6])

    boundaries = [0.0, film_t, film_t + oxide_t]
    roughnesses = [film_sigma, oxide_t / (2.0 * math.pi), 1.2]

    # Optical constants in each slab: [vacuum, film, oxide, silicon].
    slab_values = {
        "delta_xx": [0.0, delta_xx, ox_delta, si_delta],
        "delta_zz": [0.0, delta_zz, ox_delta, si_delta],
        "beta_xx": [0.0, beta_xx, ox_beta, si_beta],
        "beta_zz": [0.0, beta_zz, ox_beta, si_beta],
    }

    z_tensor = torch.from_numpy(z)
    curves = {}
    for name, values in slab_values.items():
        profile = torch.full_like(z_tensor, values[0])
        for i in range(3):
            b = boundaries[i]
            sigma = roughnesses[i]
            if sigma > 0.0:
                step = 0.5 * (1.0 + torch.erf((z_tensor - b) / (math.sqrt(2.0) * sigma)))
            else:
                step = (z_tensor >= b).to(z_tensor.dtype)
            profile = profile + (values[i + 1] - values[i]) * step
        curves[name] = profile.numpy()

    return boundaries, roughnesses, curves


def plot_sample_column(
    axes_col,
    title_prefix,
    q,
    r_s,
    r_p,
    true_phys,
    pred_phys,
    true_norm,
    pred_norm,
    sample_loss,
    substrate_sld,
):
    """Fill one column (Reflectivity, Depth Profile, Label Bars) like single_sample.png."""
    ax_refl, ax_depth, ax_bar = axes_col

    # 1. Top axis: s and p polarized reflectivity vs q (matching single_sample.png).
    ax_refl.plot(
        q, r_s, label="s polarization", color="tab:blue",
        linestyle="", marker="o", markersize=4, markerfacecolor="none",
    )
    ax_refl.plot(
        q, r_p, label="p polarization", color="tab:red",
        linestyle="", marker="o", markersize=4, markerfacecolor="none",
    )
    ax_refl.set_yscale("log")
    ax_refl.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
    ax_refl.set_ylabel("reflectivity")
    ax_refl.set_title(f"{title_prefix} — s and p reflectivity")
    ax_refl.grid(True, which="both", alpha=0.3)
    ax_refl.legend(loc="upper right", fontsize=8)

    # 2. Middle axis: optical-constant depth profile (Truth vs Network).
    # Span both true and predicted interfaces with a 15 Å buffer on each end.
    pad_above = 15.0
    pad_below = 15.0
    true_end = max(float(true_phys[0]) + float(true_phys[2]), 20.0)
    pred_end = min(max(float(pred_phys[0]) + max(float(pred_phys[2]), 0.0), 0.0), 1000.0)
    max_depth = max(true_end, pred_end)
    z = np.linspace(-pad_above, max_depth + pad_below, 1500)

    true_bounds, true_sigmas, true_curves = compute_depth_profile(true_phys, z, substrate_sld)
    pred_bounds, _, pred_curves = compute_depth_profile(pred_phys, z, substrate_sld)

    # Shade vacuum | film | oxide | silicon using the true slab boundaries.
    edges = [z[0], true_bounds[0], true_bounds[1], true_bounds[2], z[-1]]
    regions = ("vacuum", "film", "oxide", "silicon")
    shades = ("0.97", "0.92", "0.84", "0.76")
    xaxis_transform = ax_depth.get_xaxis_transform()
    z_span = z[-1] - z[0]

    for i, (region, shade) in enumerate(zip(regions, shades)):
        ax_depth.axvspan(edges[i], edges[i + 1], color=shade, zorder=0)
        slab_width = edges[i + 1] - edges[i]
        if region == "oxide" and slab_width < 0.05 * z_span:
            continue
        rotate = 90 if slab_width < 0.07 * z_span else 0
        ax_depth.text(
            0.5 * (edges[i] + edges[i + 1]), 0.96, region,
            ha="center", va="top", rotation=rotate,
            fontsize=8.0, color="0.25", transform=xaxis_transform,
        )

    for b in true_bounds:
        ax_depth.axvline(b, color="0.55", linewidth=0.8)
    ax_depth.axvline(pred_bounds[1], color="0.15", linestyle="--", linewidth=1.0)
    ax_depth.axhline(0.0, color="0.6", linewidth=0.6, linestyle=":")

    # Plot Truth as thick translucent ribbons and Network as crisp dashed curves.
    for key, (color, _) in COMPONENT_COLORS.items():
        ax_depth.plot(z, true_curves[key], color=color, linestyle="-", linewidth=3.2, alpha=0.40)
        ax_depth.plot(z, pred_curves[key], color=color, linestyle="--", linewidth=1.6, alpha=1.0)

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=tex)
        for color, tex in COMPONENT_COLORS.values()
    ] + [
        Line2D([0], [0], color="0.35", lw=3.2, alpha=0.45, linestyle="-", label="truth"),
        Line2D([0], [0], color="0.20", lw=1.6, linestyle="--", label="network"),
    ]
    ax_depth.set_xlim(z[0], z[-1])
    ax_depth.set_xlabel(r"depth [$\mathrm{\AA}$]")
    ax_depth.set_ylabel(r"$\delta$, $\beta$")
    ax_depth.set_title(
        f"depth profile: truth film {true_phys[0]:.0f} Å (σ {true_sigmas[0]:.1f} Å) "
        f"vs network {pred_phys[0]:.0f} Å (σ {max(pred_phys[1], 0.0):.1f} Å)"
    )
    ax_depth.margins(y=0.14)
    ax_depth.grid(True, which="both", alpha=0.3)
    ax_depth.legend(handles=legend_handles, loc="lower right", fontsize=7.5, ncol=3)

    # 3. Bottom axis: side-by-side normalized [0, 1] label comparison bar chart.
    x = np.arange(len(LABEL_NAMES))
    width = 0.36
    ax_bar.axhline(0.0, color="0.65", linewidth=0.7, linestyle=":")
    ax_bar.axhline(1.0, color="0.65", linewidth=0.7, linestyle=":")
    ax_bar.bar(x - width / 2, true_norm, width=width, color="0.55", label="truth")
    ax_bar.bar(x + width / 2, pred_norm, width=width, color="tab:blue", label="network")
    ax_bar.set_xticks(x, LABEL_NAMES, rotation=22, ha="right")
    ax_bar.set_ylabel("normalized [0, 1]")
    ax_bar.set_title(f"normalized labels: truth vs network (MSE = {sample_loss:.2e})")
    y_lo = min(-0.10, float(min(min(true_norm), min(pred_norm))) - 0.08)
    y_hi = max(1.15, float(max(max(true_norm), max(pred_norm))) + 0.12)
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(loc="upper right", fontsize=8, ncol=2)


class MetricStore:
    """Handles W&B logging, checkpoint files, and sample snapshot plots."""

    def __init__(self, model, criterion, dataset, train_set, val_set, device, args):
        self.model = model
        self.criterion = criterion
        self.epochs = args.epochs
        self.local = args.local
        self.max_checkpoints = args.max_checkpoints

        # In --local mode we save checkpoints less often so only a few files are stored.
        if args.checkpoint_every is not None:
            self.checkpoint_every = args.checkpoint_every
        elif args.local:
            self.checkpoint_every = 20
        else:
            self.checkpoint_every = 5

        # Pick one training sample and one validation sample with film thickness near ~200 Å.
        self.train_sample_idx = pick_sample_near_thickness(
            dataset, train_set.indices, target_thickness=200.0
        )
        self.val_sample_idx = pick_sample_near_thickness(
            dataset, val_set.indices, target_thickness=200.0, exclude_idx=self.train_sample_idx
        )

        tr_s, tr_p, tr_q, tr_target = dataset[self.train_sample_idx]
        val_s, val_p, val_q, val_target = dataset[self.val_sample_idx]
        self.sample_s = torch.stack((tr_s, val_s)).to(device)
        self.sample_p = torch.stack((tr_p, val_p)).to(device)
        self.sample_q = torch.stack((tr_q, val_q)).to(device)
        self.sample_targets = torch.stack((tr_target, val_target)).to(device)

        # Look up SiO2 / Si substrate optical constants once for both display samples.
        self.train_sld = estimate_substrate_sld(tr_q[-1].item())
        self.val_sld = estimate_substrate_sld(val_q[-1].item())

        # Start Weights & Biases (online to W&B cloud, or offline with --local).
        self.config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mix_channels": args.mix_channels,
            "val_split": args.val_split,
            "seed": args.seed,
            "device": str(device),
            "local": args.local,
            "train_sample_idx": self.train_sample_idx,
            "val_sample_idx": self.val_sample_idx,
        }
        self.run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode="offline" if args.local else "online",
            config=self.config,
            settings=wandb.Settings(base_url=WANDB_BASE_URL),
        )

        run_id = self.run.id if self.run.id else "local-run"
        self.ckpt_dir = Path(args.checkpoint_dir) / run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.saved_checkpoints = []

    @torch.no_grad()
    def store_metrics(self, epoch, train_loss, val_loss, val_mae):
        """Log epoch metrics and save checkpoints + snapshot plots when scheduled."""
        # 1. Predict normalized and physical labels on our two showcase samples.
        self.model.eval()
        pred_norm = self.model(self.sample_s, self.sample_p, self.sample_q)
        target_norm = self.criterion.normalize(self.sample_targets)
        pred_phys = self.criterion.denormalize(pred_norm)

        sq_error = (pred_norm - target_norm).pow(2) * self.criterion.weights
        sample_losses = sq_error.mean(dim=1)
        train_sample_loss = sample_losses[0].item()
        val_sample_loss = sample_losses[1].item()

        # 2. Send metrics to Weights & Biases and print a one-line progress summary.
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "sample/train_sample_loss": train_sample_loss,
            "sample/val_sample_loss": val_sample_loss,
        }
        for label_name, mae_value in zip(LABEL_NAMES, val_mae.tolist()):
            metrics[f"val/mae_{label_name}"] = mae_value
        self.run.log(metrics, step=epoch)

        print(
            f"Epoch {epoch:3d}/{self.epochs} | "
            f"train {train_loss:.2e} | val {val_loss:.2e} | "
            f"train#{self.train_sample_idx} {train_sample_loss:.2e} | "
            f"val#{self.val_sample_idx} {val_sample_loss:.2e} | "
            f"thickness MAE {val_mae[0]:.2f} Å",
            flush=True,
        )

        # 3. Update the best validation checkpoint + snapshot figure on disk.
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint_bundle(
                "best", epoch, train_loss, val_loss,
                target_norm, pred_norm, pred_phys,
                train_sample_loss, val_sample_loss,
                upload_artifact=False,
            )

        # 4. Save periodic checkpoints, upload to W&B, and prune old local files.
        if epoch % self.checkpoint_every == 0 or epoch == self.epochs:
            pair = self._save_checkpoint_bundle(
                f"epoch_{epoch:03d}", epoch, train_loss, val_loss,
                target_norm, pred_norm, pred_phys,
                train_sample_loss, val_sample_loss,
                upload_artifact=not self.local,
            )
            self.saved_checkpoints.append(pair)
            while len(self.saved_checkpoints) > self.max_checkpoints:
                old_ckpt, old_fig = self.saved_checkpoints.pop(0)
                old_ckpt.unlink(missing_ok=True)
                old_fig.unlink(missing_ok=True)

    def _save_checkpoint_bundle(
        self, name, epoch, train_loss, val_loss,
        target_norm, pred_norm, pred_phys,
        train_sample_loss, val_sample_loss, upload_artifact,
    ):
        """Write a .pt weight file and .png snapshot figure, then log to W&B."""
        ckpt_path = self.ckpt_dir / f"{name}.pt"
        fig_path = self.ckpt_dir / f"{name}.png"

        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "label_names": LABEL_NAMES,
                "config": self.config,
            },
            ckpt_path,
        )
        self._save_snapshot_plot(
            fig_path, epoch, train_loss, val_loss,
            target_norm, pred_norm, pred_phys, train_sample_loss, val_sample_loss,
        )

        image_key = "snapshots/best" if name == "best" else "snapshots/checkpoint"
        self.run.log({image_key: wandb.Image(str(fig_path))}, step=epoch)

        if upload_artifact:
            self._upload_artifact(ckpt_path, fig_path, alias=name)

        return ckpt_path, fig_path

    def _upload_artifact(self, ckpt_path, fig_path, alias):
        """Upload a checkpoint (.pt + .png) to W&B as a versioned model artifact."""
        artifact = wandb.Artifact(name=f"reflectivity-net-{self.run.id}", type="model")
        artifact.add_file(str(ckpt_path), name=ckpt_path.name)
        artifact.add_file(str(fig_path), name=fig_path.name)
        self.run.log_artifact(artifact, aliases=[alias])

    def _save_snapshot_plot(
        self, path, epoch, train_loss, val_loss,
        target_norm, pred_norm, pred_phys, train_sample_loss, val_sample_loss,
    ):
        """Save two side-by-side single_sample.png columns (Training and Validation)."""
        fig, axes = plt.subplots(
            3, 2,
            figsize=(16.0, 11.5),
            gridspec_kw={"height_ratios": [1.0, 1.15, 0.75]},
            layout="constrained",
        )

        s_np = self.sample_s.cpu().numpy()
        p_np = self.sample_p.cpu().numpy()
        q_np = self.sample_q.cpu().numpy()
        t_phys = self.sample_targets.cpu().tolist()
        p_phys = pred_phys.cpu().tolist()
        t_norm = target_norm.cpu().tolist()
        p_norm = pred_norm.cpu().tolist()

        # Left column: Training sample near ~200 Å.
        plot_sample_column(
            axes[:, 0],
            f"Training (Row #{self.train_sample_idx})",
            q_np[0], s_np[0], p_np[0],
            t_phys[0], p_phys[0], t_norm[0], p_norm[0],
            train_sample_loss, self.train_sld,
        )

        # Right column: Validation sample near ~200 Å.
        plot_sample_column(
            axes[:, 1],
            f"Validation (Row #{self.val_sample_idx})",
            q_np[1], s_np[1], p_np[1],
            t_phys[1], p_phys[1], t_norm[1], p_norm[1],
            val_sample_loss, self.val_sld,
        )

        fig.suptitle(
            f"Epoch {epoch}  |  Mean Train Loss: {train_loss:.2e}  |  Mean Val Loss: {val_loss:.2e}",
            fontsize=13,
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def finish(self):
        """Upload the final best checkpoint to W&B (when online) and close the run."""
        best_ckpt = self.ckpt_dir / "best.pt"
        best_fig = self.ckpt_dir / "best.png"
        if not self.local and best_ckpt.is_file() and best_fig.is_file():
            self._upload_artifact(best_ckpt, best_fig, alias="best")
        self.run.finish()
        print(f"Best validation loss: {self.best_val_loss:.2e} (saved in {self.ckpt_dir})")
