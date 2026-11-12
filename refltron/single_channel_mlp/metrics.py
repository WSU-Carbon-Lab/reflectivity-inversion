"""Bookkeeping during training: W&B logging, checkpoint files and snapshot figures.

``MetricStore`` is created once by train.py and called once per epoch, exactly like the
multi-channel MLP's MetricStore. Each epoch it
    - logs the losses and errors to Weights & Biases (W&B),
    - saves ``best.pt`` whenever the validation loss improves, plus a numbered
      checkpoint every few epochs,
    - draws a snapshot figure next to each checkpoint. Its two columns are one training
      row and one validation row; its rows show
        1. the input curve exactly as the network sees it: one value per fixed q point,
           so each marker is one of the n_q input neurons,
        2. the depth profile the predicted labels describe, against the truth,
        3. the labels, normalized to the training range.
At the end, ``finish`` reloads the best weights and reports each label's MAE, RMSE and
R^2 on the validation rows (W&B summary + metrics.json).
"""

import json
import math
from pathlib import Path

import matplotlib

# Save figures directly to PNG files without opening an interactive window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from refltron.common import physics
from refltron.common import plot_style as ps
from refltron.common.depth_profile import estimate_substrate_sld
from refltron.common.loss import DepthProfileLoss, expand_to_all_labels, label_range
from refltron.common.snapshot_plots import plot_depth_profile_panel, plot_label_bars
from refltron.common.tracking import WANDB_BASE_URL, WANDB_ENTITY, WANDB_PROJECT

ps.apply()  # shared figure style (colours, fonts, hairline grid)

# Snapshot rows are picked with a film thickness close to this [Å].
SNAPSHOT_THICKNESS = 200.0

# Short tick labels for the label bar chart.
TICKS = {
    "film_thickness": "film\nthickness",
    "film_roughness": "film\nroughness",
    "oxide_thickness": "oxide\nthickness",
    "delta_xx": r"$\delta_{xx}$",
    "delta_zz": r"$\delta_{zz}$",
    "beta_xx": r"$\beta_{xx}$",
    "beta_zz": r"$\beta_{zz}$",
}
# How each input curve is labelled on the snapshot's y axis.
INPUT_AXIS_LABELS = {
    "s": r"$\ln R_s$",
    "p": r"$\ln R_p$",
    "anisotropy": r"$A = (P - S c)\,/\,(P + S c)$,  $c = \cos^2 2\theta$",
}


def pick_sample_near_thickness(dataset, indices, label_min, label_max, target_thickness=200.0, exclude_idx=None):
    """Pick the row in ``indices`` (first 500 checked) whose film thickness is closest to the target."""
    best_idx = indices[0]
    best_diff = math.inf
    for idx in indices[:500]:
        if idx == exclude_idx:
            continue
        y = dataset[idx][1]  # the scaled labels of this row
        thickness = y[0].item() * (label_max[0] - label_min[0]) + label_min[0]
        diff = abs(thickness - target_thickness)
        if diff < best_diff:
            best_idx = idx
            best_diff = diff
    return best_idx


class MetricStore:
    """Handles W&B logging, checkpoint files, snapshot figures and the final error report."""

    def __init__(self, model, dataset, train_set, val_set, q_grid, label_names, device, args, optimizer=None):
        self.model = model
        self.optimizer = optimizer  # saved inside checkpoints
        self.dataset = dataset
        self.val_set = val_set
        self.device = device
        self.input_name = args.input
        self.label_names = list(label_names)
        self.label_min, self.label_max = label_range(self.label_names)
        self.q_grid = np.asarray(q_grid)
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

        # Every curve is on the same fixed q grid at one photon energy, so the substrate
        # optical constants are the same for every row.
        self.substrate = estimate_substrate_sld(self.q_grid)
        self.profile_metric = DepthProfileLoss().to(device)

        # One training and one validation row with a film near SNAPSHOT_THICKNESS.
        self.train_sample_idx = pick_sample_near_thickness(
            dataset, train_set.indices, self.label_min, self.label_max, SNAPSHOT_THICKNESS
        )
        self.val_sample_idx = pick_sample_near_thickness(
            dataset, val_set.indices, self.label_min, self.label_max, SNAPSHOT_THICKNESS,
            exclude_idx=self.train_sample_idx,
        )
        train_x, train_y = dataset[self.train_sample_idx]
        val_x, val_y = dataset[self.val_sample_idx]
        self.sample_x = torch.stack((train_x, val_x)).to(device)
        self.sample_y = torch.stack((train_y, val_y)).to(device)
        self.column_titles = [f"Training (Row #{self.train_sample_idx})", f"Validation (Row #{self.val_sample_idx})"]

        # Start Weights & Biases (offline with --local, the default here; sync later).
        if args.local:
            wandb_mode = "offline"
        else:
            wandb_mode = "online"
        if args.run_name is None:
            run_name = f"single-{args.input}"
        else:
            run_name = args.run_name
        self.config = {
            "network": "OldNet",
            "input": args.input,
            "labels": self.label_names,
            "label_min": self.label_min.tolist(),
            "label_max": self.label_max.tolist(),
            "n_q": len(self.q_grid),
            "q_range": [float(self.q_grid[0]), float(self.q_grid[-1])],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_split": args.val_split,
            "seed": args.seed,
            "device": str(device),
            "local": args.local,
            "max_samples": args.max_samples,
            "run_name": args.run_name,
            "train_sample_idx": self.train_sample_idx,
            "val_sample_idx": self.val_sample_idx,
        }
        self.run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode=wandb_mode,
            config=self.config,
            name=run_name,
            settings=wandb.Settings(base_url=WANDB_BASE_URL),
        )
        self.ckpt_dir = Path(args.checkpoint_dir) / self.run.id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = math.inf
        self.saved_checkpoints = []

    def to_profile_labels(self, normalized):
        """Scaled network outputs or targets -> all 7 physical labels (for depth profiles)."""
        scale = torch.tensor(self.label_max - self.label_min, device=normalized.device)
        offset = torch.tensor(self.label_min, device=normalized.device)
        return expand_to_all_labels(normalized * scale + offset, self.label_names)

    def profile_distance(self, pred_normalized, target_normalized):
        """Depth-profile distance of each row (the multi-channel MLP's profile metric)."""
        pred_phys = self.to_profile_labels(pred_normalized)
        true_phys = self.to_profile_labels(target_normalized)
        substrate = torch.tensor(self.substrate, dtype=torch.float32, device=pred_phys.device)
        return self.profile_metric.per_sample_loss(
            self.profile_metric.normalize(pred_phys), true_phys, substrate=substrate.expand(len(pred_phys), 4)
        )

    @torch.no_grad()
    def store_metrics(self, epoch, train_loss, val_loss, val_mae, val_profile_l2, dead_outputs):
        """Log epoch metrics and save checkpoints + snapshot plots when scheduled."""
        # 1. Predict the two showcase rows.
        self.model.eval()
        pred_norm = self.model(self.sample_x)
        profile_losses = self.profile_distance(pred_norm, self.sample_y).cpu().tolist()

        # 2. Send metrics to W&B (same keys as the multi-channel MLP where they overlap).
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/label_mse": val_loss,
            "val/profile_l2": val_profile_l2,
            "val/dead_outputs": dead_outputs,
            "sample/train_profile_l2": profile_losses[0],
            "sample/val_profile_l2": profile_losses[1],
        }
        for i in range(len(self.label_names)):
            metrics[f"val/mae_{self.label_names[i]}"] = val_mae[i].item()
        self.run.log(metrics, step=epoch)

        print(
            f"Epoch {epoch:3d}/{self.epochs} | train(mse) {train_loss:.2e} | val {val_loss:.2e} | "
            f"thickness MAE {val_mae[0]:.1f} Å | profile L2 {val_profile_l2:.2e} | dead outputs {dead_outputs}",
            flush=True,
        )

        # 3. Update the best validation checkpoint + snapshot figure on disk.
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint_bundle("best", epoch, train_loss, val_loss, pred_norm, profile_losses,
                                         upload_artifact=False)

        # 4. Save periodic checkpoints, upload to W&B (online only), and prune old local files.
        if epoch % self.checkpoint_every == 0 or epoch == self.epochs:
            pair = self._save_checkpoint_bundle(f"epoch_{epoch:03d}", epoch, train_loss, val_loss, pred_norm,
                                                profile_losses, upload_artifact=not self.local)
            self.saved_checkpoints.append(pair)
            while len(self.saved_checkpoints) > self.max_checkpoints:
                old_ckpt, old_fig = self.saved_checkpoints.pop(0)
                old_ckpt.unlink(missing_ok=True)
                old_fig.unlink(missing_ok=True)

    def _save_checkpoint_bundle(self, name, epoch, train_loss, val_loss, pred_norm, profile_losses, upload_artifact):
        """Write a .pt weight file and .png snapshot figure, then log the figure to W&B."""
        ckpt_path = self.ckpt_dir / f"{name}.pt"
        fig_path = self.ckpt_dir / f"{name}.png"
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "label_names": self.label_names,
            "config": self.config,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, ckpt_path)
        self._save_snapshot_plot(fig_path, epoch, train_loss, val_loss, pred_norm, profile_losses)

        if name == "best":
            image_key = "snapshots/best"
        else:
            image_key = "snapshots/checkpoint"
        self.run.log({image_key: wandb.Image(str(fig_path))}, step=epoch)
        if upload_artifact:
            self._upload_artifact(ckpt_path, fig_path, alias=name)
        return ckpt_path, fig_path

    def _upload_artifact(self, ckpt_path, fig_path, alias):
        """Upload a checkpoint (.pt + .png) to W&B as a versioned model artifact."""
        artifact = wandb.Artifact(name=f"single-channel-mlp-{self.run.id}", type="model")
        artifact.add_file(str(ckpt_path), name=ckpt_path.name)
        artifact.add_file(str(fig_path), name=fig_path.name)
        self.run.log_artifact(artifact, aliases=[alias])

    def _save_snapshot_plot(self, path, epoch, train_loss, val_loss, pred_norm, profile_losses):
        """Save 2 columns (training row, validation row) x 3 rows (input, depth profile, labels)."""
        fig, axes = plt.subplots(
            3, 2, figsize=(16.0, 11.5), gridspec_kw={"height_ratios": [1.0, 1.15, 0.78]}, layout="constrained"
        )
        x_np = self.sample_x.cpu().numpy()
        true_phys = self.to_profile_labels(self.sample_y).cpu().tolist()
        pred_phys = self.to_profile_labels(pred_norm).cpu().tolist()
        true_norm = self.sample_y.cpu().numpy()
        pred_norm_np = pred_norm.cpu().numpy()
        ticks = []
        for name in self.label_names:
            ticks.append(TICKS[name])

        for col in range(2):
            ax_input, ax_depth, ax_bar = axes[:, col]
            self._plot_input(ax_input, x_np[col], self.column_titles[col])
            plot_depth_profile_panel(ax_depth, true_phys[col], pred_phys[col], self.substrate, profile_losses[col])
            plot_label_bars(ax_bar, true_norm[col], pred_norm_np[col], ticks, true_phys[col], pred_phys[col])

        fig.suptitle(
            f"Single-channel MLP · input {self.input_name} · epoch {epoch}   ·   "
            f"MSE loss: train {train_loss:.2e}, validation {val_loss:.2e}",
            fontsize=16, x=0.008, ha="left", color=ps.INK,
        )
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _plot_input(self, ax, curve, title):
        """The input vector vs q: each marker is one input neuron at its fixed q value."""
        ax.plot(self.q_grid, curve, color=ps.SLOTS[0], lw=0.8, alpha=0.5)
        ax.plot(self.q_grid, curve, color=ps.SLOTS[0], ls="", marker="o", ms=2.2)
        if self.input_name == "anisotropy":
            # A is bounded by +-1; mark where cos(2 theta) = 0 (theta = 45 deg, the Brewster angle for n ~ 1).
            for level in (-1.0, 1.0):
                ax.axhline(level, color=ps.AXIS, lw=1.0)
            wavelength = physics.HC_EV_ANGSTROM / physics.estimate_energy(float(self.q_grid[0]), float(self.q_grid[-1]))
            q_brewster = 4.0 * math.pi / wavelength * math.sin(math.radians(45.0))
            if q_brewster <= self.q_grid[-1]:
                ax.axvline(q_brewster, color=ps.MUTED, ls=(0, (4, 3)), lw=1.0)
                ax.text(q_brewster, 0.97, " θ = 45°", color=ps.MUTED, fontsize=9.5, va="top",
                        transform=ax.get_xaxis_transform())
        ax.set_xlim(0, self.q_grid[-1] * 1.03)
        ax.set_xlabel(r"$q$  [$\mathrm{\AA}^{-1}$]")
        ax.set_ylabel(INPUT_AXIS_LABELS[self.input_name])
        ax.set_title(f"{title} · network input: {len(self.q_grid)} values on the fixed q grid")

    @torch.no_grad()
    def finish(self):
        """Reload the best weights, report each label's validation errors, and close the run."""
        best = torch.load(self.ckpt_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(best["model_state_dict"])
        self.model.eval()

        predictions = []
        targets = []
        for x, y in DataLoader(self.val_set, batch_size=8192):
            predictions.append(self.model(x.to(self.device)).cpu())
            targets.append(y)
        scale = torch.tensor(self.label_max - self.label_min, dtype=torch.float64)
        offset = torch.tensor(self.label_min, dtype=torch.float64)
        predicted = torch.cat(predictions).double() * scale + offset
        true = torch.cat(targets).double() * scale + offset

        # R^2 = 1 - (mean squared error) / (variance of the truth): 1 is perfect, 0 is no
        # better than always guessing the average.
        report = {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "labels": {}}
        print(f"\nBest epoch {best['epoch']} (val loss {best['val_loss']:.5f}). Validation errors:")
        for i in range(len(self.label_names)):
            name = self.label_names[i]
            error = predicted[:, i] - true[:, i]
            mae = error.abs().mean().item()
            rmse = error.pow(2).mean().sqrt().item()
            r2 = (1.0 - error.pow(2).mean() / true[:, i].var()).item()
            report["labels"][name] = {"mae": mae, "rmse": rmse, "r2": r2}
            self.run.summary[f"best/{name}/mae"] = mae
            self.run.summary[f"best/{name}/rmse"] = rmse
            self.run.summary[f"best/{name}/r2"] = r2
            print(f"  {name:16s} MAE {mae:.4g}   RMSE {rmse:.4g}   R^2 {r2:+.3f}")
        self.run.summary["best/epoch"] = best["epoch"]
        self.run.summary["best/val_loss"] = best["val_loss"]
        with open(self.ckpt_dir / "metrics.json", "w") as file:
            json.dump(report, file, indent=2)

        best_fig = self.ckpt_dir / "best.png"
        if not self.local and best_fig.is_file():
            self._upload_artifact(self.ckpt_dir / "best.pt", best_fig, alias="best")
        self.run.finish()
        print(f"Best validation loss: {self.best_val_loss:.2e} (saved in {self.ckpt_dir})")
