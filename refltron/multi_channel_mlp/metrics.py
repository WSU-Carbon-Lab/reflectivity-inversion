"""Bookkeeping during training: W&B logging, checkpoint files and snapshot figures.

``MetricStore`` is created once by train.py and called once per epoch. Each epoch it
    - logs the losses and errors to Weights & Biases (W&B),
    - saves ``best.pt`` whenever the validation loss improves, plus a numbered
      checkpoint every few epochs,
    - draws a snapshot figure next to each checkpoint. Its four columns are one training
      row, one validation row and two out-of-distribution (OOD) test curves the network
      never saw: a bare oxide with no film, and a 2000 Å film (thicker than any in training).
At the end, ``finish`` reloads the best weights and reports each label's MAE, RMSE and
R^2 on the validation rows (W&B summary + metrics.json), and saves the two standard
end-of-training figures next to them: ``loss_curves.png`` and ``parity.png``
(see refltron/common/training_plots.py). The single-channel MLP reports the same.
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
from refloxide.model import FreeUniTensor, MaterialSLD, ReflectModel
from torch.utils.data import DataLoader

import wandb
from refltron.common import physics
from refltron.common import plot_style as ps
from refltron.common.depth_profile import estimate_substrate_sld
from refltron.common.loss import LABEL_NAMES, DepthProfileLoss
from refltron.common.snapshot_plots import plot_depth_profile_panel, plot_label_bars
from refltron.common.tracking import WANDB_BASE_URL, WANDB_ENTITY, WANDB_PROJECT
from refltron.common.training_plots import plot_loss_curves, plot_parity

ps.apply()  # shared figure style (colours, fonts, hairline grid)

# W&B project: https://forge.coreweave.com/wandb/als-rixs/reflectivity-inversion
# (Training runs 100% on local hardware; W&B is only used to log metrics & checkpoints.)
# The W&B settings themselves live in refltron/common/tracking.py.

# Snapshot rows are picked with a film thickness close to this [Å].
SNAPSHOT_THICKNESS = 200.0

# Short tick labels for the label bar chart, in LABEL_NAMES order.
LABEL_TICKS = ("film\nthickness", "film\nroughness", "oxide\nthickness",
               r"$\delta_{xx}$", r"$\delta_{zz}$", r"$\beta_{xx}$", r"$\beta_{zz}$")


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




def _oxide_and_silicon(oxide_t):
    """refloxide layers for a native oxide of thickness ``oxide_t`` [Å] on silicon."""
    oxide = MaterialSLD(physics.OXIDE_FORMULA, physics.OXIDE_DENSITY, name="oxide")(
        oxide_t, physics.oxide_roughness(oxide_t)
    )
    silicon = MaterialSLD(physics.SILICON_FORMULA, physics.SILICON_DENSITY, name="silicon")(
        0.0, physics.SILICON_ROUGHNESS
    )
    return oxide | silicon


def build_ood_test_cases(n_q=256, seed=2026):
    """Simulate two dedicated out-of-distribution (OOD) physical benchmarks.

    Case 1 — Bare Oxide + Silicon Substrate (0 Å film, 10 Å native SiO2 at 285 eV):
        Tests whether the network can recognize the complete absence of a film,
        below the 20 Å minimum thickness in the training set.

    Case 2 — Very Thick Film (2,000 Å uniaxial film at 250 eV):
        Tests whether the network generalizes to rapid Kiessig fringes 2.5x
        thicker than the 800 Å maximum in the training set (simulated at 250 eV
        so 256 q points remain well inside the Nyquist fringe spacing).
    """
    rng = np.random.default_rng(seed)

    # Case 1: True 3-layer stack (vacuum | SiO2 | Si) with no organic film at all.
    e_bare = 285.0
    theta_bare = np.sort(rng.uniform(0.5, 65.0, size=n_q))
    q_bare = (4.0 * math.pi * e_bare / physics.HC_EV_ANGSTROM) * np.sin(np.deg2rad(theta_bare))
    ox_t_bare = 10.0
    vac_1 = MaterialSLD("", 0.0, name="vacuum")(0.0, 0.0)
    res_bare = ReflectModel(vac_1 | _oxide_and_silicon(ox_t_bare), energies=[e_bare], parallel=False)(
        q_bare, e_bare
    )
    target_bare = [0.0, 0.0, ox_t_bare, 0.0, 0.0, 0.0, 0.0]

    # Case 2: 2,000 Å anisotropic film | 8 Å SiO2 | Si at 250 eV.
    e_thick = 250.0
    theta_thick = np.sort(rng.uniform(0.5, 65.0, size=n_q))
    q_thick = (4.0 * math.pi * e_thick / physics.HC_EV_ANGSTROM) * np.sin(np.deg2rad(theta_thick))
    film_t = 2000.0
    film_sigma = 12.0
    ox_t_thick = 8.0
    d_xx = 2.2e-3
    d_zz = 1.1e-3
    b_xx = 0.8e-3
    b_zz = 0.3e-3

    vac_2 = MaterialSLD("", 0.0, name="vacuum")(0.0, 0.0)
    film_2 = FreeUniTensor(energies=[e_thick], name="film")
    ch = film_2.channel_at(e_thick)
    ch.delta_o.setp(d_xx)
    ch.beta_o.setp(b_xx)
    ch.delta_e.setp(d_zz)
    ch.beta_e.setp(b_zz)
    stack_thick = vac_2 | film_2(film_t, film_sigma) | _oxide_and_silicon(ox_t_thick)
    res_thick = ReflectModel(stack_thick, energies=[e_thick], parallel=False)(
        q_thick, e_thick
    )
    target_thick = [film_t, film_sigma, ox_t_thick, d_xx, d_zz, b_xx, b_zz]

    return [
        {
            "title": "OOD: Bare Oxide (0 Å film)",
            "q": torch.tensor(q_bare, dtype=torch.float32),
            "s": torch.tensor(res_bare.s, dtype=torch.float32),
            "p": torch.tensor(res_bare.p, dtype=torch.float32),
            "target": torch.tensor(target_bare, dtype=torch.float32),
            "sld": physics.substrate_optical_constants(e_bare),
            "bare_oxide": True,
        },
        {
            "title": "OOD: Thick Film (2000 Å)",
            "q": torch.tensor(q_thick, dtype=torch.float32),
            "s": torch.tensor(res_thick.s, dtype=torch.float32),
            "p": torch.tensor(res_thick.p, dtype=torch.float32),
            "target": torch.tensor(target_thick, dtype=torch.float32),
            "sld": physics.substrate_optical_constants(e_thick),
            "bare_oxide": False,
        },
    ]




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
    profile_l2,
    substrate_sld,
    bare_oxide=False,
):
    """Fill one column: reflectivity (top), depth profile (middle), labels (bottom)."""
    ax_refl = axes_col[0]
    ax_depth = axes_col[1]
    ax_bar = axes_col[2]

    # 1. Top: s and p polarized reflectivity vs q, labelled at the curve ends.
    s_color = ps.POLARIZATION_COLORS["s"]
    ax_refl.plot(q, r_s, color=s_color, lw=1.3)
    ax_refl.text(q[-1], r_s[-1], "  s", color=s_color, fontsize=11.5, va="center", weight="medium")
    p_color = ps.POLARIZATION_COLORS["p"]
    ax_refl.plot(q, r_p, color=p_color, lw=1.3)
    ax_refl.text(q[-1], r_p[-1], "  p", color=p_color, fontsize=11.5, va="center", weight="medium")
    ax_refl.set_yscale("log")
    ps.no_minor_grid(ax_refl)
    ax_refl.set_xlim(0, q[-1] * 1.08)
    ax_refl.set_xlabel(r"$q$  [$\mathrm{\AA}^{-1}$]")
    ax_refl.set_ylabel("reflectivity")
    ax_refl.set_title(title_prefix)

    # 2. Middle: optical constants vs height (truth vs network).
    plot_depth_profile_panel(ax_depth, true_phys, pred_phys, substrate_sld, profile_l2)

    # 3. Bottom: the 7 labels, normalized so 0 and 1 are the edges of the training range.
    plot_label_bars(ax_bar, true_norm, pred_norm, LABEL_TICKS, true_phys, pred_phys, bare_oxide=bare_oxide)


def save_snapshot_figure(
    path, epoch, loss_name, train_loss, val_loss,
    column_titles, column_bare_flags, column_slds,
    sample_s, sample_p, sample_q, sample_targets,
    target_norm, pred_norm, pred_phys, profile_losses,
):
    """Save the snapshot figure: 4 columns (Training, Validation, OOD Bare Oxide, OOD 2000 Å).

    Every ``column_*`` list and every ``sample_*`` / ``*_norm`` / ``pred_phys`` tensor has
    one entry per column. Used by MetricStore during training and by render_snapshot.py.
    """
    fig, axes = plt.subplots(
        3, 4,
        figsize=(26.0, 11.5),
        gridspec_kw={"height_ratios": [1.0, 1.15, 0.78]},
        layout="constrained",
    )

    s_np = sample_s.cpu().numpy()
    p_np = sample_p.cpu().numpy()
    q_np = sample_q.cpu().numpy()
    t_phys = sample_targets.cpu().tolist()
    p_phys = pred_phys.cpu().tolist()
    t_norm = target_norm.cpu().tolist()
    p_norm = pred_norm.cpu().tolist()

    for col in range(4):
        plot_sample_column(
            axes[:, col],
            column_titles[col],
            q_np[col], s_np[col], p_np[col],
            t_phys[col], p_phys[col], t_norm[col], p_norm[col],
            profile_losses[col],
            column_slds[col],
            bare_oxide=column_bare_flags[col],
        )

    fig.suptitle(
        f"Epoch {epoch}   ·   {loss_name} loss: train {train_loss:.2e}, validation {val_loss:.2e}   ·   "
        f"out-of-distribution profile L2: bare oxide {profile_losses[2]:.1e}, 2000 Å film {profile_losses[3]:.1e}",
        fontsize=16, x=0.008, ha="left", color=ps.INK,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


class MetricStore:
    """Handles W&B logging, checkpoint files, and 4-column sample + OOD snapshot plots."""

    def __init__(
        self, model, criterion, dataset, train_set, val_set, device, args,
        optimizer=None, scheduler=None, resume=None,
    ):
        self.model = model
        # Saved inside checkpoints so a stopped run can be continued exactly (--resume).
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.loss_name = args.loss
        self.val_set = val_set  # finish() predicts every validation row with the best weights
        self.device = device
        # The snapshot figures always report the depth-profile distance, whatever the training loss.
        self.profile_metric = DepthProfileLoss().to(device)
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

        # Pick one training sample and one validation sample with a film near SNAPSHOT_THICKNESS.
        self.train_sample_idx = pick_sample_near_thickness(
            dataset, train_set.indices, target_thickness=SNAPSHOT_THICKNESS
        )
        self.val_sample_idx = pick_sample_near_thickness(
            dataset, val_set.indices, target_thickness=SNAPSHOT_THICKNESS, exclude_idx=self.train_sample_idx
        )

        # Each dataset row is (s, p, q, labels).
        train_row = dataset[self.train_sample_idx]
        tr_s = train_row[0]
        tr_p = train_row[1]
        tr_q = train_row[2]
        tr_target = train_row[3]
        val_row = dataset[self.val_sample_idx]
        val_s = val_row[0]
        val_p = val_row[1]
        val_q = val_row[2]
        val_target = val_row[3]
        ood_cases = build_ood_test_cases()

        # Stack all 4 showcase columns: [Train, Val, OOD Bare Oxide, OOD Thick 2000 Å].
        self.column_titles = [
            f"Training (Row #{self.train_sample_idx})",
            f"Validation (Row #{self.val_sample_idx})",
            ood_cases[0]["title"],
            ood_cases[1]["title"],
        ]
        self.column_bare_flags = [False, False, ood_cases[0]["bare_oxide"], ood_cases[1]["bare_oxide"]]
        self.column_slds = [
            estimate_substrate_sld(tr_q),
            estimate_substrate_sld(val_q),
            ood_cases[0]["sld"],
            ood_cases[1]["sld"],
        ]
        self.sample_sld_tensor = torch.tensor(self.column_slds, dtype=torch.float32, device=device)

        self.sample_s = torch.stack((tr_s, val_s, ood_cases[0]["s"], ood_cases[1]["s"])).to(device)
        self.sample_p = torch.stack((tr_p, val_p, ood_cases[0]["p"], ood_cases[1]["p"])).to(device)
        self.sample_q = torch.stack((tr_q, val_q, ood_cases[0]["q"], ood_cases[1]["q"])).to(device)
        self.sample_targets = torch.stack(
            (tr_target, val_target, ood_cases[0]["target"], ood_cases[1]["target"])
        ).to(device)

        # Start Weights & Biases (online to W&B cloud, or offline with --local).
        self.config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "mix_channels": args.mix_channels,
            "regularizer": model.regularizer,
            "loss": self.loss_name,
            "val_split": args.val_split,
            "seed": args.seed,
            "device": str(device),
            "local": args.local,
            "activation": model.activation,
            "normalize_inputs": model.normalize_inputs,
            "lr_schedule": args.lr_schedule,
            "lr_min_frac": args.lr_min_frac,
            "max_samples": args.max_samples,
            "profile_alignment": "silicon",
            "run_name": args.run_name,
            "train_sample_idx": self.train_sample_idx,
            "val_sample_idx": self.val_sample_idx,
        }
        if resume is not None:
            self.config["resumed_from"] = str(args.resume)
            self.config["resumed_at_epoch"] = resume["epoch"]
        if args.local:
            wandb_mode = "offline"
        else:
            wandb_mode = "online"
        self.run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode=wandb_mode,
            config=self.config,
            name=args.run_name,
            settings=wandb.Settings(base_url=WANDB_BASE_URL),
        )

        # A resumed run keeps writing into the original run's folder.
        if resume is not None:
            self.ckpt_dir = Path(args.resume).parent
        else:
            run_id = self.run.id
            if not run_id:
                run_id = "local-run"
            self.ckpt_dir = Path(args.checkpoint_dir) / run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        if resume is not None:
            best_path = self.ckpt_dir / "best.pt"
            if best_path.is_file():
                best = torch.load(best_path, map_location="cpu", weights_only=False)
                self.best_val_loss = best["val_loss"]
        self.saved_checkpoints = []
        # One entry per epoch, for the loss-curve figure drawn by finish().
        self.history = []

    @torch.no_grad()
    def store_metrics(
        self, epoch, train_loss, val_loss, val_mae, val_profile_l2=None, val_mse=None, lr=None
    ):
        """Log epoch metrics and save checkpoints + 4-column snapshot plots when scheduled."""
        # 1. Predict normalized and physical labels on our 4 showcase columns
        #    (Train, Val, OOD Bare Oxide 0 Å, OOD Thick Film 2000 Å).
        self.model.eval()
        pred_norm = self.model(self.sample_s, self.sample_p, self.sample_q)
        target_norm = self.criterion.normalize(self.sample_targets)
        pred_phys = self.criterion.denormalize(pred_norm)

        profile_losses = self.profile_metric.per_sample_loss(
            pred_norm, self.sample_targets, substrate=self.sample_sld_tensor
        ).cpu().tolist()
        train_prof_l2, val_prof_l2, bare_prof_l2, thick_prof_l2 = profile_losses

        # 2. Send metrics (including cross-loss & OOD diagnostics) to W&B.
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "sample/train_profile_l2": train_prof_l2,
            "sample/val_profile_l2": val_prof_l2,
            "ood/bare_oxide_profile_l2": bare_prof_l2,
            "ood/bare_oxide_pred_thickness": pred_phys[2, 0].item(),
            "ood/thick_2000A_profile_l2": thick_prof_l2,
            "ood/thick_2000A_pred_thickness": pred_phys[3, 0].item(),
        }
        if lr is not None:
            metrics["train/lr"] = lr
        if val_profile_l2 is not None:
            metrics["val/profile_l2"] = val_profile_l2
        if val_mse is not None:
            metrics["val/label_mse"] = val_mse
        mae_values = val_mae.tolist()
        for i in range(len(LABEL_NAMES)):
            metrics[f"val/mae_{LABEL_NAMES[i]}"] = mae_values[i]
        self.run.log(metrics, step=epoch)
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "thickness_mae": mae_values[0],
        })

        print(
            f"Epoch {epoch:3d}/{self.epochs} | "
            f"train({self.loss_name}) {train_loss:.2e} | val {val_loss:.2e} | "
            f"thickness MAE {val_mae[0]:.1f} Å | "
            f"OOD 0Å->{pred_phys[2, 0].item():.1f} Å | "
            f"OOD 2000Å->{pred_phys[3, 0].item():.0f} Å",
            flush=True,
        )

        # 3. Update the best validation checkpoint + snapshot figure on disk.
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint_bundle(
                "best", epoch, train_loss, val_loss,
                target_norm, pred_norm, pred_phys, profile_losses,
                upload_artifact=False,
            )

        # 4. Save periodic checkpoints, upload to W&B, and prune old local files.
        if epoch % self.checkpoint_every == 0 or epoch == self.epochs:
            pair = self._save_checkpoint_bundle(
                f"epoch_{epoch:03d}", epoch, train_loss, val_loss,
                target_norm, pred_norm, pred_phys, profile_losses,
                upload_artifact=not self.local,
            )
            self.saved_checkpoints.append(pair)
            while len(self.saved_checkpoints) > self.max_checkpoints:
                oldest = self.saved_checkpoints.pop(0)  # (checkpoint file, figure file)
                oldest[0].unlink(missing_ok=True)
                oldest[1].unlink(missing_ok=True)

    def _save_checkpoint_bundle(
        self, name, epoch, train_loss, val_loss,
        target_norm, pred_norm, pred_phys, profile_losses, upload_artifact,
    ):
        """Write a .pt weight file and .png snapshot figure, then log to W&B."""
        ckpt_path = self.ckpt_dir / f"{name}.pt"
        fig_path = self.ckpt_dir / f"{name}.png"

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "label_names": LABEL_NAMES,
            "config": self.config,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, ckpt_path)
        self._save_snapshot_plot(
            fig_path, epoch, train_loss, val_loss,
            target_norm, pred_norm, pred_phys, profile_losses,
        )

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
        artifact = wandb.Artifact(name=f"reflectivity-net-{self.run.id}", type="model")
        artifact.add_file(str(ckpt_path), name=ckpt_path.name)
        artifact.add_file(str(fig_path), name=fig_path.name)
        self.run.log_artifact(artifact, aliases=[alias])

    def _save_snapshot_plot(
        self, path, epoch, train_loss, val_loss,
        target_norm, pred_norm, pred_phys, profile_losses,
    ):
        """Save 4 side-by-side columns: Training, Validation, OOD Bare Oxide, OOD 2000 Å."""
        save_snapshot_figure(
            path, epoch, self.loss_name, train_loss, val_loss,
            self.column_titles, self.column_bare_flags, self.column_slds,
            self.sample_s, self.sample_p, self.sample_q, self.sample_targets,
            target_norm, pred_norm, pred_phys, profile_losses,
        )

    @torch.no_grad()
    def finish(self):
        """Reload the best weights, report each label's validation errors, draw the final figures, close the run."""
        best_ckpt = self.ckpt_dir / "best.pt"
        best_fig = self.ckpt_dir / "best.png"
        best = torch.load(best_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(best["model_state_dict"])
        self.model.eval()

        # Predict every validation row with the best weights, in physical units.
        predictions = []
        targets = []
        for s, p, q, y in DataLoader(self.val_set, batch_size=4096):
            pred_norm = self.model(s.to(self.device), p.to(self.device), q.to(self.device))
            predictions.append(self.criterion.denormalize(pred_norm).cpu())
            targets.append(y)
        predicted = torch.cat(predictions).double().numpy()
        true = torch.cat(targets).double().numpy()

        # 1. Loss curves: training and validation loss per epoch.
        loss_fig = self.ckpt_dir / "loss_curves.png"
        plot_loss_curves(
            self.history, loss_fig, loss_name=self.loss_name,
            title=f"Multi-channel MLP (Refltron) · {self.loss_name} loss · {len(self.history)} epochs",
        )

        # 2. Parity plots: predicted vs true for every label, with MAE / RMSE / R^2.
        parity_fig = self.ckpt_dir / "parity.png"
        errors = plot_parity(
            true, predicted, list(LABEL_NAMES), parity_fig,
            title=f"Multi-channel MLP (Refltron) · best epoch {best['epoch']} · validation rows",
        )

        # 3. Write the numbers to the terminal, W&B and metrics.json.
        report = {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "labels": errors}
        print(f"\nBest epoch {best['epoch']} (val loss {best['val_loss']:.5f}). Validation errors:")
        for name in LABEL_NAMES:
            e = errors[name]
            self.run.summary[f"best/{name}/mae"] = e["mae"]
            self.run.summary[f"best/{name}/rmse"] = e["rmse"]
            self.run.summary[f"best/{name}/r2"] = e["r2"]
            print(f"  {name:16s} MAE {e['mae']:.4g}   RMSE {e['rmse']:.4g}   R^2 {e['r2']:+.3f}")
        self.run.summary["best/epoch"] = best["epoch"]
        self.run.summary["best/val_loss"] = best["val_loss"]
        with open(self.ckpt_dir / "metrics.json", "w") as file:
            json.dump(report, file, indent=2)
        self.run.log({"final/loss_curves": wandb.Image(str(loss_fig)), "final/parity": wandb.Image(str(parity_fig))},
                     step=self.epochs)

        if not self.local and best_ckpt.is_file() and best_fig.is_file():
            self._upload_artifact(best_ckpt, best_fig, alias="best")
        self.run.finish()
        print(f"Best validation loss: {self.best_val_loss:.2e} (saved in {self.ckpt_dir})")
        print(f"Figures: {loss_fig} and {parity_fig}")
