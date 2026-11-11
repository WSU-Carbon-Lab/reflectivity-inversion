"""Bookkeeping during training: W&B logging, checkpoint files and snapshot figures.

``MetricStore`` is created once by train.py and called once per epoch. Each epoch it
    - logs the losses and errors to Weights & Biases (W&B),
    - saves ``best.pt`` whenever the validation loss improves, plus a numbered
      checkpoint every few epochs,
    - draws a snapshot figure next to each checkpoint. Its four columns are one training
      row, one validation row and two out-of-distribution (OOD) test curves the network
      never saw: a bare oxide with no film, and a 2000 Å film (thicker than any in training).
"""

import math
from pathlib import Path

import matplotlib

# Save figures directly to PNG files without opening an interactive window.
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from refloxide.model import FreeUniTensor, MaterialSLD, ReflectModel
import torch
import wandb

import physics
from loss import LABEL_NAMES, DepthProfileLoss

# Readable tick and label sizes for the large snapshot figures.
plt.rcParams.update({"xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.labelsize": 12, "axes.titlesize": 12.5})

# W&B project: https://forge.coreweave.com/wandb/als-rixs/reflectivity-inversion
# (Training runs 100% on local hardware; W&B is only used to log metrics & checkpoints.)
WANDB_BASE_URL = "https://api.forge.coreweave.com"
WANDB_ENTITY = "als-rixs"
WANDB_PROJECT = "reflectivity-inversion"

# Snapshot rows are picked with a film thickness close to this [Å].
SNAPSHOT_THICKNESS = 200.0

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
    """Substrate optical constants (oxide_delta, oxide_beta, si_delta, si_beta) for a curve's max q."""
    energy_ev = physics.energy_from_q_max(float(q_max))
    energy_ev = min(max(energy_ev, physics.ENERGY_MIN_EV), physics.ENERGY_MAX_EV)
    return physics.substrate_optical_constants(energy_ev)


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
    film_t, film_sigma, ox_t_thick = 2000.0, 12.0, 8.0
    d_xx, d_zz, b_xx, b_zz = 2.2e-3, 1.1e-3, 0.8e-3, 0.3e-3

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


def compute_depth_profile(labels, h, substrate_sld):
    """Erf-smoothed (delta_xx, delta_zz, beta_xx, beta_zz) vs height h [Å] above silicon.

    The oxide|Si interface is pinned at ``h = 0`` (matching ``DepthProfileLoss``), the
    film|oxide interface at ``h = oxide_t`` and the vacuum|film surface at
    ``h = film_t + oxide_t``. Returns interface heights ordered vacuum -> Si.
    """
    ox_delta, ox_beta, si_delta, si_beta = substrate_sld

    film_t = max(float(labels[0]), 0.0)
    film_sigma = max(float(labels[1]), 0.0)
    oxide_t = max(float(labels[2]), 0.0)
    delta_xx = float(labels[3])
    delta_zz = float(labels[4])
    beta_xx = float(labels[5])
    beta_zz = float(labels[6])

    heights = [film_t + oxide_t, oxide_t, 0.0]
    roughnesses = [film_sigma, physics.oxide_roughness(oxide_t), physics.SILICON_ROUGHNESS]

    # Optical constants in each slab: [vacuum, film, oxide, silicon].
    slab_values = {
        "delta_xx": [0.0, delta_xx, ox_delta, si_delta],
        "delta_zz": [0.0, delta_zz, ox_delta, si_delta],
        "beta_xx": [0.0, beta_xx, ox_beta, si_beta],
        "beta_zz": [0.0, beta_zz, ox_beta, si_beta],
    }

    h_tensor = torch.from_numpy(np.asarray(h, dtype=np.float64))
    curves = {}
    for name, values in slab_values.items():
        profile = torch.full_like(h_tensor, values[0])
        for i in range(3):
            b = heights[i]
            sigma = roughnesses[i]
            # Stepping *down* in height crosses from slab i into slab i+1.
            if sigma > 0.0:
                step = 0.5 * (1.0 + torch.erf((b - h_tensor) / (math.sqrt(2.0) * sigma)))
            else:
                step = (h_tensor <= b).to(h_tensor.dtype)
            profile = profile + (values[i + 1] - values[i]) * step
        curves[name] = profile.numpy()

    return heights, roughnesses, curves


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
    """Fill one column (Reflectivity, Depth Profile, Label Bars) like single_sample.png."""
    ax_refl, ax_depth, ax_bar = axes_col

    # 1. Top axis: s and p polarized reflectivity vs q.
    ms = 2.6 if bare_oxide is False and true_phys[0] > 1000.0 else 3.6
    ax_refl.plot(
        q, r_s, label="s polarization", color="tab:blue",
        linestyle="-" if ms < 3.0 else "", linewidth=0.8,
        marker="o", markersize=ms, markerfacecolor="none",
    )
    ax_refl.plot(
        q, r_p, label="p polarization", color="tab:red",
        linestyle="-" if ms < 3.0 else "", linewidth=0.8,
        marker="o", markersize=ms, markerfacecolor="none",
    )
    ax_refl.set_yscale("log")
    ax_refl.set_xlabel(r"$q$ [$\mathrm{\AA}^{-1}$]")
    ax_refl.set_ylabel("reflectivity")
    ax_refl.set_title(f"{title_prefix} — s & p reflectivity")
    ax_refl.grid(True, which="both", alpha=0.3)
    ax_refl.legend(loc="upper right", fontsize=10.5)

    # 2. Middle axis: optical-constant height map (Truth vs Network), both pinned at
    #    the silicon interface (h = 0, right edge) so film surfaces can be compared.
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
    regions = ("vacuum", "film", "oxide", "Si")
    shades = ("0.97", "0.92", "0.84", "0.76")
    xaxis_transform = ax_depth.get_xaxis_transform()
    h_span = h[-1] - h[0]

    for i, (region, shade) in enumerate(zip(regions, shades)):
        ax_depth.axvspan(edges[i + 1], edges[i], color=shade, zorder=0)
        slab_width = edges[i] - edges[i + 1]
        if slab_width < 0.03 * h_span:
            continue
        rotate = 90 if slab_width < 0.075 * h_span else 0
        ax_depth.text(
            0.5 * (edges[i] + edges[i + 1]), 0.96, region,
            ha="center", va="top", rotation=rotate,
            fontsize=10.5, color="0.25", transform=xaxis_transform,
        )

    for b in true_heights:
        ax_depth.axvline(b, color="0.55", linewidth=0.8)
    # Predicted film surface and film|oxide interface (dashed) vs the shared Si reference.
    ax_depth.axvline(pred_heights[0], color="0.15", linestyle="--", linewidth=1.0)
    ax_depth.axvline(pred_heights[1], color="0.15", linestyle=":", linewidth=0.9)
    ax_depth.axhline(0.0, color="0.6", linewidth=0.6, linestyle=":")

    # Plot Truth as thick translucent ribbons and Network as crisp dashed curves.
    for key, (color, _) in COMPONENT_COLORS.items():
        ax_depth.plot(h, true_curves[key], color=color, linestyle="-", linewidth=3.2, alpha=0.40)
        ax_depth.plot(h, pred_curves[key], color=color, linestyle="--", linewidth=1.6, alpha=1.0)

    legend_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=tex)
        for color, tex in COMPONENT_COLORS.values()
    ] + [
        Line2D([0], [0], color="0.35", lw=3.2, alpha=0.45, linestyle="-", label="truth"),
        Line2D([0], [0], color="0.20", lw=1.6, linestyle="--", label="network"),
    ]
    # Height decreases left -> right so the silicon substrate is always at the right edge.
    ax_depth.set_xlim(h[-1], h[0])
    ax_depth.set_xlabel(r"height above Si interface [$\mathrm{\AA}$]  (vacuum $\leftarrow$ $\rightarrow$ substrate)")
    ax_depth.set_ylabel(r"$\delta$, $\beta$")
    ax_depth.set_title(
        f"film top above Si: truth {true_heights[0]:.0f} Å · net {pred_heights[0]:.0f} Å · L2 {profile_l2:.2e}",
        fontsize=12.5,
    )
    ax_depth.margins(y=0.14)
    ax_depth.grid(True, which="both", alpha=0.3)
    ax_depth.legend(handles=legend_handles, loc="lower left", fontsize=10, ncol=3)

    # 3. Bottom axis: side-by-side normalized [0, 1] label comparison bar chart.
    # Dotted lines at 0.0 and 1.0 mark the training-set bounds; OOD bars cross outside!
    x = np.arange(len(LABEL_NAMES))
    width = 0.36
    ax_bar.axhline(0.0, color="0.65", linewidth=0.7, linestyle=":")
    ax_bar.axhline(1.0, color="0.65", linewidth=0.7, linestyle=":")

    if bare_oxide:
        # At 0 Å film thickness, only film_thickness (idx 0) and oxide_thickness (idx 2)
        # are physically defined; fade the 5 unused film optical/roughness bars.
        active = [0, 2]
        unused = [1, 3, 4, 5, 6]
        ax_bar.bar(x[active] - width / 2, np.array(true_norm)[active], width=width, color="0.55", label="truth")
        ax_bar.bar(x[active] + width / 2, np.array(pred_norm)[active], width=width, color="tab:blue", label="network")
        ax_bar.bar(x[unused] + width / 2, np.array(pred_norm)[unused], width=width, color="tab:blue", alpha=0.25, hatch="//")
        bar_title = f"normalized labels (hatched = film n/a at 0 Å; ox truth {true_phys[2]:.1f} vs {pred_phys[2]:.1f} Å)"
    else:
        ax_bar.bar(x - width / 2, true_norm, width=width, color="0.55", label="truth")
        ax_bar.bar(x + width / 2, pred_norm, width=width, color="tab:blue", label="network")
        bar_title = "normalized [0, 1] labels (dotted [0, 1] = training range)"

    ax_bar.set_xticks(x, LABEL_NAMES, rotation=22, ha="right")
    ax_bar.set_ylabel("normalized [0, 1]")
    ax_bar.set_title(bar_title)
    y_lo = min(-0.12, float(min(min(true_norm), min(pred_norm))) - 0.08)
    y_hi = max(1.15, float(max(max(true_norm), max(pred_norm))) + 0.12)
    ax_bar.set_ylim(y_lo, y_hi)
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(loc="upper right", fontsize=10.5, ncol=2)


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

        tr_s, tr_p, tr_q, tr_target = dataset[self.train_sample_idx]
        val_s, val_p, val_q, val_target = dataset[self.val_sample_idx]
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
            estimate_substrate_sld(tr_q[-1].item()),
            estimate_substrate_sld(val_q[-1].item()),
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
        self.run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode="offline" if args.local else "online",
            config=self.config,
            name=args.run_name,
            settings=wandb.Settings(base_url=WANDB_BASE_URL),
        )

        # A resumed run keeps writing into the original run's folder.
        if resume is not None:
            self.ckpt_dir = Path(args.resume).parent
        else:
            run_id = self.run.id if self.run.id else "local-run"
            self.ckpt_dir = Path(args.checkpoint_dir) / run_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        if resume is not None:
            best_path = self.ckpt_dir / "best.pt"
            if best_path.is_file():
                best = torch.load(best_path, map_location="cpu", weights_only=False)
                self.best_val_loss = best["val_loss"]
        self.saved_checkpoints = []

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
        for label_name, mae_value in zip(LABEL_NAMES, val_mae.tolist()):
            metrics[f"val/mae_{label_name}"] = mae_value
        self.run.log(metrics, step=epoch)

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
                old_ckpt, old_fig = self.saved_checkpoints.pop(0)
                old_ckpt.unlink(missing_ok=True)
                old_fig.unlink(missing_ok=True)

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
        target_norm, pred_norm, pred_phys, profile_losses,
    ):
        """Save 4 side-by-side columns: Training, Validation, OOD Bare Oxide, OOD 2000 Å."""
        fig, axes = plt.subplots(
            3, 4,
            figsize=(26.0, 11.5),
            gridspec_kw={"height_ratios": [1.0, 1.15, 0.78]},
            layout="constrained",
        )

        s_np = self.sample_s.cpu().numpy()
        p_np = self.sample_p.cpu().numpy()
        q_np = self.sample_q.cpu().numpy()
        t_phys = self.sample_targets.cpu().tolist()
        p_phys = pred_phys.cpu().tolist()
        t_norm = target_norm.cpu().tolist()
        p_norm = pred_norm.cpu().tolist()

        for col in range(4):
            plot_sample_column(
                axes[:, col],
                self.column_titles[col],
                q_np[col], s_np[col], p_np[col],
                t_phys[col], p_phys[col], t_norm[col], p_norm[col],
                profile_losses[col],
                self.column_slds[col],
                bare_oxide=self.column_bare_flags[col],
            )

        fig.suptitle(
            f"Epoch {epoch}  |  Loss ({self.loss_name}): Train {train_loss:.2e}  ·  Val {val_loss:.2e}  |  "
            f"OOD Profile L2: Bare Oxide {profile_losses[2]:.2e}  ·  2000 Å Film {profile_losses[3]:.2e}",
            fontsize=15.5,
        )
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def finish(self):
        """Upload the final best checkpoint to W&B (when online) and close the run."""
        best_ckpt = self.ckpt_dir / "best.pt"
        best_fig = self.ckpt_dir / "best.png"
        if not self.local and best_ckpt.is_file() and best_fig.is_file():
            self._upload_artifact(best_ckpt, best_fig, alias="best")
        self.run.finish()
        print(f"Best validation loss: {self.best_val_loss:.2e} (saved in {self.ckpt_dir})")
