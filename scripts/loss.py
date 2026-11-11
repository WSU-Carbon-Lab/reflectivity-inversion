"""Loss functions: how wrong is a prediction?

During training the optimizer adjusts the network's weights to make the loss smaller,
so the loss defines what "good" means. Three losses are available (``--loss``):

    mse      LabelMSELoss       compare the 7 numbers directly
    profile  DepthProfileLoss   compare the physical layer stacks they describe
    hybrid   HybridLoss         the sum of both (the default)

All three share the same label ranges and the same ``normalize`` / ``denormalize``
helpers, which is why DepthProfileLoss and HybridLoss inherit from LabelMSELoss.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

import physics

# The 7 film parameters, in the same order as the dataset columns and the network outputs.
LABEL_NAMES: tuple[str, ...] = (
    "film_thickness",   # [Å]
    "film_roughness",   # [Å]
    "oxide_thickness",  # [Å]
    "delta_xx",         # in-plane refractive index decrement      (n = 1 - delta + i beta)
    "delta_zz",         # out-of-plane refractive index decrement
    "beta_xx",          # in-plane absorption
    "beta_zz",          # out-of-plane absorption
)

# Range of each label in the training data. The network predicts every label scaled to
# [0, 1] inside this range, so a 20 Å film is 0.0 and an 800 Å film is 1.0. Scaling puts
# thicknesses (hundreds of Å) and optical constants (~0.001) on equal footing.
LABEL_MIN = (20.0, 0.0, 0.0, -5.0e-3, -5.0e-3, 0.0, 0.0)
LABEL_MAX = (800.0, 40.0, 15.0, 5.0e-3, 5.0e-3, 5.0e-3, 5.0e-3)

LOSS_CHOICES = ("profile", "mse", "hybrid")


class LabelMSELoss(nn.Module):
    """Mean squared error between predicted and true labels, both scaled to [0, 1].

    Also holds the label ranges and converts between physical and normalized units.
    The ranges are stored as buffers so ``.to(device)`` moves them to the GPU too.
    """

    label_min: Tensor
    label_max: Tensor
    weights: Tensor

    def __init__(self, weights: Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("label_min", torch.tensor(LABEL_MIN, dtype=torch.float32))
        self.register_buffer("label_max", torch.tensor(LABEL_MAX, dtype=torch.float32))
        # Optional per-label weights to make some labels count more; all 1 by default.
        if weights is None:
            weights = torch.ones(len(LABEL_NAMES))
        self.register_buffer("weights", weights.to(torch.float32))

    def normalize(self, physical: Tensor) -> Tensor:
        """Physical units -> [0, 1] scale."""
        return (physical - self.label_min) / (self.label_max - self.label_min)

    def denormalize(self, normalized: Tensor) -> Tensor:
        """[0, 1] scale -> physical units."""
        return normalized * (self.label_max - self.label_min) + self.label_min

    def per_sample_loss(self, pred_normalized: Tensor, target_physical: Tensor, q: Tensor | None = None) -> Tensor:
        """One loss value per sample, shape (batch,). ``q`` is unused here."""
        squared_error = (pred_normalized - self.normalize(target_physical)) ** 2
        return (squared_error * self.weights).mean(dim=1)

    def forward(self, pred_normalized: Tensor, target_physical: Tensor, q: Tensor | None = None) -> Tensor:
        """Average loss over the batch: the single number the optimizer minimizes."""
        return self.per_sample_loss(pred_normalized, target_physical, q).mean()

    @torch.no_grad()
    def physical_mae(self, pred_normalized: Tensor, target_physical: Tensor) -> Tensor:
        """Mean absolute error of each label in physical units, shape (7,). For reporting."""
        return (self.denormalize(pred_normalized) - target_physical).abs().mean(dim=0)


def smooth_positive(x: Tensor, scale: Tensor) -> Tensor:
    """A smooth version of max(x, 0), used to keep thicknesses and roughnesses >= 0.

    Computes 0.5 * (x + sqrt(x^2 + scale^2)). Far above 0 this is ~x, far below 0 it is
    ~0, and it bends smoothly over a width of about ``scale`` around 0. Unlike a hard
    clamp, a negative prediction still has a gradient pushing it back up, and values
    can still get very close to 0 Å (a bare oxide with no film).

    For x < 0 the same value is written as 0.5 * scale^2 / (sqrt(x^2 + scale^2) - x),
    which avoids subtracting two nearly equal numbers (a float32 precision trap).
    """
    r = torch.sqrt(x * x + scale * scale)
    return torch.where(x >= 0.0, 0.5 * (x + r), 0.5 * (scale * scale) / (r - x))


class DepthProfileLoss(LabelMSELoss):
    """Distance between the predicted and the true optical depth profiles.

    The 7 labels describe a layer stack. Plotting an optical constant (say delta_xx)
    against height h above the silicon surface gives a "depth profile":

        delta
          |                         ____ Si
          |        film     ___oxide|
          |   ____________ /
          |  /
          |_/ vacuum
          +-------------------------------- height h
          film top          h = oxide_t    h = 0

    Each interface is a smoothed step: an error-function (erf) ramp whose width is the
    interface roughness. Instead of comparing the 7 numbers, this loss measures how far
    apart the two profiles are:

        loss = sqrt( mean over 4 channels of  integral (pred(h) - true(h))^2 dh  / 800 Å )

    Two stacks that differ only in a parameter the curve barely depends on give a small
    loss; a misplaced film surface gives a large one.

    Both stacks are pinned at the silicon interface (h = 0), so a thickness error shows
    up as a misplaced film surface against a shared substrate.

    The integral over all h is done exactly with a closed-form formula (step 4 of
    ``per_sample_loss``), so there is no depth grid, and very thin, missing (0 Å) or
    very thick films all get exact, smooth gradients.
    """

    substrate_table: Tensor
    positive_scales: Tensor

    OPTICAL_SCALE = 5.0e-3  # divide optical constants by this so they are of order 1
    DEPTH_SCALE = 800.0     # divide the integral by this [Å] (the largest film thickness)

    def __init__(self) -> None:
        super().__init__()

        # The oxide and silicon optical constants depend on photon energy. Precompute
        # them at every 1 eV from 200 to 500 eV so the loss can look them up on the GPU.
        energies = np.arange(physics.ENERGY_MIN_EV, physics.ENERGY_MAX_EV + 1.0)
        table = physics.substrate_optical_table(energies).astype(np.float32)
        self.register_buffer("substrate_table", torch.from_numpy(table))

        # Bend widths [Å] of ``smooth_positive`` for film thickness, film roughness, oxide thickness.
        self.register_buffer("positive_scales", torch.tensor([1.0, 0.25, 0.25]))

    def denormalize(self, normalized: Tensor) -> Tensor:
        """Like LabelMSELoss.denormalize, but lengths are kept smoothly >= 0."""
        physical = super().denormalize(normalized)
        lengths = smooth_positive(physical[:, :3], self.positive_scales)
        return torch.cat((lengths, physical[:, 3:]), dim=1)

    def lookup_substrate(self, q: Tensor) -> Tensor:
        """Estimate each curve's photon energy from its q grid and look up the substrate.

        Same rule as ``physics.estimate_energy``: curves on the fixed-q grid use its fixed
        energy; all others are estimated from their largest q.
        Returns shape (batch, 4): (oxide_delta, oxide_beta, silicon_delta, silicon_beta).
        """
        energy = physics.energy_from_q_max(q[:, -1])
        energy = energy.clamp(physics.ENERGY_MIN_EV, physics.ENERGY_MAX_EV)
        fixed = physics.is_fixed_q_grid(q[:, 0], q[:, -1])
        energy = torch.where(fixed, torch.full_like(energy, physics.FIXED_Q_ENERGY_EV), energy)
        row = (energy - physics.ENERGY_MIN_EV).round().long()
        return self.substrate_table[row]

    def per_sample_loss(
        self,
        pred_normalized: Tensor,
        target_physical: Tensor,
        q: Tensor | None = None,
        substrate: Tensor | None = None,
    ) -> Tensor:
        """Profile distance for each sample, shape (batch,).

        Pass either ``q`` (the substrate is looked up from it) or ``substrate`` directly.
        """
        if substrate is None:
            if q is None:
                raise ValueError("DepthProfileLoss needs either q or substrate.")
            substrate = self.lookup_substrate(q)

        pred = self.denormalize(pred_normalized)
        true = target_physical

        # Step 1: where the interfaces are.
        # Measured as height above the oxide|Si interface, the three interfaces of a stack
        # are at h = [film_t + oxide_t, oxide_t, 0]. We use depth z = -h (so z increases
        # from vacuum into the substrate). The predicted and true stacks together give
        # six interface positions.
        zero = torch.zeros_like(pred[:, 0])

        def interface_depths(labels):
            film_t, oxide_t = labels[:, 0], labels[:, 2]
            return torch.stack((-(film_t + oxide_t), -oxide_t, zero), dim=-1)

        positions = torch.cat((interface_depths(pred), interface_depths(true)), dim=-1)  # (batch, 6)
        # Only differences between positions matter; centering them keeps float32 accurate.
        positions = positions - positions.mean(dim=-1, keepdim=True)

        # Step 2: how rough (wide) each interface is:
        # [film roughness, oxide roughness, fixed silicon roughness].
        si_sigma = torch.full_like(zero, physics.SILICON_ROUGHNESS)

        def interface_roughness(labels):
            return torch.stack((labels[:, 1], physics.oxide_roughness(labels[:, 2]), si_sigma), dim=-1)

        sigmas = torch.cat((interface_roughness(pred), interface_roughness(true)), dim=-1)  # (batch, 6)

        # Step 3: how much each optical constant jumps at each interface.
        # The four channels are (delta_xx, delta_zz, beta_xx, beta_zz). The substrate is
        # isotropic, so its xx and zz values are the same.
        oxide_delta, oxide_beta, si_delta, si_beta = substrate.unbind(dim=-1)
        oxide = torch.stack((oxide_delta, oxide_delta, oxide_beta, oxide_beta), dim=1) / self.OPTICAL_SCALE
        silicon = torch.stack((si_delta, si_delta, si_beta, si_beta), dim=1) / self.OPTICAL_SCALE
        film_pred = pred[:, 3:7] / self.OPTICAL_SCALE
        film_true = true[:, 3:7] / self.OPTICAL_SCALE

        # The difference profile pred(z) - true(z) is a sum of six erf steps: the three
        # steps of the predicted stack minus the three of the true stack.
        # vacuum -> film, film -> oxide, oxide -> silicon.
        jumps = torch.stack(
            (
                film_pred,
                oxide - film_pred,
                silicon - oxide,
                -film_true,
                -(oxide - film_true),
                -(silicon - oxide),
            ),
            dim=-1,
        )  # (batch, 4 channels, 6 steps)

        # Step 4: integrate (pred - true)^2 over all depths, exactly.
        # For a sum of erf steps with heights c_m, positions a_m and widths s_m, the
        # integral of the square is
        #     -0.5 * sum over pairs (m, n) of  c_m * c_n * psi(a_m - a_n, sqrt(s_m^2 + s_n^2))
        # with psi(d, s) = d * erf(d / (sqrt(2) s)) + sqrt(2 / pi) * s * exp(-d^2 / (2 s^2)).
        # (The jump heights sum to zero, since both stacks start in vacuum and end in Si,
        # which is what makes the integral finite.)
        d = positions.unsqueeze(-1) - positions.unsqueeze(-2)  # (batch, 6, 6)
        s = torch.sqrt(sigmas.unsqueeze(-1) ** 2 + sigmas.unsqueeze(-2) ** 2 + 1.0e-4)  # tiny floor avoids 0/0
        u = d / (math.sqrt(2.0) * s)
        psi = d * torch.erf(u) + math.sqrt(2.0 / math.pi) * s * torch.exp(-u * u)

        jump_pairs = jumps.unsqueeze(-1) * jumps.unsqueeze(-2)  # (batch, 4, 6, 6)
        integral = -0.5 * (jump_pairs * psi.unsqueeze(1)).sum(dim=(-2, -1))  # (batch, 4)
        integral = integral.clamp_min(0.0)  # it is >= 0 mathematically; remove rounding noise

        # Average the 4 channels, scale by the depth range, take the square root (L2 distance).
        return torch.sqrt(integral.mean(dim=1) / self.DEPTH_SCALE + 1.0e-8)


class HybridLoss(DepthProfileLoss):
    """Depth-profile distance + label MSE.

    The profile term rewards getting the physical stack right; the label term makes
    sure every label is learned, including ones the profile barely depends on (such as
    the oxide thickness).
    """

    def per_sample_loss(
        self,
        pred_normalized: Tensor,
        target_physical: Tensor,
        q: Tensor | None = None,
        substrate: Tensor | None = None,
    ) -> Tensor:
        profile = DepthProfileLoss.per_sample_loss(self, pred_normalized, target_physical, q, substrate)
        labels = LabelMSELoss.per_sample_loss(self, pred_normalized, target_physical)
        return profile + labels


def build_loss(name: str = "hybrid") -> LabelMSELoss:
    """Create a loss by its command-line name: "mse", "profile" or "hybrid"."""
    losses = {"mse": LabelMSELoss, "profile": DepthProfileLoss, "hybrid": HybridLoss}
    if name not in losses:
        raise ValueError(f"Unknown loss {name!r}; choose from {LOSS_CHOICES}")
    return losses[name]()
