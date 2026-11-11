"""Normalized MSE loss and physical error metrics for reflectivity inversion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Order of the 7 target columns in carbon-lab/reflectivity-1m.
LABEL_NAMES: tuple[str, ...] = (
    "film_thickness",
    "film_roughness",
    "oxide_thickness",
    "delta_xx",
    "delta_zz",
    "beta_xx",
    "beta_zz",
)


class ReflectivityLoss(nn.Module):
    """Min-max normalized MSE loss with physical-unit MAE reporting.

    Replaces ``normalize_labels``, ``mean_squared_error``, and
    ``y_absolute_error`` from ``code-old/training.py`` so parameters spanning
    hundreds of Angstroms down to ``1e-3`` optical constants contribute
    equally during training.
    """

    label_min: Tensor
    label_max: Tensor
    weights: Tensor

    def __init__(self, weights: Tensor | None = None) -> None:
        super().__init__()

        # Lower physical bounds for each of the 7 labels.
        self.register_buffer(
            "label_min",
            torch.tensor(
                [
                    20.0,     # film_thickness [Angstrom]
                    0.0,      # film_roughness [Angstrom]
                    0.0,      # oxide_thickness [Angstrom]
                    -5.0e-3,  # delta_xx
                    -5.0e-3,  # delta_zz
                    0.0,      # beta_xx
                    0.0,      # beta_zz
                ],
                dtype=torch.float32,
            ),
        )

        # Upper physical bounds for each of the 7 labels.
        self.register_buffer(
            "label_max",
            torch.tensor(
                [
                    800.0,   # film_thickness [Angstrom]
                    40.0,    # film_roughness [Angstrom]
                    15.0,    # oxide_thickness [Angstrom]
                    5.0e-3,  # delta_xx
                    5.0e-3,  # delta_zz
                    5.0e-3,  # beta_xx
                    5.0e-3,  # beta_zz
                ],
                dtype=torch.float32,
            ),
        )

        # Per-label loss weights (defaults to equal weight 1.0 for all 7 labels).
        default_weights = torch.ones(7, dtype=torch.float32)
        self.register_buffer(
            "weights",
            # If we pass in custom weights, use them; otherwise use default weights.
            default_weights if weights is None else weights.to(torch.float32),
        )

    def normalize(self, target_physical: Tensor) -> Tensor:
        """Scale physical labels into [0, 1] using their min and max bounds."""
        return (target_physical - self.label_min) / (
            self.label_max - self.label_min
        )

    def denormalize(self, pred_normalized: Tensor) -> Tensor:
        """Convert [0, 1] model predictions back into physical units."""
        return (
            pred_normalized * (self.label_max - self.label_min) + self.label_min
        )

    def forward(self, pred_normalized: Tensor, target_physical: Tensor) -> Tensor:
        """Compute mean squared error in normalized [0, 1] label space."""
        target_normalized = self.normalize(target_physical)
        squared_error = (pred_normalized - target_normalized).pow(2)
        return (squared_error * self.weights).mean()

    @torch.no_grad()
    def physical_mae(
        self, pred_normalized: Tensor, target_physical: Tensor
    ) -> Tensor:
        """Mean absolute error for each of the 7 labels in physical units."""
        pred_physical = self.denormalize(pred_normalized)
        return (pred_physical - target_physical).abs().mean(dim=0)
