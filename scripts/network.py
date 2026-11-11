"""Naive 3-channel mixing MLP for polarized reflectivity inversion."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReflectivityNet(nn.Module):
    """Fresnel R*q^4 transform + pointwise 3-channel mixer + dense MLP.

    Expands the 7-layer dense network from ``code-old/training.py`` to accept
    polarized reflectivity channels ``(s, p, q)`` of length ``n_q``.
    """

    def __init__(
        self,
        n_q: int = 256,
        n_labels: int = 7,
        mix_channels: int = 8,
        log_rq4: bool = True,
        eps: float = 1e-30,
    ) -> None:
        super().__init__()
        self.log_rq4 = log_rq4
        self.eps = eps

        # Mix the 3 input channels (s*q^4, p*q^4, q) linearly at each q point:
        # (batch, 3, n_q) -> (batch, mix_channels, n_q) -> (batch, mix_channels * n_q)
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(3, mix_channels, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        # Multi-layer perceptron (MLP) mapping flattened features to film labels.
        # Each Linear layer mixes features across all hidden units; ReLU adds non-linearity.
        self.mlp = nn.Sequential(
            nn.Linear(n_q * mix_channels, 400),
            nn.ReLU(),
            nn.Linear(400, 800),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            # Final linear layer maps to the output labels (no ReLU so negative
            # values like delta_xx and delta_zz are not clipped).
            nn.Linear(100, n_labels),
        )

    def forward(self, s: Tensor, p: Tensor, q: Tensor) -> Tensor:
        # Multiply s and p by q^4 to flatten the steep Fresnel falloff.
        q4 = q.pow(4)
        s_q4 = s * q4
        p_q4 = p * q4
        if self.log_rq4:
            s_q4 = torch.log10(s_q4.clamp_min(self.eps))
            p_q4 = torch.log10(p_q4.clamp_min(self.eps))

        # Stack into 3 channels (batch, 3, n_q), mix channels, and run the MLP.
        x = torch.stack((s_q4, p_q4, q), dim=1)
        mixed = self.channel_mixer(x)
        return self.mlp(mixed)
