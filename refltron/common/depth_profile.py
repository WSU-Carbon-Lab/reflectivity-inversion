"""Depth profiles: the optical constants of a stack as a function of height.

Shared by every model's metrics and figures. A stack is described by the 7 physical
labels (see refltron.common.loss.LABEL_NAMES) on top of the fixed SiO2 / Si substrate.
"""

import math

import numpy as np
import torch

from refltron.common import physics


def estimate_substrate_sld(q):
    """Substrate optical constants (oxide_delta, oxide_beta, si_delta, si_beta) for one curve's q grid."""
    energy_ev = physics.estimate_energy(float(q[0]), float(q[-1]))
    return physics.substrate_optical_constants(energy_ev)


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
