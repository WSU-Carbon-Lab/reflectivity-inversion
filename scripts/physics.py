"""Physical constants and the substrate model shared by the loss, metrics and plots.

Every sample in the dataset is a stack of four layers, from top to bottom:

    vacuum | organic film | native SiO2 oxide | Si substrate

The film is what the network has to describe (its thickness, roughness and optical
constants). The oxide and silicon are fixed materials whose optical constants only
depend on the photon energy, so we look them up here instead of predicting them.
"""

from __future__ import annotations

import math

import numpy as np
from refloxide.model import MaterialSLD

# Photon energy E [eV] times wavelength [Å] is a constant: E * lambda = h * c.
HC_EV_ANGSTROM = 12398.419843320026

# The simulated curves use grazing angles up to about 65 degrees. The largest angle of
# a 256-point grid is typically ~64.75 deg, which lets us recover the photon energy
# from the largest q value of a curve (see ``energy_from_q_max``).
THETA_MAX_DEG = 64.75

# The fixed-q dataset (branch "fixed-q" of carbon-lab/reflectivity-1m) measures every
# curve at one energy on one evenly spaced q grid, so its largest q says nothing about
# the energy. We recognize that grid by its first and last q instead (see
# ``estimate_energy``). These must match ENERGY_EV and Q_RANGE in generate_reflectivity.py.
FIXED_Q_ENERGY_EV = 250.0
FIXED_Q_RANGE = (0.0044, 0.184)  # first and last q [1/Å]
FIXED_Q_TOLERANCE = 1e-6         # [1/Å]; float32 storage moves q by far less than this

# Photon energies covered by the training data [eV].
ENERGY_MIN_EV = 200.0
ENERGY_MAX_EV = 500.0

# Substrate materials: chemical formula and mass density [g/cm^3].
OXIDE_FORMULA, OXIDE_DENSITY = "SiO2", 2.2
SILICON_FORMULA, SILICON_DENSITY = "Si", 2.33

# Roughness of the oxide | silicon interface [Å]. The oxide's own (top) roughness is
# tied to its thickness: sigma_oxide = oxide_thickness / (2 pi).
SILICON_ROUGHNESS = 1.2


def oxide_roughness(oxide_thickness):
    """Roughness of the film | oxide interface [Å] for a given oxide thickness [Å]."""
    return oxide_thickness / (2.0 * math.pi)


def energy_from_q_max(q_max):
    """Estimate the photon energy [eV] from the largest q [1/Å] of a curve.

    q = 4 pi / lambda * sin(theta) and lambda = hc / E, so E = q * hc / (4 pi sin(theta)).
    Works on plain floats and on torch tensors.
    """
    return q_max * HC_EV_ANGSTROM / (4.0 * math.pi * math.sin(math.radians(THETA_MAX_DEG)))


def is_fixed_q_grid(q_min, q_max):
    """True if a curve's first and last q are those of the fixed-q grid.

    Works on plain floats and, element by element, on torch tensors. A random-angle
    curve essentially never matches both ends to within FIXED_Q_TOLERANCE.
    """
    first_matches = abs(q_min - FIXED_Q_RANGE[0]) < FIXED_Q_TOLERANCE
    last_matches = abs(q_max - FIXED_Q_RANGE[1]) < FIXED_Q_TOLERANCE
    return first_matches & last_matches


def estimate_energy(q_min, q_max):
    """Photon energy [eV] of one curve from its first and last q [1/Å] (plain floats).

    Fixed-q curves get FIXED_Q_ENERGY_EV. Every other curve uses ``energy_from_q_max``,
    clamped to the training range [ENERGY_MIN_EV, ENERGY_MAX_EV].
    """
    if is_fixed_q_grid(q_min, q_max):
        return FIXED_Q_ENERGY_EV
    energy_ev = energy_from_q_max(q_max)
    return min(max(energy_ev, ENERGY_MIN_EV), ENERGY_MAX_EV)


def substrate_optical_constants(energy_ev):
    """Return (oxide_delta, oxide_beta, silicon_delta, silicon_beta) at one energy [eV].

    The refractive index of each material is n = 1 - delta + i * beta.
    """
    energies = np.array([float(energy_ev)])
    oxide = MaterialSLD(OXIDE_FORMULA, OXIDE_DENSITY).tensor_at_many(energies)[0, 0, 0]
    silicon = MaterialSLD(SILICON_FORMULA, SILICON_DENSITY).tensor_at_many(energies)[0, 0, 0]
    return float(oxide.real), float(oxide.imag), float(silicon.real), float(silicon.imag)


def substrate_optical_table(energies_ev):
    """Optical constants for many energies at once: array of shape (n_energies, 4).

    Columns are (oxide_delta, oxide_beta, silicon_delta, silicon_beta), as in
    ``substrate_optical_constants``.
    """
    energies_ev = np.asarray(energies_ev, dtype=float)
    oxide = MaterialSLD(OXIDE_FORMULA, OXIDE_DENSITY).tensor_at_many(energies_ev)[:, 0, 0]
    silicon = MaterialSLD(SILICON_FORMULA, SILICON_DENSITY).tensor_at_many(energies_ev)[:, 0, 0]
    return np.stack((oxide.real, oxide.imag, silicon.real, silicon.imag), axis=1)
