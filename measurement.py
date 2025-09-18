# qgledger/measurement.py
"""
qgledger.measurement
====================
First-write (measurement) thresholds and lab-facing diagnostics for the
IR Quantum Gravity (ledger/NPR) framework.

What this module does
---------------------
- Encodes the minimal **energy** required to perform a single reversible write
  on the ledger (four classical bits), using Clausius with entropy in *nats*.
- Converts apparatus settings (beam power/area, photon frequency) into **flux
  densities** and checks whether a write can occur within a given gate time.
- Provides clean mappings between acceleration ↔ surface gravity κ, and Unruh
  temperature, to parametrize local “effective” temperatures in lab settings.

Key relations
-------------
Entropy bookkeeping in nats implies the Clausius energy for one minimal write:
    ΔQ_write = T · (k_B ΔS_nats) = (ħ κ / 2π) · (4 ln 2),
where T = ħ κ / (2π k_B) is the local Unruh/Hawking temperature and ΔS_nats = 4 ln 2.
Per *area*, the minimal reversible write consumes 4 A_bit of ledger surface, so
    (ΔQ/A)_min = ΔQ_write / (4 A_bit).

Given a flux density F [W/m^2] on the (would-be) write patch, the *shortest*
time that can supply the energy is
    t_write = (ΔQ/A)_min / F.

Between writes, NPR forbids longitudinal transport across the ledger
(T_{++}=T_{--}=0). During a write, the Clausius balance is the operative rule.

Units
-----
- SI everywhere (W, m, s, K).
- Entropy is in nats (dimensionless); thermodynamic entropy is k_B × nats.

Quick start
-----------
>>> from qgledger.measurement import energy_per_write_total, energy_per_area_min
>>> from qgledger.measurement import intensity_from_power_area, time_to_write
>>> kappa = 1e5  # 1/s
>>> E = energy_per_write_total(kappa)
>>> E_area = energy_per_area_min(kappa)
>>> F = intensity_from_power_area(power_w=1e-3, area_m2=1e-6)
>>> tmin = time_to_write(F, kappa)
"""

from __future__ import annotations

import math
from typing import Tuple

from .constants import hbar, k_B, c, A_bit, LN2

# -----------------------------------------------------------------------------
# κ, Unruh temperature, and conversions
# -----------------------------------------------------------------------------

def kappa_from_acceleration(a_mps2: float) -> float:
    """
    Map a proper acceleration a [m/s^2] to a surface-gravity scale κ [1/s]
    via κ = a / c.

    Parameters
    ----------
    a_mps2 : float
        Proper acceleration [m/s^2].

    Returns
    -------
    float
        κ in [1/s].
    """
    if a_mps2 < 0.0:
        raise ValueError("Acceleration should be non-negative.")
    return a_mps2 / c


def unruh_temperature_from_kappa(kappa: float) -> float:
    """
    Unruh temperature corresponding to κ:

        T = ħ κ / (2π k_B)  [K].

    Parameters
    ----------
    kappa : float
        κ in [1/s].

    Returns
    -------
    float
        Temperature [K].
    """
    return (hbar * kappa) / (2.0 * math.pi * k_B)


# -----------------------------------------------------------------------------
# Minimal energy required for a single reversible write
# -----------------------------------------------------------------------------

def energy_per_write_total(kappa: float) -> float:
    """
    Total energy required for the minimal reversible record (four classical bits):

        ΔQ_write = (ħ κ / 2π) · (4 ln 2)   [J].

    Parameters
    ----------
    kappa : float
        κ in [1/s].

    Returns
    -------
    float
        Energy in joules.
    """
    return (hbar * kappa / (2.0 * math.pi)) * (4.0 * LN2)


def energy_per_area_min(kappa: float) -> float:
    """
    Minimal **energy per unit area** required to write, assuming the write
    occupies the minimal ledger area ΔA = 4 A_bit:

        (ΔQ/A)_min = ΔQ_write / (4 A_bit)
                   = (ħ κ / 2π) · (ln 2) / A_bit    [J/m^2].

    Parameters
    ----------
    kappa : float
        κ in [1/s].

    Returns
    -------
    float
        Energy per area [J/m^2].
    """
    return (hbar * kappa / (2.0 * math.pi)) * (LN2 / A_bit)


# -----------------------------------------------------------------------------
# Apparatus-facing helpers: intensities, photon flux, time-to-write
# -----------------------------------------------------------------------------

def intensity_from_power_area(power_w: float, area_m2: float) -> float:
    """
    Intensity (power density) from a beam or source footprint:

        F = P / A  [W/m^2].

    Parameters
    ----------
    power_w : float
        Power [W].
    area_m2 : float
        Illuminated area [m^2].

    Returns
    -------
    float
        Intensity [W/m^2].
    """
    if power_w < 0.0 or area_m2 <= 0.0:
        raise ValueError("Require power >= 0 and area > 0.")
    return power_w / area_m2


def beam_area_from_waist(w0_m: float) -> float:
    """
    Gaussian beam 1/e^2 **spot area** A ≈ π w0^2.

    Parameters
    ----------
    w0_m : float
        Beam waist radius [m].

    Returns
    -------
    float
        Area [m^2].
    """
    if w0_m <= 0.0:
        raise ValueError("w0 must be positive.")
    return math.pi * (w0_m ** 2)


def photon_energy(hz: float) -> float:
    """
    Photon energy E = h ν = 2π ħ ν  [J].

    Parameters
    ----------
    hz : float
        Frequency ν [Hz].

    Returns
    -------
    float
        Energy [J].
    """
    return (2.0 * math.pi) * hbar * hz


def photon_flux(power_w: float, hz: float) -> float:
    """
    Photon rate \\.N = P / (h ν) [1/s].

    Parameters
    ----------
    power_w : float
        Power [W].
    hz : float
        Frequency [Hz].

    Returns
    -------
    float
        Photons per second.
    """
    E = photon_energy(hz)
    if E <= 0.0:
        raise ValueError("Photon energy must be positive.")
    return power_w / E


def time_to_write(F_w_per_m2: float, kappa: float) -> float:
    """
    Minimal time to supply the energy **per unit area** for a write:

        t_write = (ΔQ/A)_min / F.

    Parameters
    ----------
    F_w_per_m2 : float
        Intensity (flux density) [W/m^2].
    kappa : float
        κ in [1/s].

    Returns
    -------
    float
        Time [s].
    """
    if F_w_per_m2 <= 0.0:
        raise ValueError("Intensity must be positive.")
    return energy_per_area_min(kappa) / F_w_per_m2


def can_write_within_gate(power_w: float, area_m2: float, kappa: float, gate_time_s: float) -> Tuple[bool, float]:
    """
    Decision helper: given (P, A, κ, τ_gate), decide if a write is feasible.

    Parameters
    ----------
    power_w : float
        Power delivered to the patch [W].
    area_m2 : float
        Patch/spot area [m^2].
    kappa : float
        κ [1/s].
    gate_time_s : float
        Available interaction time [s].

    Returns
    -------
    (ok, t_needed) : (bool, float)
        ok = True if t_write ≤ gate_time_s,
        t_needed = minimal time [s] required at the provided intensity.
    """
    F = intensity_from_power_area(power_w, area_m2)
    t_needed = time_to_write(F, kappa)
    return (t_needed <= gate_time_s), t_needed
