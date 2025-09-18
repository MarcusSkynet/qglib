# qgledger/npr.py
"""
qgledger.npr
============
Null-Pair Removal (NPR) rules, ledger write accounting, and simple diagnostics
for the IR Quantum Gravity (ledger/NPR) framework.

This module encodes the *boundary mechanics* at the ledger worldvolume 𝓛:

• **Placement (geometry only):** the ledger is the invariant level set
  𝓘(x) = 𝓘_* where 𝓘 = 𝓚 r_A^4 and 𝓘_* = 12/alpha_H = 48 ln 2 (dimensionless).

• **NPR between writes:** no longitudinal transport across 𝓛:
  T_{++} = T_{--} = 0 (pure phase response).

• **Write thermodynamics (minimal reversible record):**
  ΔS = 4 ln 2 (nats), ΔA = −4 A_bit, T = κ/(2π), and the Clausius balance at the
  instant of writing: T_{++} + T_{--} = (κ/2π) · (4 ln 2).

All functions use **SI units** unless stated otherwise. Entropy is in **nats**.

Quick start
-----------
>>> from qgledger.npr import write_cost, temperature_from_surface_gravity, clausius_required_flux
>>> dS, dA = write_cost()
>>> T = temperature_from_surface_gravity(kappa=1e4)  # [K] if κ in 1/s
>>> F_needed = clausius_required_flux(kappa=1e4)     # energy flux density sum [W/m^2]

Notes on scope
--------------
These routines provide bookkeeping and validation. They do not solve field equations.
Use :mod:`qgledger.geom` to evaluate 𝓘 and place the ledger; use :mod:`qgledger.tortoise`
and :mod:`qgledger.echoes` for detector-facing echo timing and spectra.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .constants import (
    LN2, k_B, c, alpha_H, I_star, A_bit, RTOL, ATOL
)

# -----------------------------------------------------------------------------
# Placement / geometry-facing helpers (dimensionless invariant)
# -----------------------------------------------------------------------------

def placement_residual(I_value: float) -> float:
    """
    Residual of the ledger placement condition: r = 𝓘 − 𝓘_*.

    Parameters
    ----------
    I_value : float
        Dimensionless invariant 𝓘 at a point on the candidate worldvolume.

    Returns
    -------
    float
        Residual r = I_value - I_star. The ledger satisfies r = 0.
    """
    return float(I_value - I_star)


def on_ledger(I_value: float, *, rtol: float = RTOL, atol: float = ATOL) -> bool:
    """
    Test whether 𝓘 ≈ 𝓘_* within tolerances.

    Parameters
    ----------
    I_value : float
        Dimensionless invariant 𝓘.
    rtol, atol : float
        Relative and absolute tolerances.

    Returns
    -------
    bool
        True if |𝓘 − 𝓘_*| ≤ atol + rtol · |𝓘_*|.
    """
    return abs(I_value - I_star) <= (atol + rtol * abs(I_star))


# -----------------------------------------------------------------------------
# NPR constraints and write accounting
# -----------------------------------------------------------------------------

def npr_satisfied(flux_pp: float, flux_mm: float, *, writing: bool = False, tol: float = 0.0) -> bool:
    """
    Check NPR longitudinal-transport constraints.

    Parameters
    ----------
    flux_pp : float
        T_{++} (energy flux density along + null normal) [W/m^2].
    flux_mm : float
        T_{--} (energy flux density along − null normal) [W/m^2].
    writing : bool, default False
        If False: require T_{++} = T_{--} = 0 (no transport). If True: do not enforce.
    tol : float, default 0.0
        Allowed tolerance for the “zero” check.

    Returns
    -------
    bool
        True if NPR constraints are satisfied given `writing`.
    """
    if writing:
        # During a write, NPR transport prohibition is lifted; thermodynamics applies instead.
        return True
    return (abs(flux_pp) <= tol) and (abs(flux_mm) <= tol)


def write_cost() -> Tuple[float, float]:
    """
    Minimal reversible write cost at the ledger (four classical bits).

    Returns
    -------
    (ΔS, ΔA) : tuple of floats
        ΔS = 4 ln 2 [nats],  ΔA = −4 A_bit [m^2].

    Notes
    -----
    The 4-bit register is (b_r, X, Y, Z): one branch bit and three sign bits
    capturing principal-shear and twist orientation on the 2D screen.
    """
    dS = 4.0 * LN2
    dA = -4.0 * A_bit
    return dS, dA


def temperature_from_surface_gravity(kappa: float) -> float:
    """
    Local Unruh/Hawking temperature from surface gravity.

    Parameters
    ----------
    kappa : float
        Surface gravity κ in [1/s] (i.e., acceleration divided by c).

    Returns
    -------
    float
        Temperature T = κ / (2π) · (ħ/k_B) in kelvin.

    Units caveat
    ------------
    In SI, κ has units of m/s^2 divided by c to give 1/s. If you provide κ in 1/s
    (as is common when c=1), this function yields T in kelvin using k_B and ħ implicitly
    (through κ’s definition). For strict SI from scratch, one can write:
        T = ħ κ / (2π k_B).
    Here we take κ as 1/s so the constant factor is (ħ/k_B) multiplied by κ/(2π).
    """
    # Using ħ/k_B implicitly via units of κ; more explicit form would multiply by ħ/k_B.
    # Here we keep the customary T = (ħ κ)/(2π k_B). Since ħ/k_B is not imported here,
    # we treat κ in 1/s and return numerical kelvin using constants:
    #   ħ/k_B ≈ 7.63823e-12 K·s
    hbar_over_kB = 1.054_571_817e-34 / k_B  # [J·s]/[J/K] = K·s
    return (hbar_over_kB * kappa) / (2.0 * math.pi)


def clausius_required_flux(kappa: float) -> float:
    """
    Minimal **sum** of longitudinal energy-flux densities required at a write.

        T_{++} + T_{--} = (κ / 2π) · (4 ln 2)

    Parameters
    ----------
    kappa : float
        Surface gravity κ in [1/s].

    Returns
    -------
    float
        Required sum of longitudinal flux densities [W/m^2].

    Notes
    -----
    This follows directly from Clausius with ΔS = 4 ln 2 nats and T = ħ κ / (2π k_B),
    expressed as a condition on the null–projected stress-energy components.
    """
    return (kappa / (2.0 * math.pi)) * (4.0 * LN2)


def clausius_check(kappa: float, flux_pp: float, flux_mm: float, *, rtol: float = RTOL, atol: float = ATOL) -> bool:
    """
    Check the Clausius write balance: T_{++} + T_{--} ≈ (κ/2π) (4 ln 2).

    Parameters
    ----------
    kappa : float
        Surface gravity κ [1/s].
    flux_pp : float
        T_{++} [W/m^2].
    flux_mm : float
        T_{--} [W/m^2].
    rtol, atol : float
        Relative/absolute tolerances for the equality.

    Returns
    -------
    bool
        True if the Clausius balance holds within tolerances.
    """
    lhs = flux_pp + flux_mm
    rhs = clausius_required_flux(kappa)
    return abs(lhs - rhs) <= (atol + rtol * abs(rhs))


def write_event_diagnostics(
    kappa: float,
    flux_pp: float,
    flux_mm: float,
    *,
    I_value: float | None = None,
    tol_zero: float = 0.0,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Dict[str, float | bool]:
    """
    Produce a compact diagnostic dict for a candidate write event at the ledger.

    Parameters
    ----------
    kappa : float
        Surface gravity κ [1/s].
    flux_pp, flux_mm : float
        Longitudinal flux densities [W/m^2].
    I_value : float, optional
        Dimensionless invariant 𝓘 evaluated at the point; if provided, we include
        placement checks (on_ledger and residual).
    tol_zero : float, default 0.0
        Tolerance for NPR zero-transport test (between writes).
    rtol, atol : float
        Tolerances controlling approximate equalities.

    Returns
    -------
    dict
        Keys:
          - 'on_ledger' (bool, if I_value provided)
          - 'placement_residual' (float, if I_value provided)
          - 'npr_between_writes_ok' (bool)
          - 'required_flux_sum' (float)
          - 'actual_flux_sum' (float)
          - 'clausius_ok' (bool)
          - 'DeltaS_min' (float, nats)
          - 'DeltaA_min' (float, m^2)
          - 'T_local' (float, K)
    """
    dS, dA = write_cost()
    T_loc = temperature_from_surface_gravity(kappa)
    req = clausius_required_flux(kappa)
    sum_flux = flux_pp + flux_mm
    diag: Dict[str, float | bool] = {
        "npr_between_writes_ok": npr_satisfied(flux_pp, flux_mm, writing=False, tol=tol_zero),
        "required_flux_sum": req,
        "actual_flux_sum": sum_flux,
        "clausius_ok": clausius_check(kappa, flux_pp, flux_mm, rtol=rtol, atol=atol),
        "DeltaS_min": dS,
        "DeltaA_min": dA,
        "T_local": T_loc,
    }
    if I_value is not None:
        diag["on_ledger"] = on_ledger(I_value, rtol=rtol, atol=atol)
        diag["placement_residual"] = placement_residual(I_value)
    return diag


__all__ = [
    "placement_residual",
    "on_ledger",
    "npr_satisfied",
    "write_cost",
    "temperature_from_surface_gravity",
    "clausius_required_flux",
    "clausius_check",
    "write_event_diagnostics",
]
