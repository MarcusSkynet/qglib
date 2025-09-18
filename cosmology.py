# qgledger/cosmology.py
"""
qgledger.cosmology
==================
Flat-FRW (and mildly curved) horizon thermodynamics and IR invariants for the
ledger/NPR framework. This module offers helper functions to compute the Hubble
rate H(a), the apparent-horizon radius R_A, Gibbons–Hawking temperature T_A,
horizon area/entropy, and the **IR-scaled invariant**
    𝓘_FRW(a) = 𝓚_FRW(a) · R_A(a)^4,
with a simple placement check against the universal threshold 𝓘_*.

Scope & philosophy
------------------
- **Units:** SI (meters, seconds, kelvin). Scale factor `a` is unitless and
  normalized so that a=1 today.
- **Model:** minimal ΛCDM + optional constant-w dark energy and small curvature.
  The formulas here are pedagogical but dimensionally consistent.
- **IR invariant:** We use a standard FRW Kretschmann scalar (k≈0 case) expressed
  in terms of H(a) and Ḣ(a), then form 𝓘=𝓚 R_A^4, mirroring the black-hole
  construction (with r_A ↦ R_A). This yields a *dimensionless* diagnostic that
  can be compared to 𝓘_* = 48 ln 2.

References (flat FRW)
---------------------
- Friedmann:  H^2 = H0^2 [ Ω_r a^{-4} + Ω_m a^{-3} + Ω_k a^{-2} + Ω_DE a^{-3(1+w)} ].
- Raychaudhuri:  Ḣ = − (H0^2) [ 2 Ω_r a^{-4} + (3/2) Ω_m a^{-3} + (3/2)(1+w) Ω_DE a^{-3(1+w)} ] + Ω_k H0^2 a^{-2}.
- Apparent horizon (k≈0): R_A ≈ c / H.
- Gibbons–Hawking temperature: T_A = ħ H / (2π k_B).
- Kretschmann (k=0): 𝓚_FRW = 12 [ (Ḣ + H^2)^2 + H^4 ].

Notes
-----
- For small |Ω_k| we include a simple correction in Ḣ; the horizon radius uses
  the flat expression R_A = c/H for clarity (good for |Ω_k| ≪ 1 over late times).
- If you need exact curved-FRW horizon radius, use R_A = c / sqrt(H^2 + k c^2/a^2),
  with k∈{−1,0,+1} and appropriate normalization; we keep the API flat-focused.

Quick start
-----------
>>> from qgledger.cosmology import hubble_H, apparent_horizon_radius, horizon_temperature
>>> H = hubble_H(a=1.0, H0=70e3/3.085677581e22, Om=0.315, Or=8.5e-5, Ode=0.685, w=-1.0)
>>> R = apparent_horizon_radius(H)
>>> T = horizon_temperature(H)
>>> I = invariant_I_frw(a=1.0, H0=H0, Om=0.315, Or=8.5e-5, Ode=0.685, w=-1.0)
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore

from .constants import c, hbar, k_B, LN2, alpha_H, I_star

# -----------------------------------------------------------------------------
# Hubble rate H(a) and its time derivative Ḣ(a)
# -----------------------------------------------------------------------------

def hubble_H(
    a: float,
    *,
    H0: float,
    Om: float,
    Or: float = 0.0,
    Ode: float = 0.0,
    Ok: float | None = None,
    w: float = -1.0,
) -> float:
    """
    Hubble rate H(a) (SI units: 1/s) for a constant-w dark-energy model.

    Parameters
    ----------
    a : float
        Scale factor (dimensionless), normalized so that a=1 today.
    H0 : float
        Present Hubble rate [1/s].
    Om : float
        Matter density fraction today (baryons + CDM).
    Or : float, default 0.0
        Radiation density fraction today.
    Ode : float, default 0.0
        Dark-energy density fraction today.
    Ok : float, optional
        Curvature density fraction today; if None, set Ok = 1 − Om − Or − Ode.
    w : float, default −1.0
        Constant equation of state for the dark-energy component (w=−1 is Λ).

    Returns
    -------
    float
        H(a) in [1/s].

    Notes
    -----
    H^2(a) = H0^2 [ Or a^{-4} + Om a^{-3} + Ok a^{-2} + Ode a^{-3(1+w)} ].
    """
    if a <= 0.0:
        raise ValueError("Scale factor a must be positive.")
    if Ok is None:
        Ok = 1.0 - (Om + Or + Ode)
    term_r = Or * a ** (-4.0)
    term_m = Om * a ** (-3.0)
    term_k = Ok * a ** (-2.0)
    term_de = Ode * a ** (-3.0 * (1.0 + w))
    H2 = (H0 ** 2) * (term_r + term_m + term_k + term_de)
    if H2 < 0.0:
        raise ValueError("Computed H^2 < 0; check Ω parameters.")
    return math.sqrt(H2)


def hubble_Hdot(
    a: float,
    *,
    H0: float,
    Om: float,
    Or: float = 0.0,
    Ode: float = 0.0,
    Ok: float | None = None,
    w: float = -1.0,
) -> float:
    """
    Time derivative Ḣ(a) in [1/s^2] for constant-w dark energy.

    Parameters
    ----------
    a, H0, Om, Or, Ode, Ok, w : as in `hubble_H`.

    Returns
    -------
    float
        Ḣ(a) in [1/s^2].

    Derivation (flat-friendly)
    --------------------------
    Using the second Friedmann/Raychaudhuri equation one finds for mixed fluids:
      Ḣ = − H0^2 [ 2 Or a^{-4} + (3/2) Om a^{-3} + (3/2)(1+w) Ode a^{-3(1+w)} ] + Ok H0^2 a^{-2}.
    For flat models set Ok = 0 (default is Ok = 1 − Om − Or − Ode).
    """
    if Ok is None:
        Ok = 1.0 - (Om + Or + Ode)
    term_r = 2.0 * Or * a ** (-4.0)
    term_m = 1.5 * Om * a ** (-3.0)
    term_de = 1.5 * (1.0 + w) * Ode * a ** (-3.0 * (1.0 + w))
    term_k = Ok * a ** (-2.0)
    return - (H0 ** 2) * (term_r + term_m + term_de) + (H0 ** 2) * term_k


# -----------------------------------------------------------------------------
# Apparent horizon radius and thermodynamics
# -----------------------------------------------------------------------------

def apparent_horizon_radius(H: float) -> float:
    """
    Apparent-horizon radius R_A ≈ c / H (flat-FRW approximation).

    Parameters
    ----------
    H : float
        Hubble rate [1/s].

    Returns
    -------
    float
        R_A in meters.

    Notes
    -----
    For |Ω_k| ≪ 1 this is an excellent approximation at late times.
    """
    if H <= 0.0:
        raise ValueError("H must be positive.")
    return c / H


def horizon_temperature(H: float) -> float:
    """
    Gibbons–Hawking temperature T_A = ħ H / (2π k_B) [K].

    Parameters
    ----------
    H : float
        Hubble rate [1/s].

    Returns
    -------
    float
        Temperature in kelvin.
    """
    return (hbar * H) / (2.0 * math.pi * k_B)


def horizon_area(R_A: float) -> float:
    """
    Horizon area A = 4π R_A^2 [m^2].

    Parameters
    ----------
    R_A : float
        Apparent-horizon radius [m].

    Returns
    -------
    float
        Area [m^2].
    """
    return 4.0 * math.pi * (R_A ** 2)


def horizon_entropy_nats(R_A: float, ell_P: float) -> float:
    """
    Horizon entropy in nats: S = A / (4 ℓ_P^2).

    Parameters
    ----------
    R_A : float
        Apparent-horizon radius [m].
    ell_P : float
        Planck length [m] (e.g., from qgledger.constants.ell_P).

    Returns
    -------
    float
        Entropy in nats.
    """
    A = horizon_area(R_A)
    return A / (4.0 * (ell_P ** 2))


# -----------------------------------------------------------------------------
# IR-scaled invariant 𝓘_FRW(a) = 𝓚_FRW(a) · R_A^4
# -----------------------------------------------------------------------------

def kretschmann_frw_flat(H: float, Hdot: float) -> float:
    """
    Kretschmann scalar for k=0 FRW in terms of H and Ḣ:

        𝓚_FRW = 12 [ (Ḣ + H^2)^2 + H^4 ]   [units: 1/s^4].

    Parameters
    ----------
    H : float
        Hubble rate [1/s].
    Hdot : float
        Time derivative Ḣ [1/s^2].

    Returns
    -------
    float
        𝓚_FRW in [1/s^4].

    Notes
    -----
    This expression uses cosmic time; it is standard for flat FRW. For small
    curvature the corrections are subleading at late times.
    """
    term1 = (Hdot + H * H)
    return 12.0 * (term1 * term1 + (H ** 4))


def invariant_I_frw(
    a: float,
    *,
    H0: float,
    Om: float,
    Or: float = 0.0,
    Ode: float = 0.0,
    Ok: float | None = None,
    w: float = -1.0,
) -> float:
    """
    Dimensionless IR invariant 𝓘_FRW(a) = 𝓚_FRW(a) · R_A(a)^4.

    Parameters
    ----------
    a : float
        Scale factor (dimensionless).
    H0, Om, Or, Ode, Ok, w : as in `hubble_H`.

    Returns
    -------
    float
        𝓘_FRW(a) (dimensionless).

    Implementation
    --------------
    1) Compute H(a) and Ḣ(a).
    2) Compute R_A = c/H.
    3) Compute 𝓚_FRW = 12 [ (Ḣ + H^2)^2 + H^4 ].
    4) Return 𝓘_FRW = 𝓚_FRW · R_A^4  (units cancel).
    """
    H = hubble_H(a, H0=H0, Om=Om, Or=Or, Ode=Ode, Ok=Ok, w=w)
    Hdot = hubble_Hdot(a, H0=H0, Om=Om, Or=Or, Ode=Ode, Ok=Ok, w=w)
    RA = apparent_horizon_radius(H)
    K = kretschmann_frw_flat(H, Hdot)
    return K * (RA ** 4)


def ledger_condition_frw(
    a: float,
    *,
    H0: float,
    Om: float,
    Or: float = 0.0,
    Ode: float = 0.0,
    Ok: float | None = None,
    w: float = -1.0,
) -> Tuple[float, float, float]:
    """
    Evaluate the FRW invariant and compare to the universal threshold 𝓘_*.

    Parameters
    ----------
    a : float
        Scale factor.
    H0, Om, Or, Ode, Ok, w : as in `hubble_H`.

    Returns
    -------
    (I_val, I_star, diff) : tuple of floats
        I_val = 𝓘_FRW(a), I_star = 48 ln 2, diff = I_val − I_star.

    Interpretation
    --------------
    This mirrors the black-hole ledger placement logic but for the cosmological
    horizon. It serves as a diagnostic: where (if anywhere) does the FRW 𝓘 cross
    the universal level 𝓘_*? In Λ-dominated epochs, 𝓘_FRW → 48 ln 2 as H approaches
    a constant (de Sitter-like), up to small corrections.
    """
    I_val = invariant_I_frw(a, H0=H0, Om=Om, Or=Or, Ode=Ode, Ok=Ok, w=w)
    return float(I_val), float(I_star), float(I_val - I_star)


__all__ = [
    "hubble_H",
    "hubble_Hdot",
    "apparent_horizon_radius",
    "horizon_temperature",
    "horizon_area",
    "horizon_entropy_nats",
    "kretschmann_frw_flat",
    "invariant_I_frw",
    "ledger_condition_frw",
]
