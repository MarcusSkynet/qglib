# qgledger/tortoise.py
"""
qgledger.tortoise
=================
Tortoise-coordinate utilities and echo-delay helpers for the IR Quantum Gravity
(ledger/NPR) framework.

This module cleanly separates *outside-horizon* wave–propagation geometry
(from the photon-sphere barrier down to an **effective** inner boundary at
r = r_s (1 + ε_eff)) from the *inside-horizon* ledger placement used in the
IR theory. The aim is to provide detector-facing echo timing estimates while
documenting how ε_eff is a phenomenological stand-in for the true inner cavity
set by the ledger (which lies *inside* the horizon).

What you get
------------
- Schwarzschild tortoise coordinate r_*(r) and its gap Δr_* (outside the horizon).
- Photon-sphere radius r_ph and Schwarzschild radius r_s from mass (SI units).
- Echo roundtrip delay estimate Δt ≈ (2/c) Δr_* using an effective wall ε_eff.
- An **inverse mapping** that, given a desired gap Δr_*, solves for ε_eff by
  inverting r_* near the horizon (analytic form uses Lambert W; here we provide
  a robust Newton/bisection solver without SciPy).

Conventions
-----------
- Units: SI by default (meters, seconds, kilograms).
- We do **not** simulate inside the horizon here. The ledger radius r_L is defined
  in :mod:`qgledger.geom`; a future helper will map r_L to an ε_eff(M) so that the
  *outside* Δr_* matches the *inside* geometric cavity length.

References
----------
- Tortoise coordinate (Schwarzschild, r > r_s):
  r_* = r + r_s ln(r/r_s − 1).
- Photon sphere: r_ph = 3 G M / c^2 = 1.5 r_s.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

try:
    import numpy as _np  # optional vectorization
    _HAS_NUMPY = True
    Number = Union[float, _np.ndarray]
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    Number = float

from .constants import G, c

# ---------------------------------------------------------------------------
# Basic radii from mass (SI)
# ---------------------------------------------------------------------------

def schwarzschild_radius(M_kg: float) -> float:
    """
    Schwarzschild radius r_s = 2 G M / c^2 [m].

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.

    Returns
    -------
    float
        r_s in meters.
    """
    return 2.0 * G * M_kg / (c ** 2)


def photon_sphere_radius(M_kg: float) -> float:
    """
    Photon-sphere (light ring) radius r_ph = 3 G M / c^2 = 1.5 r_s [m].

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.

    Returns
    -------
    float
        r_ph in meters.
    """
    return 3.0 * G * M_kg / (c ** 2)


# ---------------------------------------------------------------------------
# Tortoise coordinate (outside the horizon only)
# ---------------------------------------------------------------------------

def tortoise_coordinate(r: Number, r_s: float) -> Number:
    """
    Schwarzschild tortoise coordinate r_* = r + r_s ln(r/r_s − 1), valid for r > r_s.

    Parameters
    ----------
    r : float or ndarray
        Radius [m], strictly greater than r_s.
    r_s : float
        Schwarzschild radius [m].

    Returns
    -------
    float or ndarray
        Tortoise coordinate r_* [m].

    Raises
    ------
    ValueError if any input radius ≤ r_s.
    """
    if _HAS_NUMPY and isinstance(r, _np.ndarray):
        if _np.any(r <= r_s):
            raise ValueError("tortoise_coordinate requires r > r_s.")
        return r + r_s * _np.log(r / r_s - 1.0)
    if r <= r_s:
        raise ValueError("tortoise_coordinate requires r > r_s.")
    return r + r_s * math.log(r / r_s - 1.0)


def tortoise_gap_outside(r_in: float, r_out: float, r_s: float) -> float:
    """
    Tortoise gap Δr_* between r_in and r_out (both > r_s):
        Δr_* = r_*(r_out) − r_*(r_in).

    Typical echo estimate uses r_in = r_s (1+ε_eff) and r_out ≈ r_ph.

    Parameters
    ----------
    r_in, r_out : float
        Radii with r_out > r_in > r_s [m].
    r_s : float
        Schwarzschild radius [m].

    Returns
    -------
    float
        Δr_* in meters.
    """
    if not (r_out > r_in > r_s):
        raise ValueError("Require r_out > r_in > r_s.")
    return float(tortoise_coordinate(r_out, r_s) - tortoise_coordinate(r_in, r_s))


# ---------------------------------------------------------------------------
# Echo delay using an effective outside boundary (ECO-style)
# ---------------------------------------------------------------------------

def echo_delay_seconds(M_kg: float, eps_eff: float = 1e-6) -> float:
    """
    Crude echo roundtrip delay with an **effective** wall at r_eff = r_s (1+ε_eff):

        Δt ≈ (2/c) [ r_*(r_pk) − r_*(r_eff) ],

    where r_pk ≈ r_ph = 1.5 r_s is the photon-sphere radius.

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    eps_eff : float, default 1e-6
        Dimensionless offset placing the effective wall just outside the horizon.
        Must satisfy eps_eff > 0.

    Returns
    -------
    float
        Echo roundtrip delay Δt in seconds.

    Important
    ---------
    This follows the ECO phenomenology (outer barrier + outside wall). In the IR ledger
    framework the true inner boundary lies inside the horizon. A separate mapping can
    be used to choose ε_eff(M) so that the outside Δr_* matches the cavity implied by
    the inside ledger; see :func:`epsilon_for_target_gap`.
    """
    if eps_eff <= 0.0:
        raise ValueError("eps_eff must be positive (wall must be outside the horizon).")
    r_s = schwarzschild_radius(M_kg)
    r_pk = photon_sphere_radius(M_kg)
    r_eff = r_s * (1.0 + eps_eff)
    delta_rstar = tortoise_gap_outside(r_in=r_eff, r_out=r_pk, r_s=r_s)
    return 2.0 * delta_rstar / c


# ---------------------------------------------------------------------------
# Inverting r_* near the horizon: solve for ε given a target Δr_*
# ---------------------------------------------------------------------------

def _rstar_near_horizon_eps(eps: float, r_s: float) -> float:
    """
    r_*(r_s(1+ε)) = r_s ( 1 + ε + ln ε ), exact formula specialized to r = r_s (1+ε).
    Requires ε > 0.
    """
    if eps <= 0.0:
        return float("inf")
    return r_s * (1.0 + eps + math.log(eps))


def epsilon_for_target_gap(
    r_s: float,
    delta_rstar_target: float,
    r_ref: Optional[float] = None,
    *,
    max_iter: int = 64,
    tol: float = 1e-14,
) -> float:
    """
    Given a **target** tortoise gap Δr_* = r_*(r_ref) − r_*(r_eff), solve for ε_eff
    such that r_eff = r_s (1+ε_eff).

    Parameters
    ----------
    r_s : float
        Schwarzschild radius [m].
    delta_rstar_target : float
        Desired tortoise gap [m] from r_eff up to r_ref (typically r_ref ≈ r_ph).
    r_ref : float, optional
        Outer reference radius [m]; if None, uses r_ref = 1.5 r_s (photon sphere).
    max_iter : int
        Maximum Newton iterations (with bisection fallback).
    tol : float
        Absolute tolerance on ε.

    Returns
    -------
    float
        ε_eff > 0 such that r_*(r_ref) − r_*(r_s(1+ε_eff)) = delta_rstar_target.

    Method
    ------
    We need r_*(r_s(1+ε)) = r_* (r_ref) − Δr_*^target.
    Using the near-horizon exact specialization r_* (r_s(1+ε)) = r_s(1 + ε + ln ε),
    define the scalar function
        f(ε) = r_s(1 + ε + ln ε) − r_star_target = 0,
    with derivative
        f'(ε) = r_s ( 1 + 1/ε ).
    We solve with Newton steps and guard with bisection on (ε_min, ε_max).

    Notes
    -----
    - The solution ε is unique and small for astrophysically relevant Δr_*.
    - For extremely large Δr_* targets, ε becomes exponentially small; the method
      remains stable due to the bisection guard.
    """
    if r_ref is None:
        r_ref = 1.5 * r_s
    # target r_* at the inner boundary
    rstar_ref = tortoise_coordinate(r_ref, r_s)
    rstar_target = rstar_ref - delta_rstar_target

    # Bracket: ε in (0, ε_max), choose ε_max modest (e.g., 0.5)
    eps_min = 1e-300
    eps_max = 0.5
    def f(eps: float) -> float:
        return _rstar_near_horizon_eps(eps, r_s) - rstar_target
    f_min = f(eps_min)  # very negative (→ -∞) as ε→0
    f_max = f(eps_max)  # should be positive for typical targets

    # Ensure we have a sign change
    if f_max < 0.0:
        # The target gap is too small; increase eps_max until sign change or fail gracefully.
        while f_max < 0.0 and eps_max < 0.9:
            eps_max *= 1.5
            f_max = f(eps_max)
        if f_max < 0.0:
            raise ValueError("Could not bracket solution for ε; check delta_rstar_target / r_ref.")

    # Initial guess via exponential heuristic: ε0 ≈ exp( r_*target / r_s - 1 ), ignoring the +ε term.
    eps = math.exp((rstar_target / r_s) - 1.0)
    eps = min(max(eps, 1e-12), 0.25)

    # Newton with bisection guard
    a, b = eps_min, eps_max
    fa, fb = f_min, f_max
    for _ in range(max_iter):
        # Newton step
        val = f(eps)
        if abs(val) < tol:
            return eps
        deriv = r_s * (1.0 + 1.0 / eps)
        step = val / deriv
        new_eps = eps - step
        # If new_eps is out of bracket or non-positive, take bisection
        if not (a < new_eps < b):
            new_eps = 0.5 * (a + b)
        # Update bracket
        fv = f(new_eps)
        if fv > 0.0:
            b, fb = new_eps, fv
        else:
            a, fa = new_eps, fv
        eps = new_eps
        if b - a < tol:
            return 0.5 * (a + b)

    # Fallback: midpoint
    return 0.5 * (a + b)


def eps_for_equal_gap_to_photon_sphere(M_kg: float, delta_rstar_target: float) -> float:
    """
    Convenience wrapper: return ε_eff for a given target Δr_* up to r_ref = r_ph.

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    delta_rstar_target : float
        Desired tortoise gap [m].

    Returns
    -------
    float
        ε_eff such that r_*(r_ph) − r_*(r_s(1+ε_eff)) = delta_rstar_target.
    """
    r_s = schwarzschild_radius(M_kg)
    return epsilon_for_target_gap(r_s, delta_rstar_target, r_ref=photon_sphere_radius(M_kg))


# ---------------------------------------------------------------------------
# Roundtrip time from target Δr_* (useful when mapping from inner cavity)
# ---------------------------------------------------------------------------

def echo_delay_from_target_gap(delta_rstar_target: float) -> float:
    """
    Convert a target tortoise gap Δr_* [m] into a roundtrip time Δt = 2 Δr_* / c [s].

    Parameters
    ----------
    delta_rstar_target : float
        Tortoise gap in meters.

    Returns
    -------
    float
        Roundtrip time in seconds.
    """
    return 2.0 * delta_rstar_target / c


__all__ = [
    "schwarzschild_radius",
    "photon_sphere_radius",
    "tortoise_coordinate",
    "tortoise_gap_outside",
    "echo_delay_seconds",
    "epsilon_for_target_gap",
    "eps_for_equal_gap_to_photon_sphere",
    "echo_delay_from_target_gap",
]
