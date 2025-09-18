# qgledger/barrier.py
"""
qgledger.barrier
================
Regge–Wheeler (axial) barrier for Schwarzschild black holes, peak finding,
and a first-order WKB transmission/reflection estimate suitable for simple
echo/ringdown pedagogy.

Scope
-----
- Exact formulas here are for *4D Schwarzschild* and the **axial** (odd-parity)
  Regge–Wheeler potential:
      V_RW(r) = f(r) [ ℓ(ℓ+1)/r^2 − 6 M_geo/r^3 ],   f(r) = 1 − r_s/r,
  with M_geo = G M / c^2 and r_s = 2 M_geo. Units: with r in meters and M_geo in
  meters, V has units 1/m^2 (geometric, c=G=1 internally).
- We locate the barrier **maximum** r_pk ≈ 3 M_geo (near the photon sphere),
  return V0 = V(r_pk), and compute the curvature in tortoise coordinate:
      d^2V/dr_*^2 = f^2 V''(r) + f f'(r) V'(r),
  which controls the WKB thickness near the top of the barrier.
- A first-order (Schutz–Will) WKB transmission for a wave of angular frequency ω
  provides a fast estimate of transmission/reflection across the peak.

Why this matters
----------------
The barrier around the photon sphere shapes ringdown and sets, together with an
inner boundary, the **echo timing and spectral comb**. Even in our IR snap/ledger
framework (where the true inner boundary lies inside the horizon), detector-facing
models use the *exterior* barrier plus an *effective* outside inner wall.

References
----------
- Regge & Wheeler (1957); Zerilli (1970); Chandrasekhar (1983);
  Berti, Cardoso & Starinets (2009) for broader context on BH perturbations.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore

from .constants import G, c
from .geom import mass_geometric_length, schwarzschild_radius

# -----------------------------------------------------------------------------
# Regge–Wheeler potential and derivatives (analytic in r)
# -----------------------------------------------------------------------------

def V_RW(r: float, l: int, *, M_kg: float | None = None, M_geo: float | None = None) -> float:
    """
    Regge–Wheeler (axial) potential V(r) for Schwarzschild.

        V(r) = f(r) [ L/r^2 − 6 M_geo/r^3 ],   f(r) = 1 − r_s/r,  L = ℓ(ℓ+1).

    Parameters
    ----------
    r : float
        Areal radius [m], r > r_s.
    l : int
        Multipole index ℓ (use ℓ ≥ 2 for gravitational perturbations).
    M_kg : float, optional
        Mass in kilograms (used if M_geo not supplied).
    M_geo : float, optional
        Mass in geometric length units [m] (M_geo = G M / c^2). Takes precedence.

    Returns
    -------
    float
        V(r) in [1/m^2].

    Raises
    ------
    ValueError if r ≤ r_s or ℓ < 2.
    """
    if l < 2:
        raise ValueError("Use l >= 2 for gravitational axial modes.")
    if M_geo is None:
        if M_kg is None:
            raise ValueError("Provide either M_kg or M_geo.")
        M_geo = mass_geometric_length(M_kg)
    r_s = 2.0 * M_geo
    if r <= r_s:
        raise ValueError("V_RW is defined here for r > r_s (exterior region).")
    f = 1.0 - r_s / r
    L = l * (l + 1.0)
    return f * (L / (r * r) - 6.0 * M_geo / (r ** 3))


def _dV_dr(r: float, l: int, M_geo: float) -> float:
    """
    First derivative dV/dr (analytic) for V_RW.
    """
    r_s = 2.0 * M_geo
    f = 1.0 - r_s / r
    fp = r_s / (r * r)                 # f' = + r_s / r^2
    L = l * (l + 1.0)
    A = L / (r * r) - 6.0 * M_geo / (r ** 3)
    Ap = -2.0 * L / (r ** 3) + 18.0 * M_geo / (r ** 4)
    return fp * A + f * Ap


def _d2V_dr2(r: float, l: int, M_geo: float) -> float:
    """
    Second derivative d^2V/dr^2 (analytic) for V_RW.
    """
    r_s = 2.0 * M_geo
    f = 1.0 - r_s / r
    fp = r_s / (r * r)
    fpp = -2.0 * r_s / (r ** 3)
    L = l * (l + 1.0)
    A = L / (r * r) - 6.0 * M_geo / (r ** 3)
    Ap = -2.0 * L / (r ** 3) + 18.0 * M_geo / (r ** 4)
    App = 6.0 * L / (r ** 4) - 72.0 * M_geo / (r ** 5)
    return fpp * A + 2.0 * fp * Ap + f * App


def d2V_drstar2(r: float, l: int, *, M_kg: float | None = None, M_geo: float | None = None) -> float:
    """
    Tortoise-curvature d^2V/dr_*^2 at radius r (outside region).

    Using dr_*/dr = 1/f, so d/dr_* = f d/dr ⇒
        d^2V/dr_*^2 = f^2 V''(r) + f f'(r) V'(r).

    Parameters
    ----------
    r : float
        Areal radius [m], r > r_s.
    l : int
        Multipole index ℓ ≥ 2.
    M_kg, M_geo : float
        Mass (kg) or geometric mass-length (m).

    Returns
    -------
    float
        d^2V/dr_*^2 in [1/m^4].
    """
    if M_geo is None:
        if M_kg is None:
            raise ValueError("Provide either M_kg or M_geo.")
        M_geo = mass_geometric_length(M_kg)
    r_s = 2.0 * M_geo
    if r <= r_s:
        raise ValueError("d2V/dr_*^2 here is for r > r_s.")
    f = 1.0 - r_s / r
    fp = r_s / (r * r)
    Vp = _dV_dr(r, l, M_geo)
    Vpp = _d2V_dr2(r, l, M_geo)
    return (f * f) * Vpp + f * fp * Vp


# -----------------------------------------------------------------------------
# Barrier peak location (golden-section maximize V_RW)
# -----------------------------------------------------------------------------

def _golden_max(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """
    Golden-section search to locate the maximum of a unimodal function on [a,b].
    """
    invphi = (math.sqrt(5.0) - 1.0) / 2.0  # 1/φ
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0 # 1/φ^2
    (a, b) = (float(a), float(b))
    h = b - a
    if h <= tol:
        return 0.5 * (a + b)
    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)
    for _ in range(n):
        if yc < yd:  # move left bound up (since we maximize)
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
        else:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
    return c if yc > yd else d


def rw_barrier_peak(l: int, *, M_kg: float | None = None, M_geo: float | None = None) -> Tuple[float, float, float]:
    """
    Locate the Regge–Wheeler barrier peak for a given ℓ and mass.

    Parameters
    ----------
    l : int
        Multipole index (use ℓ ≥ 2).
    M_kg, M_geo : float
        Mass (kg) or geometric mass-length (m). One must be provided.

    Returns
    -------
    (r_peak, V0, Vpp_star) : tuple of floats
        Peak radius r_peak [m], barrier height V0 [1/m^2], and tortoise curvature
        Vpp_star = d^2V/dr_*^2 at the peak [1/m^4] (negative near the top).

    Notes
    -----
    We bracket the maximum in [2.1 M_geo, 6 M_geo], which safely contains the
    photon-sphere region for ℓ ≥ 2.
    """
    if l < 2:
        raise ValueError("Use l >= 2 for gravitational axial modes.")
    if M_geo is None:
        if M_kg is None:
            raise ValueError("Provide either M_kg or M_geo.")
        M_geo = mass_geometric_length(M_kg)
    a = 2.1 * M_geo
    b = 6.0 * M_geo
    f = lambda r: V_RW(r, l, M_geo=M_geo)
    r_pk = _golden_max(f, a, b, tol=1e-12, max_iter=200)
    V0 = V_RW(r_pk, l, M_geo=M_geo)
    Vpp_star = d2V_drstar2(r_pk, l, M_geo=M_geo)
    return r_pk, V0, Vpp_star


# -----------------------------------------------------------------------------
# First-order WKB transmission / reflection near the peak
# -----------------------------------------------------------------------------

def wkb_transmission_reflection(
    f_hz: float,
    l: int,
    *,
    M_kg: float | None = None,
    M_geo: float | None = None,
) -> Tuple[float, float, float, float, float]:
    """
    First-order (Schutz–Will) WKB transmission/reflection at the barrier peak.

    We approximate the potential near the top as an inverted parabola in r_*:
        V(r_*) ≈ V0 − (1/2) |V''_*| (r_* − r_*0)^2,
    then the transmission for an incident wave with angular frequency ω is
        T ≈ 1 / ( 1 + exp( 2π (V0 − ω^2) / sqrt(−2 V''_*) ) ),
    and R = 1 − T. Here V and ω^2 carry units 1/m^2 in geometric units.

    Parameters
    ----------
    f_hz : float
        Physical frequency in hertz (cycles per second).
    l : int
        Multipole index (ℓ ≥ 2).
    M_kg, M_geo : float
        Mass (kg) or geometric mass-length (m). One must be provided.

    Returns
    -------
    (T, R, r_peak, V0, Vpp_star) : tuple of floats
        Transmission probability T ∈ (0,1), reflection R = 1 − T,
        peak radius r_peak [m], barrier height V0 [1/m^2],
        tortoise curvature Vpp_star = d^2V/dr_*^2 [1/m^4] (negative).

    Units & conversions
    -------------------
    We convert the *physical* angular frequency ω_phys = 2π f [rad/s]
    to geometric units via ω = ω_phys / c [1/m], so that ω^2 can be
    compared directly to V0 in 1/m^2.
    """
    if M_geo is None:
        if M_kg is None:
            raise ValueError("Provide either M_kg or M_geo.")
        M_geo = mass_geometric_length(M_kg)
    r_peak, V0, Vpp_star = rw_barrier_peak(l, M_geo=M_geo)
    # Angular frequency in geometric units (1/m)
    omega_geom = (2.0 * math.pi * f_hz) / c
    # WKB exponent parameter
    # Guard against roundoff if Vpp_star is near zero (should be negative at peak).
    if Vpp_star >= 0.0:
        # Degenerate case (flat-top); return full reflection as a safe default.
        return 0.0, 1.0, r_peak, V0, Vpp_star
    num = V0 - (omega_geom ** 2)
    denom = math.sqrt(-2.0 * Vpp_star)
    a = num / denom
    # Limit handling to avoid overflow in exp for large |a|
    if a <= -40.0:
        T = 1.0  # way above barrier
    elif a >= 40.0:
        T = 0.0  # way below barrier
    else:
        T = 1.0 / (1.0 + math.exp(2.0 * math.pi * a))
    R = 1.0 - T
    return T, R, r_peak, V0, Vpp_star


__all__ = [
    "V_RW",
    "d2V_drstar2",
    "rw_barrier_peak",
    "wkb_transmission_reflection",
]
