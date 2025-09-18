# qgledger/geom.py
"""
qgledger.geom
=============
Geometry and invariant helpers for the IR Quantum Gravity (ledger/NPR) framework.

This module provides:
- The **IR–scaled curvature invariant** 𝓘 = 𝓚 r_A^4 (with 𝓚 = R_{abcd}R^{abcd}),
  used to *place the ledger* via the universal level set 𝓘(r_L) = 𝓘_*.
- Schwarzschild specializations: Kretschmann scalar, photon sphere, horizon/ledger radii.
- Proper and tortoise coordinate gaps (outside the horizon), useful for echo estimates.
- A simple echo-delay estimate using an **effective** outside boundary (ECO-style),
  with a clear note on how this phenomenology maps to the IR placement inside the horizon.

Conventions
-----------
- SI units by default (meters, seconds, kilograms). Helper functions convert mass to
  geometric length M_geo = G M / c^2 where needed.
- Entropy bookkeeping is in nats elsewhere in the package; here we focus on geometry.
- The areal radius r_A is defined from the area A of the (D−2)-sphere:
  r_A = (A / Ω_{D-2})^{1/(D-2)}. In 4D Schwarzschild, r_A = r (standard areal coordinate).

Notes on scope
--------------
- We implement exact, closed-form expressions for **4D Schwarzschild** (non-rotating).
  Extensions to higher-D Schwarzschild–Tangherlini and to Kerr (spin) are planned
  and will appear as separate functions with careful documentation.

References
----------
- Kretschmann scalar (Schwarzschild, 4D): 𝓚(r) = 48 M_geo^2 / r^6 (G=c=1).
- Photon sphere: r_ph = 3 M_geo = 1.5 r_s.
- Tortoise coordinate (outside): r_* = r + r_s ln(r/r_s − 1).
"""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple

try:
    import numpy as _np  # optional: allow numpy arrays as inputs/outputs
    _HAS_NUMPY = True
    Number = Union[float, _np.ndarray]
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    Number = float

from .constants import (
    G, c, LN2,
    alpha_H, I_star,
    omega_n,
)

# -----------------------------------------------------------------------------
# Areal radius and invariant 𝓘 = 𝓚 r_A^4
# -----------------------------------------------------------------------------

def areal_radius_from_area(A: float, D: int = 4) -> float:
    """
    Compute the areal radius r_A from a (D−2)-surface area A.

    Parameters
    ----------
    A : float
        Area of the (D−2)-sphere [m^2].
    D : int, default 4
        Spacetime dimension. In usual GR, D=4 ⇒ Ω_{2}=4π.

    Returns
    -------
    float
        Areal radius r_A [m].

    Notes
    -----
    r_A satisfies A = Ω_{D-2} r_A^{D-2}. Therefore:
        r_A = (A / Ω_{D-2})^{1/(D-2)}.
    """
    if D < 3:
        raise ValueError("Areal radius requires D >= 3 (at least one angular dimension).")
    Om = omega_n(D - 2)
    if A <= 0.0 or Om <= 0.0:
        raise ValueError("Area and Ω_{D-2} must be positive.")
    return (A / Om) ** (1.0 / (D - 2))


def invariant_I(K: Number, r_A: Number) -> Number:
    """
    Compute the IR-scaled dimensionless invariant 𝓘 = 𝓚 r_A^4.

    Parameters
    ----------
    K : float or ndarray
        Kretschmann scalar 𝓚 = R_{abcd}R^{abcd} [1/m^4].
    r_A : float or ndarray
        Areal radius [m]. In 4D Schwarzschild, r_A equals the usual coordinate r.

    Returns
    -------
    float or ndarray
        Dimensionless 𝓘.

    Rationale
    ---------
    𝓘 is dimensionless without introducing an explicit UV scale (like ℓ_P),
    making it the right object to define a *universal* level set (ledger placement)
    across a wide IR range. The ledger worldvolume satisfies 𝓘(r_L) = 𝓘_*.
    """
    return K * (r_A ** 4)


# -----------------------------------------------------------------------------
# Schwarzschild (4D) helpers
# -----------------------------------------------------------------------------

def mass_geometric_length(M_kg: float) -> float:
    """
    Convert mass (kg) to geometric length M_geo = G M / c^2 [m].

    This is the usual relation in geometric units with G=c=1 (then M_geo has length units).
    """
    return G * M_kg / (c ** 2)


def schwarzschild_radius(M_kg: float) -> float:
    """
    Schwarzschild radius r_s = 2 G M / c^2 [m].
    """
    return 2.0 * G * M_kg / (c ** 2)


def photon_sphere_radius(M_kg: float) -> float:
    """
    Photon-sphere radius r_ph = 3 G M / c^2 = 1.5 r_s [m].
    """
    return 3.0 * G * M_kg / (c ** 2)


def kretschmann_schwarzschild(r: Number, M_kg: Optional[float] = None, M_geo: Optional[float] = None) -> Number:
    """
    Kretschmann scalar 𝓚(r) for 4D Schwarzschild: 𝓚 = 48 M_geo^2 / r^6.

    Parameters
    ----------
    r : float or ndarray
        Areal radius coordinate [m]; must be > 0.
    M_kg : float, optional
        Mass in kilograms; used if M_geo not provided.
    M_geo : float, optional
        Mass in geometric length units [m] (M_geo = G M / c^2). Takes precedence.

    Returns
    -------
    float or ndarray
        Kretschmann scalar 𝓚 [1/m^4].

    Notes
    -----
    This formula is valid for all r>0 (outside and inside). For r very close to 0,
    the classical expression diverges; DCT/NPR excises that regime by snap (IR scope).
    """
    if M_geo is None:
        if M_kg is None:
            raise ValueError("Provide either M_kg or M_geo.")
        M_geo = mass_geometric_length(M_kg)
    if _HAS_NUMPY and isinstance(r, _np.ndarray):
        return 48.0 * (M_geo ** 2) / (r ** 6)
    # scalar path
    if r <= 0.0:
        raise ValueError("r must be positive.")
    return 48.0 * (M_geo ** 2) / (r ** 6)


def invariant_I_schwarzschild(r: Number, M_kg: Optional[float] = None, M_geo: Optional[float] = None) -> Number:
    """
    Dimensionless invariant 𝓘(r) = 𝓚(r) r^4 for 4D Schwarzschild.

    Parameters
    ----------
    r : float or ndarray
        Areal radius [m].
    M_kg : float, optional
        Mass in kilograms (used if M_geo not provided).
    M_geo : float, optional
        Mass as geometric length [m] (M_geo = G M / c^2).

    Returns
    -------
    float or ndarray
        𝓘(r) = 48 (M_geo/r)^2 (dimensionless).
    """
    K = kretschmann_schwarzschild(r, M_kg=M_kg, M_geo=M_geo)
    return invariant_I(K, r_A=r)


def ledger_radius_schwarzschild(M_kg: float) -> float:
    """
    Ledger placement radius r_L for 4D Schwarzschild from 𝓘(r_L) = 𝓘_*.

    Using 𝓘(r) = 48 (M_geo/r)^2 and 𝓘_* = 12/alpha_H = 48 ln 2:
        r_L = sqrt(48 / 𝓘_*) * M_geo = M_geo / sqrt(ln 2).

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.

    Returns
    -------
    float
        Ledger placement radius r_L [m].

    Also useful forms
    -----------------
    In terms of the Schwarzschild radius r_s = 2 M_geo:
        r_L = sqrt(alpha_H) * r_s = r_s / sqrt(ln 2) / 2 * 2 = (1/√ln2) M_geo.
    """
    M_geo = mass_geometric_length(M_kg)
    return M_geo / math.sqrt(LN2)


# -----------------------------------------------------------------------------
# Proper and tortoise gaps (outside the horizon)
# -----------------------------------------------------------------------------

def proper_gap_outside(r1: float, r2: float, r_s: float) -> float:
    """
    Proper radial distance on a t=const Schwarzschild slice between r1 and r2 (both > r_s).

    ds = dr / sqrt(1 - r_s/r),  r > r_s.

    Closed-form anti-derivative:
      S(r) = sqrt{r(r - r_s)} + r_s * ln( sqrt{r} + sqrt{r - r_s} )

    Therefore:
      s(r1, r2) = S(r2) - S(r1)

    Parameters
    ----------
    r1, r2 : float
        Radii with r_s < r1 <= r2 [m].
    r_s : float
        Schwarzschild radius [m].

    Returns
    -------
    float
        Proper distance s(r1, r2) [m].

    Notes
    -----
    This formula is only valid for r > r_s (outside-horizon region). For echo phenomenology
    we typically pair the outer barrier region (near r≈1.5 r_s) with an *effective* inner
    boundary at r = r_s (1+ε), not the true ledger (which lies inside).
    """
    if not (r2 >= r1 > r_s > 0.0):
        raise ValueError("Require r_s < r1 <= r2 and r_s > 0.")
    def S(r: float) -> float:
        return math.sqrt(r * (r - r_s)) + r_s * math.log(math.sqrt(r) + math.sqrt(r - r_s))
    return S(r2) - S(r1)


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
    Tortoise gap Δr_* between r_in and r_out (both > r_s): Δr_* = r_*(r_out) − r_*(r_in).

    Common use in echo estimates: r_in = r_s (1+ε), r_out ≈ r_ph or the peak of the
    Regge–Wheeler barrier. Echo delay (one roundtrip) is Δt ≈ 2 Δr_* / c.
    """
    if not (r_out > r_in > r_s):
        raise ValueError("Require r_out > r_in > r_s.")
    return float(tortoise_coordinate(r_out, r_s) - tortoise_coordinate(r_in, r_s))


# -----------------------------------------------------------------------------
# Echo delay estimate (ECO-style effective boundary)
# -----------------------------------------------------------------------------

def echo_delay_seconds(M_kg: float, eps_eff: float = 1e-6) -> float:
    """
    Crude echo-delay estimate using an *effective* outside boundary at r_eff = r_s (1+ε_eff).

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
        Echo roundtrip delay Δt ≈ (2/c) [ r_*(r_pk) − r_*(r_eff) ] in seconds,
        where r_pk ≈ 1.5 r_s is the photon-sphere radius.

    Important
    ---------
    This **phenomenology follows the ECO literature**, where inner boundary conditions are
    imposed outside the horizon for time-domain simulations. In our IR ledger framework the
    true ledger lies *inside* (at r_L = M_geo/√ln 2). A future mapping will relate the inner
    geometric cavity length implied by r_L to an effective ε_eff(M) used here, so that
    the same Δt is reproduced without simulating inside the horizon.

    See also
    --------
    - :func:`ledger_radius_schwarzschild` for the IR placement inside the horizon.
    - :func:`tortoise_gap_outside` for the geometric piece used here.
    """
    if eps_eff <= 0.0:
        raise ValueError("eps_eff must be positive (effective wall must be outside the horizon).")
    r_s = schwarzschild_radius(M_kg)
    r_pk = photon_sphere_radius(M_kg)  # ~ peak of the RW/Zerilli barrier
    r_eff = r_s * (1.0 + eps_eff)
    delta_rstar = tortoise_gap_outside(r_in=r_eff, r_out=r_pk, r_s=r_s)
    return 2.0 * delta_rstar / c


# -----------------------------------------------------------------------------
# Diagnostics / checks
# -----------------------------------------------------------------------------

def check_ledger_condition(r: float, M_kg: float, atol: float = 1e-12, rtol: float = 1e-9) -> Tuple[float, float, float]:
    """
    Evaluate 𝓘(r) for Schwarzschild and report (𝓘(r), 𝓘_*, 𝓘(r)-𝓘_*).

    Parameters
    ----------
    r : float
        Radius at which to evaluate [m].
    M_kg : float
        Mass [kg].
    atol, rtol : float
        Tolerances for downstream comparisons.

    Returns
    -------
    (I_r, I_star, diff) : tuple of floats
        I_r = 𝓘(r), I_star = 𝓘_*, diff = I_r - I_star.

    Example
    -------
    >>> M = 30 * 1.98847e30  # 30 Msun
    >>> rL = ledger_radius_schwarzschild(M)
    >>> I_r, I_star, diff = check_ledger_condition(rL, M)
    >>> abs(diff) < 1e-12
    True
    """
    I_r = invariant_I_schwarzschild(r, M_kg=M_kg)
    return float(I_r), float(I_star), float(I_r - I_star)

# Kerr (rotating, uncharged) and Reissner–Nordström (charged, non-rotating) helpers.
# They mirror the Schwarzschild utilities already present in this module, including
# clean docstrings and SI-facing signatures. Where a geometric-length version is
# needed we convert using M_geo = G M / c^2 and (for charge) Q_geo^2 = G Q^2 / (4π ε0 c^4).

from typing import Tuple, Optional
import math

# Local EM constant for charge conversion (SI)
_EPS0 = 8.854_187_8128e-12  # F/m (2022 CODATA)

# ----------------------------- Kerr (a ≠ 0, Q = 0) ---------------------------------

def kerr_horizons(M_kg: float, a_dimless: Optional[float] = None, a_geo: Optional[float] = None) -> Tuple[float, float]:
    """
    Event and Cauchy horizons for a Kerr black hole (uncharged, rotating).

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    a_dimless : float, optional
        Dimensionless spin parameter χ = a / M_geo = c J / (G M^2), |χ| ≤ 1.
    a_geo : float, optional
        Spin parameter a in geometric length units [m]. If provided, takes precedence.

    Returns
    -------
    (r_minus, r_plus) : tuple of floats
        Inner (Cauchy) and outer (event) horizon radii in meters.

    Notes
    -----
    r_± = M_geo ± sqrt(M_geo^2 − a^2), with M_geo = G M / c^2 and a = χ M_geo.
    """
    M_geo = mass_geometric_length(M_kg)
    if a_geo is None:
        if a_dimless is None:
            raise ValueError("Provide either a_dimless (χ) or a_geo (length).")
        a_geo = a_dimless * M_geo
    if abs(a_geo) > M_geo:
        raise ValueError("Extremality violated: require |a| ≤ M_geo (|χ| ≤ 1).")
    root = math.sqrt(max(M_geo * M_geo - a_geo * a_geo, 0.0))
    r_plus = M_geo + root
    r_minus = M_geo - root
    return (r_minus, r_plus)


def kerr_photon_ring_equatorial(M_kg: float, a_dimless: float, prograde: bool = True) -> float:
    """
    Equatorial circular photon orbit (light ring) for Kerr.

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    a_dimless : float
        Dimensionless spin χ = a / M_geo, with |χ| ≤ 1.
    prograde : bool, default True
        If True, returns the co-rotating (prograde) light ring radius; otherwise retrograde.

    Returns
    -------
    float
        Equatorial photon circular-orbit radius r_ph [m].

    Formula
    -------
    In geometric units (M_geo = 1) the radii are
      r_ph^± = 2 [ 1 + cos( (2/3) arccos( ∓ χ ) ) ],
    where + is retrograde, − is prograde. Restoring M_geo rescales r → r * M_geo.

    Checks
    ------
    χ = 0 ⇒ r_ph = 3 M_geo (Schwarzschild);
    χ = +1 ⇒ r_ph^pro = 1 M_geo,   r_ph^retro = 4 M_geo.
    """
    if abs(a_dimless) > 1.0:
        raise ValueError("Require |a_dimless| ≤ 1.")
    M_geo = mass_geometric_length(M_kg)
    # Choose the sign inside arccos per sense: prograde uses '−', retrograde uses '+'
    arg = -a_dimless if prograde else +a_dimless
    r_over_M = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(arg)))
    return r_over_M * M_geo


# ---------------- Reissner–Nordström (Q ≠ 0, a = 0) -------------------------------

def _charge_length_from_coulomb(Q_C: float) -> float:
    """
    Convert electric charge in Coulombs to geometric-length units [m].

    Uses: Q_geo^2 = G Q^2 / (4 π ε0 c^4), so Q_geo = sqrt(G/(4π ε0)) * Q / c^2.

    Parameters
    ----------
    Q_C : float
        Electric charge in Coulombs.

    Returns
    -------
    float
        Q_geo in meters.
    """
    return math.sqrt(G / (4.0 * math.pi * _EPS0)) * Q_C / (c * c)


def rn_horizons(M_kg: float, Q_C: Optional[float] = None, Q_geo: Optional[float] = None) -> Tuple[float, float]:
    """
    Event and Cauchy horizons for a Reissner–Nordström black hole (charged, non-rotating).

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    Q_C : float, optional
        Electric charge in Coulombs (SI). Used if Q_geo not provided.
    Q_geo : float, optional
        Charge in geometric-length units [m], where Q_geo^2 = G Q^2 / (4π ε0 c^4). Takes precedence.

    Returns
    -------
    (r_minus, r_plus) : tuple of floats
        Inner (Cauchy) and outer (event) horizon radii in meters.

    Notes
    -----
    r_± = M_geo ± sqrt(M_geo^2 − Q_geo^2).
    Extremal when |Q_geo| = M_geo (double root); naked singularity if |Q_geo| > M_geo.
    """
    M_geo = mass_geometric_length(M_kg)
    if Q_geo is None:
        if Q_C is None:
            raise ValueError("Provide either Q_C (Coulombs) or Q_geo (length).")
        Q_geo = _charge_length_from_coulomb(Q_C)
    disc = M_geo * M_geo - Q_geo * Q_geo
    if disc < 0.0:
        # Not a black hole (overcharged); still return complex-like info via ValueError.
        raise ValueError("Overcharged (|Q| > M): no horizons (naked singularity).")
    root = math.sqrt(disc)
    r_plus = M_geo + root
    r_minus = M_geo - root
    return (r_minus, r_plus)


def rn_photon_sphere_radii(M_kg: float, Q_C: Optional[float] = None, Q_geo: Optional[float] = None) -> Tuple[float, float]:
    """
    Photon-sphere radii for Reissner–Nordström (two circular null orbits).

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    Q_C : float, optional
        Electric charge in Coulombs. Used if Q_geo not provided.
    Q_geo : float, optional
        Charge in geometric-length units [m]; takes precedence.

    Returns
    -------
    (r_ph_outer, r_ph_inner) : tuple of floats
        Outer (unstable) and inner radii in meters. The physically relevant photon sphere
        outside the event horizon is the larger root r_ph_outer.

    Formula
    -------
    r_ph = (3 M_geo / 2) * ( 1 ± sqrt( 1 − (8/9) (Q_geo/M_geo)^2 ) ).

    Domain
    ------
    The expression is real for Q_geo/M_geo ≤ 3/√8 ≈ 1.06066. For a true RN black hole
    (|Q_geo| ≤ M_geo) both radii are real; only the outer one lies outside the event horizon.
    """
    M_geo = mass_geometric_length(M_kg)
    if Q_geo is None:
        if Q_C is None:
            raise ValueError("Provide either Q_C (Coulombs) or Q_geo (length).")
        Q_geo = _charge_length_from_coulomb(Q_C)
    x = Q_geo / M_geo
    disc = 1.0 - (8.0 / 9.0) * (x * x)
    if disc < 0.0:
        raise ValueError("No real photon-sphere radii for given (M,Q).")
    root = math.sqrt(disc)
    pref = 1.5 * M_geo
    r_outer = pref * (1.0 + root)
    r_inner = pref * (1.0 - root)
    return (r_outer, r_inner)


def kretschmann_reissner_nordstrom(r: float, M_kg: float, Q_C: Optional[float] = None, Q_geo: Optional[float] = None) -> float:
    """
    Kretschmann scalar K(r) for Reissner–Nordström (charged, non-rotating).

    Parameters
    ----------
    r : float
        Areal radius [m], r > 0.
    M_kg : float
        Mass in kilograms.
    Q_C : float, optional
        Electric charge in Coulombs (SI). Used if Q_geo not provided.
    Q_geo : float, optional
        Charge in geometric length units [m]; takes precedence.

    Returns
    -------
    float
        K(r) in [1/m^4].

    Formula
    -------
    K(r) = 48 M_geo^2 / r^6 − 96 M_geo Q_geo^2 / r^7 + 56 Q_geo^4 / r^8.

    Notes
    -----
    Reduces to Schwarzschild K = 48 M_geo^2 / r^6 when Q=0. Diverges as r→0.
    """
    if r <= 0.0:
        raise ValueError("Require r > 0.")
    M_geo = mass_geometric_length(M_kg)
    if Q_geo is None:
        if Q_C is None:
            raise ValueError("Provide either Q_C (Coulombs) or Q_geo (length).")
        Q_geo = _charge_length_from_coulomb(Q_C)
    r2, r6, r7, r8 = r*r, r**6, r**7, r**8
    return 48.0*(M_geo**2)/r6 - 96.0*(M_geo*(Q_geo**2))/r7 + 56.0*((Q_geo**4))/r8


def invariant_I_reissner_nordstrom(r: float, M_kg: float, Q_C: Optional[float] = None, Q_geo: Optional[float] = None) -> float:
    """
    Dimensionless invariant 𝓘(r) = K(r) r^4 for Reissner–Nordström.

    Parameters
    ----------
    r : float
        Areal radius [m].
    M_kg : float
        Mass in kilograms.
    Q_C : float, optional
        Electric charge in Coulombs. Used if Q_geo not provided.
    Q_geo : float, optional
        Charge in geometric-length units [m]; takes precedence.

    Returns
    -------
    float
        𝓘(r) = K(r) r^4 (dimensionless).
    """
    K = kretschmann_reissner_nordstrom(r, M_kg=M_kg, Q_C=Q_C, Q_geo=Q_geo)
    return K * (r ** 4)

