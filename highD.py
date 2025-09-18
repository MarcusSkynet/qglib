# qgledger/highD.py
"""
qgledger.highD
==============
Schwarzschild–Tangherlini (D ≥ 4) helpers: horizon radius r_s(D), unit-sphere
areas Ω_{D-2}, Kretschmann scalar 𝓚_D(r), the IR-invariant
    𝓘_D(r) = 𝓚_D(r) · r^4,
and the **ledger radius** r_L(D) solving 𝓘_D(r_L) = 𝓘_* = 48 ln 2.

Why this module?
----------------
Our IR QG placement rule uses a dimensionless curvature–area invariant. In 4D
(Schwarzschild) we recover
    𝓘(r) = K(r) r^4 = 12 (r_s^2 / r^2),
so 𝓘(r_L) = 48 ln 2 ⇒ r_L = r_s / (2 √ln 2).
This module generalizes all of that to arbitrary spacetime dimension D ≥ 4.

**Key results (Tangherlini black hole in D):**
- Metric:  f(r) = 1 − μ / r^{D-3},  with mass parameter μ = r_s^{D-3}.
- Horizon radius:
    r_s^{D-3} = (16 π G_D M) / ((D−2) Ω_{D-2} c^2),
  where G_D is the D-dimensional gravitational constant and Ω_{n} is area of the
  unit n-sphere.
- Kretschmann scalar:
    𝓚_D(r) = C_D · μ^2 / r^{2(D−1)},
  with  C_D = (D−1)(D−2)^2(D−3).
  (Checks: in D=4, 𝓚_4 = 12 r_s^2 / r^6 = 48 M_geo^2 / r^6.)
- IR invariant:
    𝓘_D(r) = 𝓚_D(r) · r^4 = C_D · μ^2 / r^{2D−6}.
- Ledger radius (solve 𝓘_D(r_L) = 𝓘_*):
    r_L(D) = [ C_D μ^2 / 𝓘_* ]^{1/(2D−6)}.
  Using μ = r_s^{D−3} this becomes the **mass-independent ratio**
    r_L(D) = r_s · [ C_D / 𝓘_* ]^{1/(2D−6)}.

Practical notes
---------------
• For D ≠ 4 you must provide **G_D** (the D-dimensional Newton constant) if you
  want *numerical* r_s in SI units. The *ratio* r_L/r_s does **not** depend on G_D.
• All functions use SI units (M in kg, radii in meters) except for Ω_n which is
  dimensionless. 𝓘_* is imported from :mod:`qgledger.constants`.

References
----------
- Tangherlini, F.R. (1963): "Schwarzschild field in n dimensions and the
  dimensionality of space problem." Nuovo Cimento 27, 636–651.
- Standard BH geometry reviews generalizing curvature invariants to D>4.
"""

from __future__ import annotations

import math
from typing import Optional

from .constants import c, I_star


# ---------------------------------------------------------------------------
# Unit n-sphere area Ω_n
# ---------------------------------------------------------------------------

def omega_n(n: int) -> float:
    """
    Area of the unit n-sphere S^n:

        Ω_n = 2 π^{(n+1)/2} / Γ( (n+1)/2 ).

    Parameters
    ----------
    n : int
        Nonnegative integer (n ≥ 0).

    Returns
    -------
    float
        Ω_n (dimensionless).
    """
    if n < 0:
        raise ValueError("n must be ≥ 0")
    return 2.0 * math.pi ** ((n + 1) / 2.0) / math.gamma((n + 1) / 2.0)


# ---------------------------------------------------------------------------
# Horizon radius r_s(D) for Tangherlini (requires G_D)
# ---------------------------------------------------------------------------

def schwarzschild_radius_tangherlini(
    M_kg: float,
    D: int,
    *,
    G_D_SI: Optional[float] = None,
) -> float:
    """
    Tangherlini horizon radius r_s in D ≥ 4:

        r_s^{D−3} = (16 π G_D M) / ( (D−2) Ω_{D−2} c^2 ).

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    D : int
        Spacetime dimension (D ≥ 4).
    G_D_SI : float, optional
        D-dimensional gravitational constant in SI units. **Required** for D ≠ 4
        if you want numerical r_s. If omitted for D=4, we implicitly assume
        G_4 = qgledger.constants.G via your 4D pipeline elsewhere; here we leave
        G_D unspecified to emphasize dimensional dependence.

    Returns
    -------
    float
        r_s in meters.

    Raises
    ------
    ValueError if inputs are invalid or G_D is missing for D ≠ 4.

    Notes
    -----
    For D=4 this reduces to r_s = 2 G_4 M / c^2. You can pass your 4D G as G_D_SI
    to use this function uniformly across dimensions.
    """
    if D < 4:
        raise ValueError("Use D ≥ 4.")
    if M_kg <= 0.0:
        raise ValueError("Mass must be positive.")
    if G_D_SI is None and D != 4:
        raise ValueError("For D ≠ 4 you must provide G_D_SI to compute r_s numerically.")
    if G_D_SI is None and D == 4:
        # Fall back to the usual 4D expression using the standard G (import lazily)
        from .constants import G as G4
        return 2.0 * G4 * M_kg / (c ** 2)

    # General D
    Om = omega_n(D - 2)
    numer = 16.0 * math.pi * G_D_SI * M_kg
    denom = (D - 2) * Om * (c ** 2)
    rs_pow = numer / denom  # this equals r_s^{D-3}
    if rs_pow <= 0.0:
        raise ValueError("Computed r_s^{D-3} non-positive; check inputs.")
    return rs_pow ** (1.0 / (D - 3))


# ---------------------------------------------------------------------------
# Mass parameter μ and Kretschmann scalar 𝓚_D(r)
# ---------------------------------------------------------------------------

def mu_from_rs(rs: float, D: int) -> float:
    """
    Mass parameter μ = r_s^{D−3} (units: meters^{D−3}).

    Parameters
    ----------
    rs : float
        Horizon radius [m].
    D : int
        Dimension (D ≥ 4).

    Returns
    -------
    float
        μ in m^{D−3}.
    """
    if D < 4:
        raise ValueError("Use D ≥ 4.")
    if rs <= 0.0:
        raise ValueError("r_s must be positive.")
    return rs ** (D - 3)


def mu_from_mass(M_kg: float, D: int, *, G_D_SI: Optional[float] = None) -> float:
    """
    Compute μ directly from mass by first computing r_s(D).

    See `schwarzschild_radius_tangherlini` for caveats about G_D_SI.
    """
    rs = schwarzschild_radius_tangherlini(M_kg, D, G_D_SI=G_D_SI)
    return mu_from_rs(rs, D)


def kretschmann_tangherlini(r: float, D: int, *, rs: Optional[float] = None, mu: Optional[float] = None) -> float:
    """
    Kretschmann scalar 𝓚_D(r) for Tangherlini (D ≥ 4):

        𝓚_D(r) = C_D · μ^2 / r^{2(D−1)},
        C_D = (D−1)(D−2)^2(D−3),   μ = r_s^{D−3}.

    Parameters
    ----------
    r : float
        Areal radius [m], r > 0.
    D : int
        Dimension (D ≥ 4).
    rs : float, optional
        Horizon radius [m]. If provided, μ is taken as rs^{D−3}.
    mu : float, optional
        Mass parameter μ [m^{D−3}]. If provided, takes precedence over rs.

    Returns
    -------
    float
        𝓚_D(r) in [1/m^4].

    Checks
    ------
    D=4, μ=r_s ⇒ 𝓚_4 = 12 r_s^2 / r^6 = 48 M_geo^2 / r^6.
    """
    if D < 4:
        raise ValueError("Use D ≥ 4.")
    if r <= 0.0:
        raise ValueError("Require r > 0.")
    if mu is None:
        if rs is None:
            raise ValueError("Provide either mu or rs.")
        mu = mu_from_rs(rs, D)
    C_D = (D - 1) * ((D - 2) ** 2) * (D - 3)
    return C_D * (mu ** 2) / (r ** (2 * (D - 1)))


# ---------------------------------------------------------------------------
# IR invariant 𝓘_D(r) = 𝓚_D(r) · r^4 and ledger radius r_L(D)
# ---------------------------------------------------------------------------

def invariant_I_highD(r: float, D: int, *, rs: Optional[float] = None, mu: Optional[float] = None) -> float:
    """
    Dimensionless invariant:

        𝓘_D(r) = 𝓚_D(r) · r^4 = C_D · μ^2 / r^{2D−6}.

    Parameters
    ----------
    r : float
        Areal radius [m].
    D : int
        Dimension (D ≥ 4).
    rs, mu : as in `kretschmann_tangherlini`.

    Returns
    -------
    float
        𝓘_D(r) (dimensionless).
    """
    K = kretschmann_tangherlini(r, D, rs=rs, mu=mu)
    return K * (r ** 4)


def ledger_radius_factor(D: int) -> float:
    """
    Dimensionless ratio r_L / r_s for Tangherlini in D:

        r_L = r_s · [ C_D / 𝓘_* ]^{1/(2D−6)},
        C_D = (D−1)(D−2)^2(D−3),  𝓘_* = 48 ln 2.

    Parameters
    ----------
    D : int
        Dimension (D ≥ 4).

    Returns
    -------
    float
        r_L / r_s (dimensionless).

    Checks
    ------
    D=4 ⇒ C_4=12 ⇒ r_L/r_s = sqrt(12 / 𝓘_*) = 1 / (2 √ln 2) ≈ 0.6005.
    """
    if D < 4:
        raise ValueError("Use D ≥ 4.")
    C_D = (D - 1) * ((D - 2) ** 2) * (D - 3)
    expo = 1.0 / (2.0 * D - 6.0)
    return (C_D / I_star) ** expo


def ledger_radius_highD_from_rs(rs: float, D: int) -> float:
    """
    Ledger radius r_L given r_s (mass cancels except through r_s):

        r_L(D) = r_s · (r_L/r_s) = r_s · ledger_radius_factor(D).
    """
    if rs <= 0.0:
        raise ValueError("r_s must be positive.")
    return rs * ledger_radius_factor(D)


def ledger_radius_highD(
    M_kg: float,
    D: int,
    *,
    G_D_SI: Optional[float] = None,
) -> float:
    """
    Ledger radius r_L in meters for a Tangherlini BH in D ≥ 4.

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.
    D : int
        Spacetime dimension (D ≥ 4).
    G_D_SI : float, optional
        D-dimensional gravitational constant (required for D ≠ 4).

    Returns
    -------
    float
        r_L [m].

    Notes
    -----
    We first compute r_s(D) and then multiply by the universal **dimensionless**
    factor ledger_radius_factor(D). The dependence on G_D enters only through r_s.
    """
    rs = schwarzschild_radius_tangherlini(M_kg, D, G_D_SI=G_D_SI)
    return ledger_radius_highD_from_rs(rs, D)
