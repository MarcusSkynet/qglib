# qgledger/barrier_kerr.py
"""
qgledger.barrier_kerr
=====================
Toy Kerr (rotating) black-hole helpers for **detector-facing** echo/ringdown use:
- event-horizon radii r_±,
- equatorial photon-ring (light-ring) radii r_ph^{pro}, r_ph^{ret},
- exterior Boyer–Lindquist tortoise gap Δr_* via numerical quadrature,
- echo roundtrip delay Δt ≈ 2 Δr_* / c with an effective outside wall r_eff = r_+(1+ε),
- solve ε from a target Δr_* (bracketed root),
- light-ring orbital frequency f_ph (eikonal-QNM guide).

**Scope & philosophy**
This is a **pedagogical** module. Full Kerr perturbations require the Teukolsky /
Sasaki–Nakamura machinery; here we expose the minimal geometry needed to:
(i) place the outer barrier at the light ring,
(ii) compute an exterior tortoise gap against an effective inner boundary, and
(iii) estimate echo delays and spectral comb spacings.

All functions use **SI units** (meters, seconds, kilograms) and convert internally
to geometric quantities when needed. NumPy is not required.

References
----------
- Bardeen, Press & Teukolsky (1972): Equations for equatorial photon orbits.
- Cardoso et al. (2009): eikonal-QNM ≈ light-ring frequency and Lyapunov exponent.
"""

from __future__ import annotations

import math
from typing import Literal, Callable

from .constants import G, c
from .utils import brentq

# ---------------------------------------------------------------------------
# Basic Kerr radii and spin conventions
# ---------------------------------------------------------------------------

def mass_geometric_length(M_kg: float) -> float:
    """
    Geometric mass length M_geo = G M / c^2  [m].
    """
    if M_kg <= 0.0:
        raise ValueError("Mass must be positive.")
    return G * M_kg / (c ** 2)


def kerr_horizons(M_kg: float, a_hat: float) -> tuple[float, float, float]:
    """
    Kerr horizons and spin length.

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    a_hat : float
        Dimensionless spin a/M in [0, 1). (We assume prograde sign by convention.)

    Returns
    -------
    (r_plus, r_minus, a_len) : tuple of floats
        Outer and inner horizon radii r_± [m], and spin length a = a_hat * M_geo [m].

    Notes
    -----
    M_geo = G M / c^2,  r_± = M_geo ± sqrt(M_geo^2 − a^2).
    """
    if not (0.0 <= a_hat < 1.0):
        raise ValueError("a_hat must be in [0, 1).")
    M_geo = mass_geometric_length(M_kg)
    a_len = a_hat * M_geo
    disc = M_geo * M_geo - a_len * a_len
    if disc < 0.0:
        raise ValueError("Extremal/super-extremal spin: a_hat must satisfy a_hat < 1.")
    root = math.sqrt(disc)
    r_plus = M_geo + root
    r_minus = M_geo - root
    return r_plus, r_minus, a_len


def kerr_photon_ring_equatorial(M_kg: float, a_hat: float) -> tuple[float, float]:
    """
    Equatorial (θ=π/2) circular photon-orbit radii r_ph^{pro,ret} [m].

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    a_hat : float
        Dimensionless spin a/M in [0, 1).

    Returns
    -------
    (r_ph_pro, r_ph_ret) : tuple of floats
        Prograde (co-rotating) and retrograde (counter-rotating) photon-ring radii.

    Formula
    -------
    In geometric units (G=c=1) with M_geo as a length, Bardeen–Press–Teukolsky:
      r_ph^{pro/ret} = 2 M_geo { 1 + cos[ (2/3) arccos( ∓ a/M_geo ) ] }.
    We implement this with a_hat = a/M_geo and convert back to SI lengths.
    """
    if not (0.0 <= a_hat < 1.0):
        raise ValueError("a_hat must be in [0, 1).")
    M_geo = mass_geometric_length(M_kg)
    # Helper: r_ph = 2 M (1 + cos( (2/3) arccos( ±X ) )), with sign choice for pro/ret
    def _r_ph(sign: float) -> float:
        # sign = -1 for prograde (∓ → −), +1 for retrograde (∓ → +)
        arg = clamp_abs(a_hat * sign, 1.0)  # guard against rounding >1
        return 2.0 * M_geo * (1.0 + math.cos((2.0 / 3.0) * math.acos(arg)))
    # Prograde uses minus inside arccos → sign = -1
    r_pro = _r_ph(-1.0)
    r_ret = _r_ph(+1.0)
    return r_pro, r_ret


def clamp_abs(x: float, maxabs: float) -> float:
    """
    Clamp |x| to ≤ maxabs while preserving sign.
    """
    if x > maxabs:
        return maxabs
    if x < -maxabs:
        return -maxabs
    return x


# ---------------------------------------------------------------------------
# Boyer–Lindquist tortoise gap (exterior, equatorial)
# ---------------------------------------------------------------------------

def _delta(r: float, M_geo: float, a_len: float) -> float:
    """
    Kerr Δ(r) = r^2 − 2 M r + a^2  (geometric units; lengths).
    """
    return r * r - 2.0 * M_geo * r + a_len * a_len


def _drstar_dr_equatorial(r: float, M_geo: float, a_len: float) -> float:
    """
    Boyer–Lindquist tortoise derivative (equatorial):
      dr_*/dr = (r^2 + a^2) / Δ.
    """
    return (r * r + a_len * a_len) / _delta(r, M_geo, a_len)


def _adaptive_simpson(f: Callable[[float], float], a: float, b: float, tol: float = 1e-9, max_depth: int = 20) -> float:
    """
    Simple adaptive Simpson integrator for a smooth integrand with possible
    integrable 1/(r - r_+) divergence at the lower limit (we keep b > a > r_+).
    """
    def simpson(f, a, b):
        c = 0.5 * (a + b)
        fa, fb, fc = f(a), f(b), f(c)
        return (b - a) * (fa + 4.0 * fc + fb) / 6.0, fa, fb, fc

    def recurse(a, b, S, fa, fb, fc, depth):
        c = 0.5 * (a + b)
        S_left, fa, fc_left, fcl = simpson(f, a, c)
        S_right, fcr, fb, fcright = simpson(f, c, b)
        if depth <= 0:
            return S_left + S_right
        if abs(S_left + S_right - S) <= 15.0 * tol:
            return S_left + S_right + (S_left + S_right - S) / 15.0
        return recurse(a, c, S_left, fa, fc_left, fcl, depth - 1) + \
               recurse(c, b, S_right, fcr, fb, fcright, depth - 1)

    S0, fa, fb, fc = simpson(f, a, b)
    return recurse(a, b, S0, fa, fb, fc, max_depth)


def tortoise_gap_kerr(
    M_kg: float,
    a_hat: float,
    r_in: float,
    r_out: float,
    *,
    tol: float = 1e-9,
) -> float:
    """
    Compute the exterior tortoise gap
        Δr_* = ∫_{r_in}^{r_out} (r^2 + a^2)/Δ(r) dr,
    for Kerr in Boyer–Lindquist coordinates (equatorial).

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    a_hat : float
        Dimensionless spin a/M in [0, 1).
    r_in : float
        Lower radius [m] (must satisfy r_in > r_+).
    r_out : float
        Upper radius [m] (e.g., photon ring).
    tol : float
        Absolute tolerance for the integrator.

    Returns
    -------
    float
        Δr_* in meters.

    Raises
    ------
    ValueError for invalid ordering or r_in too close to r_+.

    Notes
    -----
    The integrand ∝ 1/Δ has a simple pole at r = r_+. Keep r_in sufficiently
    above r_+ (e.g., r_in = r_+ (1 + ε) with ε ≳ 1e-12 for macroscopic M).
    """
    if not (0.0 <= a_hat < 1.0):
        raise ValueError("a_hat must be in [0, 1).")
    if r_out <= r_in:
        raise ValueError("Require r_out > r_in.")
    M_geo = mass_geometric_length(M_kg)
    r_plus, _, a_len = kerr_horizons(M_kg, a_hat)
    if not (r_in > r_plus):
        raise ValueError("r_in must satisfy r_in > r_+.")
    f = lambda r: _drstar_dr_equatorial(r, M_geo, a_len)
    return _adaptive_simpson(f, r_in, r_out, tol=tol, max_depth=30)


# ---------------------------------------------------------------------------
# Effective wall model and ε inversion
# ---------------------------------------------------------------------------

def echo_delay_seconds_kerr(
    M_kg: float,
    a_hat: float,
    *,
    eps_eff: float = 1e-6,
    branch: Literal["pro", "ret"] = "pro",
    tol: float = 1e-9,
) -> float:
    """
    Echo roundtrip delay for Kerr using an **outside** effective wall:

        Δt ≈ (2/c) · Δr_*,
        Δr_* = ∫_{r_eff}^{r_ph^{branch}} (r^2 + a^2)/Δ dr,
        r_eff = r_+ (1 + ε_eff).

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    a_hat : float
        Dimensionless spin a/M in [0,1).
    eps_eff : float, default 1e-6
        Dimensionless offset for the effective wall.
    branch : {"pro","ret"}, default "pro"
        Use prograde ("pro") or retrograde ("ret") photon ring as r_out.
    tol : float
        Integration tolerance.

    Returns
    -------
    float
        Δt in seconds.
    """
    if eps_eff <= 0.0:
        raise ValueError("eps_eff must be positive.")
    r_plus, _, _ = kerr_horizons(M_kg, a_hat)
    r_eff = r_plus * (1.0 + eps_eff)
    r_pro, r_ret = kerr_photon_ring_equatorial(M_kg, a_hat)
    r_out = r_pro if branch == "pro" else r_ret
    gap = tortoise_gap_kerr(M_kg, a_hat, r_eff, r_out, tol=tol)
    return 2.0 * gap / c


def epsilon_from_target_gap_kerr(
    M_kg: float,
    a_hat: float,
    delta_rstar_target: float,
    *,
    branch: Literal["pro", "ret"] = "pro",
    tol: float = 1e-12,
) -> float:
    """
    Solve for ε_eff > 0 such that the exterior tortoise gap equals the target:

        ∫_{r_+(1+ε)}^{r_ph} (r^2 + a^2)/Δ dr  =  Δr_*^{target}.

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    a_hat : float
        Dimensionless spin in [0,1).
    delta_rstar_target : float
        Target gap [m].
    branch : {"pro","ret"}, default "pro"
        Choose the outer radius r_ph^{branch}.
    tol : float
        Absolute tolerance on ε for the root finder.

    Returns
    -------
    float
        ε_eff > 0 (dimensionless).

    Strategy
    --------
    Monotonicity: as ε → 0^+ (wall → r_+), Δr_* → ∞ (log divergence); as ε grows,
    Δr_* decreases. Hence there is a unique ε solving the equation for any finite,
    positive target gap. We bracket ε ∈ (ε_min, ε_max) and use `utils.brentq`.
    """
    if delta_rstar_target <= 0.0:
        raise ValueError("delta_rstar_target must be positive.")
    r_plus, _, _ = kerr_horizons(M_kg, a_hat)
    r_pro, r_ret = kerr_photon_ring_equatorial(M_kg, a_hat)
    r_out = r_pro if branch == "pro" else r_ret

    def gap_of_eps(eps: float) -> float:
        r_eff = r_plus * (1.0 + eps)
        return tortoise_gap_kerr(M_kg, a_hat, r_eff, r_out, tol=1e-9)

    # Root of F(ε) = gap(ε) − target = 0
    def F(eps: float) -> float:
        return gap_of_eps(eps) - delta_rstar_target

    # Bracket: start with a generous [1e-12, 0.5]
    a_eps, b_eps = 1e-12, 0.5
    Fa = F(a_eps)
    Fb = F(b_eps)
    # If Fb > 0, expand b
    while Fb > 0.0 and b_eps < 0.99:
        b_eps *= 1.5
        Fb = F(b_eps)
    if Fa * Fb > 0.0:
        raise ValueError("Could not bracket ε: target gap may be too large/small.")

    return brentq(F, a_eps, b_eps, tol=tol, max_iter=200)


# ---------------------------------------------------------------------------
# Light-ring orbital frequency (eikonal guide)
# ---------------------------------------------------------------------------

def light_ring_frequency_hz(
    M_kg: float,
    a_hat: float,
    *,
    branch: Literal["pro", "ret"] = "pro",
) -> float:
    """
    Equatorial light-ring **orbital** frequency f_ph [Hz] (coordinate-time rate).

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    a_hat : float
        Dimensionless spin a/M in [0,1).
    branch : {"pro","ret"}, default "pro"
        Choose prograde or retrograde ring.

    Returns
    -------
    float
        f_ph in Hz.

    Approximation
    -------------
    In geometric units (G=c=1), the coordinate angular velocity at the equatorial
    photon ring is approximated by
        Ω = 1 / (a ± r_ph^{3/2} / M_geo^{1/2}),
    with the upper sign for **prograde** (denominator a + …) and lower sign for
    **retrograde** (−a + …). We convert Ω [1/m] → f [Hz] via f = (c Ω)/(2π).

    Caveat
    ------
    This is a standard eikonal guide, not an exact Teukolsky/QNM result.
    """
    if not (0.0 <= a_hat < 1.0):
        raise ValueError("a_hat must be in [0,1).")
    M_geo = mass_geometric_length(M_kg)
    a_len = a_hat * M_geo
    r_pro, r_ret = kerr_photon_ring_equatorial(M_kg, a_hat)
    r = r_pro if branch == "pro" else r_ret
    denom = a_len + math.sqrt(r ** 3 / M_geo) if branch == "pro" else (-a_len + math.sqrt(r ** 3 / M_geo))
    if denom <= 0.0:
        # Degenerate/retrograde near-extremal corner; return 0 as safe default
        return 0.0
    omega_geom = 1.0 / denom  # [1/m]
    return (c * omega_geom) / (2.0 * math.pi)


__all__ = [
    "kerr_horizons",
    "kerr_photon_ring_equatorial",
    "tortoise_gap_kerr",
    "echo_delay_seconds_kerr",
    "epsilon_from_target_gap_kerr",
    "light_ring_frequency_hz",
]
