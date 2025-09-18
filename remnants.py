# qgledger/remnants.py
"""
qgledger.remnants
=================
Hawking evaporation (pedagogical level), IR/DCT suppression toward a remnant,
and simple time–evolution utilities for black-hole mass history in SI units.

What this provides
------------------
- Closed-form Hawking temperature and (species-aggregated) power scaling.
- Dimensionally consistent dM/dt for Schwarzschild evaporation.
- A smooth **suppression** factor that enforces stall at a chosen remnant mass
  M_rem, reflecting the IR snap/NPR picture (no extreme UV).
- Closed-form lifetime estimate (Hawking-only) and a simple ODE integrator
  (Hawking × suppression) that stops at M_rem.

Scope & caveats
---------------
This is a **teaching** module. It aggregates greybody factors and particle
content into a single effective parameter `g_eff` (dimensionless). For detector-
grade work, one would integrate frequency-dependent emissivities channel by
channel. Here we retain the M^{-2} power law and a g_eff rescaling.

Units
-----
All functions are **SI** by default:
- Mass in kilograms [kg], power in watts [W], time in seconds [s], temperature [K].

References (standard results)
-----------------------------
- Hawking temperature:  T_H = ħ c^3 / (8 π G k_B M).
- Power (scaling):      P ≈ (ħ c^6) / (15360 π G^2) * g_eff / M^2.
- Lifetime (scaling):   τ ≈ (5120 π G^2 / (ħ c^4)) * M^3 / g_eff.
The coefficients depend on particle content and greybody factors; we expose
`g_eff` to absorb these differences in a single knob.

IR/DCT idea
-----------
In the IR ledger/NPR framework, extreme UV is excised. Practically, we model
the late-time approach to a finite **remnant** by multiplying Hawking dM/dt by
a smooth suppression S(M; M_rem) that → 0 as M→M_rem^+ and enforces stall.

Quick start
-----------
>>> from qgledger.remnants import hawking_temperature, lifetime_hawking_closedform
>>> from qgledger.remnants import integrate_evaporation_suppressed
>>> M0 = 5e11  # kg
>>> T = hawking_temperature(M0)
>>> tH = lifetime_hawking_closedform(M0, g_eff=1.0)
>>> t, M = integrate_evaporation_suppressed(M0, M_rem=2e5, g_eff=1.0, dt=1e6)
"""

from __future__ import annotations

import math
from typing import Callable, Tuple, List, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore

from .constants import G, c, hbar, k_B

# -----------------------------------------------------------------------------
# Hawking temperature and power (species-aggregated)
# -----------------------------------------------------------------------------

def hawking_temperature(M_kg: float) -> float:
    """
    Hawking temperature for a Schwarzschild black hole (SI).

        T_H(M) = ħ c^3 / (8 π G k_B M)

    Parameters
    ----------
    M_kg : float
        Mass [kg].

    Returns
    -------
    float
        Temperature [K].
    """
    if M_kg <= 0.0:
        raise ValueError("Mass must be positive.")
    return (hbar * c**3) / (8.0 * math.pi * G * k_B * M_kg)


def hawking_power(M_kg: float, g_eff: float = 1.0) -> float:
    """
    Effective Hawking power (aggregate over species) with a single knob g_eff.

        P(M) ≈ g_eff * (ħ c^6) / (15360 π G^2) * 1/M^2

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    g_eff : float, default 1.0
        Effective degrees-of-freedom/greybody factor (dimensionless). Use g_eff≈1
        for simple pedagogy. Larger values radiate faster.

    Returns
    -------
    float
        Power [W] radiated to infinity.

    Notes
    -----
    The numeric coefficient (15360 π) corresponds to a standard idealized
    aggregation; detailed spectra change this O(1) factor. We keep it explicit.
    """
    if M_kg <= 0.0:
        raise ValueError("Mass must be positive.")
    C = (hbar * c**6) / (15360.0 * math.pi * G**2)
    return g_eff * C / (M_kg**2)


def dMdt_hawking(M_kg: float, g_eff: float = 1.0) -> float:
    """
    Hawking mass-loss rate (Schwarzschild), derived from P = −d(M c^2)/dt.

        dM/dt = − P / c^2 = − g_eff * C / (c^2 M^2),   C = ħ c^6 / (15360 π G^2)

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    g_eff : float, default 1.0
        Effective emissivity factor (dimensionless).

    Returns
    -------
    float
        dM/dt in [kg/s] (negative).
    """
    P = hawking_power(M_kg, g_eff=g_eff)
    return - P / (c**2)


def lifetime_hawking_closedform(M0_kg: float, g_eff: float = 1.0) -> float:
    """
    Closed-form Hawking lifetime (to M → 0) for the M^{-2} law.

        τ_H ≈ (5120 π G^2 / (ħ c^4)) * M0^3 / g_eff

    Parameters
    ----------
    M0_kg : float
        Initial mass [kg].
    g_eff : float, default 1.0
        Effective emissivity factor.

    Returns
    -------
    float
        Lifetime [s] in the idealized Hawking-only picture.

    Notes
    -----
    In the IR/DCT framework we expect **stall** at a nonzero M_rem; use the
    integrator below with a suppression factor to estimate the time-to-stall.
    """
    if M0_kg <= 0.0:
        raise ValueError("Initial mass must be positive.")
    C = (5120.0 * math.pi * G**2) / (hbar * c**4)
    return (C / g_eff) * (M0_kg**3)


# -----------------------------------------------------------------------------
# IR/DCT suppression toward a remnant
# -----------------------------------------------------------------------------

def suppression_smooth(M_kg: float, M_rem_kg: float, p: float = 4.0) -> float:
    """
    Smooth, dimensionless suppression S(M; M_rem) ∈ [0,1] that enforces stall.

        S = clip( (1 − (M_rem / M))^p , 0, 1 ),   for M > M_rem ;  S = 0 otherwise.

    Parameters
    ----------
    M_kg : float
        Current mass [kg].
    M_rem_kg : float
        Remnant mass [kg] at which evaporation stalls (IR snap/NPR end-state).
    p : float, default 4.0
        Smoothness exponent; larger p makes a steeper cutoff near M_rem.

    Returns
    -------
    float
        Suppression factor S ∈ [0,1].

    Rationale
    ---------
    We want S→1 when M ≫ M_rem (Hawking regime) and S→0 smoothly as M→M_rem^+,
    ensuring numerically stable decay to a finite mass without oscillation.
    """
    if M_kg <= M_rem_kg:
        return 0.0
    x = 1.0 - (M_rem_kg / M_kg)
    return max(0.0, min(1.0, x ** p))


def dMdt_suppressed(M_kg: float, M_rem_kg: float, g_eff: float = 1.0, p: float = 4.0) -> float:
    """
    IR-suppressed mass-loss rate:

        dM/dt |_IR = S(M; M_rem) * dM/dt |_Hawking,   with S from `suppression_smooth`.

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    M_rem_kg : float
        Remnant mass [kg].
    g_eff : float, default 1.0
        Effective emissivity factor.
    p : float, default 4.0
        Suppression exponent.

    Returns
    -------
    float
        dM/dt [kg/s] (≤ 0); equals 0 for M ≤ M_rem.
    """
    S = suppression_smooth(M_kg, M_rem_kg, p=p)
    return S * dMdt_hawking(M_kg, g_eff=g_eff)


# -----------------------------------------------------------------------------
# Simple ODE integrator (explicit RK4 with step control)
# -----------------------------------------------------------------------------

def integrate_evaporation_suppressed(
    M0_kg: float,
    M_rem_kg: float,
    *,
    g_eff: float = 1.0,
    p: float = 4.0,
    t_max: float | None = None,
    dt: float = 1.0,
    max_steps: int = 2_000_000,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Integrate dM/dt = S(M) dM/dt |_Hawking from M(0)=M0 to stall at M_rem.

    Parameters
    ----------
    M0_kg : float
        Initial mass [kg].
    M_rem_kg : float
        Remnant mass [kg]. Integration stops when M <= M_rem_kg (stall).
    g_eff : float, default 1.0
        Effective emissivity factor (dimensionless).
    p : float, default 4.0
        Suppression exponent for S(M).
    t_max : float, optional
        Hard cap on integration time [s]. If None, integrate until stall or max_steps.
    dt : float, default 1.0
        Initial time step [s]. A simple adaptive scheme shrinks dt if step would
        undershoot the remnant or if |ΔM| is too large.
    max_steps : int, default 2_000_000
        Safety cap on number of steps.

    Returns
    -------
    (t, M) : (ndarray, ndarray)
        Time array [s] and mass history [kg].

    Notes
    -----
    - Uses explicit **RK4** with a crude step controller:
      halves dt if a step would cross M_rem; doubles dt slowly when safe.
    - For astrophysical masses the timescales are enormous; choose dt accordingly.
    - Requires NumPy; raises if unavailable.
    """
    if not _HAS_NUMPY:  # pragma: no cover
        raise ImportError("integrate_evaporation_suppressed requires NumPy.")
    if not (M0_kg > 0.0 and M_rem_kg > 0.0 and M0_kg > M_rem_kg):
        raise ValueError("Require M0 > M_rem > 0.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    # Buffers (will grow; for long runs you might preallocate/chunk)
    t: List[float] = [0.0]
    M: List[float] = [M0_kg]

    def f(m: float) -> float:
        return dMdt_suppressed(m, M_rem_kg, g_eff=g_eff, p=p)

    steps = 0
    time = 0.0
    h = dt
    while steps < max_steps:
        m = M[-1]
        if m <= M_rem_kg:
            M[-1] = M_rem_kg
            break
        if t_max is not None and time >= t_max:
            break
        # RK4 proposal
        k1 = f(m)
        k2 = f(m + 0.5 * h * k1)
        k3 = f(m + 0.5 * h * k2)
        k4 = f(m + h * k3)
        dm = (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        m_proposed = m + dm
        # If we would cross below M_rem, shrink step
        if m_proposed <= M_rem_kg:
            h *= 0.5
            if h < 1e-30:
                # Just clamp and stop
                M.append(M_rem_kg)
                t.append(time + h)
                break
            continue
        # If mass drop is "too large", shrink step (heuristic)
        if abs(dm) > 0.05 * (m - M_rem_kg):
            h *= 0.5
            continue
        # Accept step
        time += h
        M.append(m_proposed)
        t.append(time)
        steps += 1
        # Gently increase step if very small change
        if abs(dm) < 1e-6 * (m - M_rem_kg):
            h = min(h * 1.2, dt * 10.0)

    return np.asarray(t, dtype=float), np.asarray(M, dtype=float)


# -----------------------------------------------------------------------------
# Convenience: time-to-stall (numerical), wrapper
# -----------------------------------------------------------------------------

def time_to_stall(
    M0_kg: float,
    M_rem_kg: float,
    *,
    g_eff: float = 1.0,
    p: float = 4.0,
    dt: float = 1.0,
    t_max: float | None = None,
) -> float:
    """
    Convenience wrapper returning the total time until M(t) reaches M_rem (stall).

    Parameters
    ----------
    M0_kg : float
        Initial mass [kg].
    M_rem_kg : float
        Remnant mass [kg].
    g_eff : float, default 1.0
        Effective emissivity factor.
    p : float, default 4.0
        Suppression exponent.
    dt : float, default 1.0
        Initial step [s] for the integrator.
    t_max : float, optional
        Optional cap [s].

    Returns
    -------
    float
        Time to stall [s].
    """
    t, M = integrate_evaporation_suppressed(
        M0_kg, M_rem_kg, g_eff=g_eff, p=p, dt=dt, t_max=t_max
    )
    return float(t[-1]) if len(t) else 0.0


__all__ = [
    "hawking_temperature",
    "hawking_power",
    "dMdt_hawking",
    "lifetime_hawking_closedform",
    "suppression_smooth",
    "dMdt_suppressed",
    "integrate_evaporation_suppressed",
    "time_to_stall",
]
