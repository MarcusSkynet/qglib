# qgledger/utils.py
"""
qgledger.utils
==============
Shared numerics, unit helpers, and small general-purpose utilities used across
the IR Quantum Gravity (ledger/NPR) library.

Design goals
------------
- **Dependency-light:** pure Python with optional NumPy acceleration where helpful.
- **Clear units:** small helpers to convert between SI and geometric frequencies.
- **Robust numerics:** bracketing root-finder, Newton/Halley for 1D, and
  stable finite-difference derivatives with sensible defaults.

Contents
--------
Numerics
    - clamp(x, lo, hi)
    - is_close(a, b, rtol=..., atol=...)
    - relerr(a, b)
    - derivative_central(f, x, h=..., order=1)
    - derivative_second_central(f, x, h=...)
    - brentq(func, a, b, tol=..., max_iter=...)
    - newton_1d(func, x0, dfunc=None, tol=..., max_iter=..., bracket=None)

Units & conversions
    - mass_geometric_length(M_kg)  [re-export alias to keep imports local]
    - omega_geom_from_f_hz(f_hz)   (ω = 2π f / c)  [1/m]
    - f_hz_from_omega_geom(omega)  (f = c ω / 2π)  [Hz]
    - time_from_length(L) = L / c  [s]
    - length_from_time(t) = c t    [m]

Safety helpers
    - safe_log(x, minval)
    - safe_log1p(x, minval)

Notes
-----
These utilities are kept modest in scope. Anything problem-specific (e.g.
tortoise inversions, barrier curvature) lives in the relevant modules.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import math

try:
    import numpy as np
    _HAS_NUMPY = True
    ArrayLike = Union[float, int, "np.ndarray"]
except Exception:  # pragma: no cover
    _HAS_NUMPY = False
    np = None  # type: ignore
    ArrayLike = float  # type: ignore

from .constants import c, RTOL, ATOL
from .geom import mass_geometric_length as _mass_geometric_length  # local alias


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp x into [lo, hi].

    Raises
    ------
    ValueError if lo > hi.
    """
    if lo > hi:
        raise ValueError("clamp bounds must satisfy lo <= hi")
    return min(max(x, lo), hi)


def is_close(a: float, b: float, *, rtol: float = RTOL, atol: float = ATOL) -> bool:
    """
    True if |a - b| ≤ atol + rtol · |b|.
    """
    return abs(a - b) <= (atol + rtol * abs(b))


def relerr(a: float, b: float) -> float:
    """
    Relative error |a - b| / max(|a|, |b|, tiny).
    """
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def safe_log(x: float, minval: float = 1e-300) -> float:
    """
    log(max(x, minval)) to avoid -inf or NaN for x ≤ 0.
    """
    return math.log(max(x, minval))


def safe_log1p(x: float, minval: float = -0.999999999999) -> float:
    """
    log1p(max(x, minval)) to avoid domain errors near -1.
    """
    return math.log1p(max(x, minval))


# -----------------------------------------------------------------------------
# Finite-difference derivatives (scalar f)
# -----------------------------------------------------------------------------

def derivative_central(f: Callable[[float], float], x: float, h: float = 1e-6, order: int = 1) -> float:
    """
    Central finite-difference derivatives of order 1 or 2 for a scalar function.

    Parameters
    ----------
    f : callable
        Scalar function f(x).
    x : float
        Point at which to evaluate the derivative.
    h : float, default 1e-6
        Step size (absolute). Adjust based on x-scale to balance truncation/roundoff.
    order : {1,2}
        1 for f'(x), 2 for f''(x) using standard 3-point stencil.

    Returns
    -------
    float
        Approximate derivative.

    Notes
    -----
    - 1st derivative: (f(x+h) - f(x-h)) / (2h)
    - 2nd derivative: (f(x+h) - 2 f(x) + f(x-h)) / h^2
    """
    if order == 1:
        return (f(x + h) - f(x - h)) / (2.0 * h)
    if order == 2:
        return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    raise ValueError("order must be 1 or 2")


def derivative_second_central(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """
    Convenience alias for second derivative via 3-point central stencil.
    """
    return derivative_central(f, x, h=h, order=2)


# -----------------------------------------------------------------------------
# Root finding: bracketing Brent and Newton with optional bracket guard
# -----------------------------------------------------------------------------

def brentq(
    func: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """
    Find a root of func in [a,b] with a robust Brent-like method.

    Parameters
    ----------
    func : callable
        Scalar function.
    a, b : float
        Bracket endpoints with opposite signs (func(a) * func(b) < 0).
    tol : float
        Absolute tolerance on x (and relative on f implicitly).
    max_iter : int
        Iteration cap.

    Returns
    -------
    float
        Approximate root.

    Raises
    ------
    ValueError if no sign change or if algorithm fails to converge.

    Implementation
    --------------
    Follows the standard approach mixing bisection, secant, and inverse quadratic
    interpolation. This is a compact implementation intended for moderate accuracy.
    """
    fa = func(a)
    fb = func(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("brentq requires a sign change on [a,b].")

    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb == 0.0:
            return b
        # Ensure |fb| <= |fc|
        if abs(fc) < abs(fb):
            a, b = b, c
            fa, fb = fb, fc
            c, fc = a, fa
        # Convergence check
        tol1 = 2.0 * tol * max(1.0, abs(b))
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1:
            return b
        # Attempt inverse quadratic interpolation or secant
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            min1 = 3.0 * xm * q - abs(tol1 * q)
            min2 = abs(e * q)
            if 2.0 * p < min(min1, min2):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += math.copysign(tol1, xm)
        fb = func(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
    raise ValueError("brentq failed to converge within max_iter.")


def newton_1d(
    func: Callable[[float], float],
    x0: float,
    dfunc: Optional[Callable[[float], float]] = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
    bracket: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Newton (or secant) root-finder with optional bracketing fallback.

    Parameters
    ----------
    func : callable
        Scalar function.
    x0 : float
        Initial guess.
    dfunc : callable, optional
        Derivative f'(x). If None, uses secant updates.
    tol : float
        Absolute tolerance on x.
    max_iter : int
        Iteration cap.
    bracket : (a,b), optional
        If provided and Newton fails (diverges/NaNs), fall back to brentq on [a,b].

    Returns
    -------
    float
        Approximate root.
    """
    x = float(x0)
    fx = func(x)
    if fx == 0.0:
        return x
    x_prev = x + 2.0 * tol
    for _ in range(max_iter):
        if dfunc is not None:
            dfx = dfunc(x)
            if dfx == 0.0 or not math.isfinite(dfx):
                break
            step = fx / dfx
        else:
            # Secant: use last two points (x_prev, x)
            f_prev = func(x_prev)
            denom = (fx - f_prev)
            if denom == 0.0 or not math.isfinite(denom):
                break
            step = fx * (x - x_prev) / denom
        x_prev, x = x, x - step
        fx = func(x)
        if not math.isfinite(x) or not math.isfinite(fx):
            break
        if abs(step) <= tol and abs(fx) <= 1e-12:
            return x
    if bracket is not None:
        a, b = bracket
        return brentq(func, a, b, tol=tol, max_iter=200)
    raise ValueError("newton_1d failed to converge; consider providing a bracket.")


# -----------------------------------------------------------------------------
# Units & conversions
# -----------------------------------------------------------------------------

def mass_geometric_length(M_kg: float) -> float:
    """
    Alias to :func:`qgledger.geom.mass_geometric_length` for convenience.

    Parameters
    ----------
    M_kg : float
        Mass in kilograms.

    Returns
    -------
    float
        M_geo = G M / c^2 in meters.
    """
    return _mass_geometric_length(M_kg)


def omega_geom_from_f_hz(f_hz: float) -> float:
    """
    Convert physical frequency f [Hz] to geometric angular frequency ω [1/m]:

        ω_geom = (2π f) / c.

    Parameters
    ----------
    f_hz : float
        Frequency [Hz].

    Returns
    -------
    float
        ω in [1/m].
    """
    return (2.0 * math.pi * f_hz) / c


def f_hz_from_omega_geom(omega_geom: float) -> float:
    """
    Convert geometric angular frequency ω [1/m] to physical f [Hz]:

        f = (c ω) / (2π).
    """
    return (c * omega_geom) / (2.0 * math.pi)


def time_from_length(L_m: float) -> float:
    """
    Convert length to light-travel time: t = L / c  [s].
    """
    return L_m / c


def length_from_time(t_s: float) -> float:
    """
    Convert time to light-travel length: L = c t  [m].
    """
    return c * t_s


__all__ = [
    # small helpers
    "clamp", "is_close", "relerr", "safe_log", "safe_log1p",
    # derivatives & roots
    "derivative_central", "derivative_second_central", "brentq", "newton_1d",
    # units
    "mass_geometric_length", "omega_geom_from_f_hz", "f_hz_from_omega_geom",
    "time_from_length", "length_from_time",
]
