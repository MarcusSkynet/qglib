# qgledger/cavity.py
"""
qgledger.cavity
===============
Map the *interior* ledger placement to an *exterior* effective inner boundary
for echo calculations.

Goal
----
Given a black-hole mass M and the **ledger radius** r_L(M) (which lies *inside*
the event horizon at r_s), construct an **effective outside offset**
    r_eff = r_s (1 + ε_eff),  ε_eff > 0,
so that the **exterior** tortoise-gap up to the photon sphere,
    Δr_*^ext = r_*(r_ph) − r_*(r_eff),
matches a chosen **interior proxy** for the cavity length implied by the ledger.

Why?
----
Detector-facing echo models (and much of the ECO literature) work entirely
*outside* the horizon with a reflective boundary at r = r_s (1+ε). In the IR
ledger/NPR framework, the real boundary is the ledger at r_L **inside** the
horizon. This module supplies a clean, documented mapping between the two.

Caveat
------
Inside the horizon the coordinate r is timelike; there is no literal “standing
wave cavity.” Here we define a *proxy* target gap using a **real interior
tortoise** (see `rstar_interior_real`) in a way that preserves the dominant
near-horizon logarithms and yields a stable, monotonic mapping ε_eff(M).

API
---
- rstar_exterior(r, r_s):      standard tortoise for r>r_s.
- rstar_interior_real(r, r_s): real branch for r<r_s (proxy).
- target_gap_from_ledger(M):   Δr_*^target = r_*^ext(r_ref) − r_*^in(r_L).
- epsilon_from_ledger(M):      ε_eff solving Δr_*^ext(ε) = Δr_*^target.
- roundtrip_delay_from_ledger(M): Δt = 2 Δr_*^target / c (echo delay).

Defaults
--------
- r_ref = r_ph(M) = 1.5 r_s (photon sphere).
- “mixed” model includes both linear and logarithmic terms; a “logonly” option
  is provided for analytical control (dominant terms near the horizon).

All functions use **SI units**. NumPy is not required.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from .tortoise import (
    tortoise_coordinate as rstar_exterior,
    photon_sphere_radius,
    schwarzschild_radius,
    epsilon_for_target_gap,
    echo_delay_from_target_gap,
)
from .geom import ledger_radius_schwarzschild
from .constants import c


# ---------------------------------------------------------------------------
# Interior/exterior tortoise helpers
# ---------------------------------------------------------------------------

def rstar_interior_real(r: float, r_s: float) -> float:
    """
    Real-branch *interior* tortoise proxy (for r < r_s):

        r_*^in(r) = r + r_s ln(1 − r/r_s),   r < r_s.

    Notes
    -----
    - The standard exterior tortoise is r_* = r + r_s ln(r/r_s − 1) for r > r_s.
    - Inside the horizon that log is complex; here we take the *real* branch with
      ln(1 − r/r_s), which → −∞ as r → r_s^−, mirroring the exterior divergence.
    - This proxy is used purely to *define* a target gap; it is not a PDE inside.
    """
    if not (0.0 < r < r_s):
        raise ValueError("rstar_interior_real requires 0 < r < r_s.")
    return r + r_s * math.log(1.0 - (r / r_s))


# ---------------------------------------------------------------------------
# Target gap from ledger placement
# ---------------------------------------------------------------------------

def target_gap_from_ledger(
    M_kg: float,
    *,
    r_L: Optional[float] = None,
    r_ref: Optional[float] = None,
    model: Literal["mixed", "logonly"] = "mixed",
) -> float:
    """
    Build a **target tortoise gap** Δr_*^target from the interior ledger to an
    exterior reference radius (default: photon sphere).

        Δr_*^target = r_*^ext(r_ref) − r_*^in(r_L).

    Parameters
    ----------
    M_kg : float
        Black-hole mass [kg].
    r_L : float, optional
        Ledger radius [m]. If None, computed from `ledger_radius_schwarzschild(M)`.
    r_ref : float, optional
        Exterior reference radius [m], default r_ph(M).
    model : {"mixed","logonly"}
        - "mixed"  : use full forms r + r_s ln(...), interior & exterior.
        - "logonly": use only the logarithmic parts (dominant near the horizon).

    Returns
    -------
    float
        Δr_*^target in meters.

    Raises
    ------
    ValueError for inconsistent radii.

    Discussion
    ----------
    The “mixed” model preserves both linear and log terms:
        Δr_*^target = [r_ref + r_s ln(r_ref/r_s − 1)] − [r_L + r_s ln(1 − r_L/r_s)].
    The “logonly” model keeps only r_s [ ln(…) − ln(…) ], which is often a good
    approximation when both radii are near r_s; here r_L ≈ 0.6005 r_s so linear
    terms contribute O(r_s).
    """
    r_s = schwarzschild_radius(M_kg)
    if r_L is None:
        r_L = ledger_radius_schwarzschild(M_kg)
    if not (0.0 < r_L < r_s):
        raise ValueError("Ledger radius must satisfy 0 < r_L < r_s.")
    if r_ref is None:
        r_ref = photon_sphere_radius(M_kg)
    if not (r_ref > r_s):
        raise ValueError("Reference radius must satisfy r_ref > r_s.")

    if model == "mixed":
        rstar_out = rstar_exterior(r_ref, r_s)
        rstar_in = rstar_interior_real(r_L, r_s)
        return rstar_out - rstar_in
    elif model == "logonly":
        term_out = math.log((r_ref / r_s) - 1.0)
        term_in = math.log(1.0 - (r_L / r_s))
        return r_s * (term_out - term_in)
    else:
        raise ValueError("model must be 'mixed' or 'logonly'.")


# ---------------------------------------------------------------------------
# Map to ε_eff and echo delay
# ---------------------------------------------------------------------------

def epsilon_from_ledger(
    M_kg: float,
    *,
    r_L: Optional[float] = None,
    r_ref: Optional[float] = None,
    model: Literal["mixed", "logonly"] = "mixed",
) -> float:
    """
    Compute the **effective outside offset** ε_eff such that

        r_*(r_ref) − r_*(r_s(1+ε_eff)) = Δr_*^target(M),

    where Δr_*^target is constructed from the interior ledger position.

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    r_L : float, optional
        Ledger radius [m]; if None, computed from M.
    r_ref : float, optional
        Exterior reference radius [m]; if None, uses r_ph(M).
    model : {"mixed","logonly"}
        Choice for Δr_*^target construction.

    Returns
    -------
    float
        ε_eff > 0 (dimensionless).
    """
    r_s = schwarzschild_radius(M_kg)
    if r_ref is None:
        r_ref = photon_sphere_radius(M_kg)
    gap = target_gap_from_ledger(M_kg, r_L=r_L, r_ref=r_ref, model=model)
    return epsilon_for_target_gap(r_s, gap, r_ref=r_ref)


def roundtrip_delay_from_ledger(
    M_kg: float,
    *,
    r_L: Optional[float] = None,
    r_ref: Optional[float] = None,
    model: Literal["mixed", "logonly"] = "mixed",
) -> float:
    """
    Echo **roundtrip delay** implied by the interior ledger mapping:

        Δt = 2 Δr_*^target / c.

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    r_L : float, optional
        Ledger radius [m]; if None, computed from M.
    r_ref : float, optional
        Exterior reference radius [m]; if None, r_ph(M).
    model : {"mixed","logonly"}
        Choice for Δr_*^target construction.

    Returns
    -------
    float
        Δt in seconds.
    """
    gap = target_gap_from_ledger(M_kg, r_L=r_L, r_ref=r_ref, model=model)
    return echo_delay_from_target_gap(gap)


__all__ = [
    "rstar_interior_real",
    "target_gap_from_ledger",
    "epsilon_from_ledger",
    "roundtrip_delay_from_ledger",
]
