# qgledger/viz/tortoise.py
"""
Exterior tortoise visualization r_*(r) for Schwarzschild, with optional
effective wall r_eff = r_s (1 + ε) and photon sphere marker.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..geom import schwarzschild_radius, photon_sphere_radius
from ..tortoise import tortoise_coordinate


def plot_rstar_exterior(
    M_kg: float,
    *,
    eps_eff: Optional[float] = None,
    rmax_factor: float = 6.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot r_*(r) for r ∈ [r_s(1+1e-6), rmax], marking r_s, r_ph, and optional r_eff.

    Parameters
    ----------
    M_kg : float
        Mass [kg].
    eps_eff : float, optional
        If provided, mark r_eff = r_s (1 + eps_eff).
    rmax_factor : float, default 6.0
        r_max = rmax_factor * (G M / c^2).

    Returns
    -------
    (fig, ax)
    """
    r_s = schwarzschild_radius(M_kg)
    r_ph = photon_sphere_radius(M_kg)
    r_min = r_s * (1.0 + 1e-6)
    r_max = r_s * rmax_factor

    r = np.linspace(r_min, r_max, 1000)
    rstar = np.array([tortoise_coordinate(ri, r_s) for ri in r])

    fig, ax = plt.subplots()
    ax.plot(r, rstar)
    ax.set_xlabel("radius r [m]")
    ax.set_ylabel("tortoise r_* [m]")
    ax.set_title("Exterior tortoise coordinate r_*(r)")

    ax.axvline(r_s, linestyle="--", linewidth=1.0)
    ax.text(r_s, rstar.min(), "r_s", rotation=90, va="bottom", ha="right")
    ax.axvline(r_ph, linestyle="--", linewidth=1.0)
    ax.text(r_ph, 0.5 * (rstar.min() + rstar.max()), "r_ph", rotation=90, va="center", ha="left")

    if eps_eff is not None and eps_eff > 0.0:
        r_eff = r_s * (1.0 + eps_eff)
        ax.axvline(r_eff, linestyle=":", linewidth=1.0)
        ax.text(r_eff, 0.75 * (rstar.min() + rstar.max()), "r_eff", rotation=90, va="center", ha="left")

    ax.grid(True, which="both", ls=":")
    return fig, ax
