# qgledger/viz/cosmology.py
"""
Cosmology viz: FRW invariant 𝓘_FRW(a) over scale factor a ∈ [a_min, a_max].
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..cosmology import invariant_I_frw, I_star


def plot_I_FRW(
    a_min: float,
    a_max: float,
    *,
    H0: float,
    Om: float,
    Or: float = 0.0,
    Ode: float = 0.0,
    Ok: float | None = None,
    w: float = -1.0,
    N: int = 400,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 𝓘_FRW(a) and the universal threshold 𝓘_* across a range of scale factors.

    Parameters
    ----------
    a_min, a_max : float
        Range of scale factor (must satisfy 0 < a_min < a_max).
    H0, Om, Or, Ode, Ok, w : as in qgledger.cosmology.hubble_H
    N : int
        Number of sampling points.

    Returns
    -------
    (fig, ax)
    """
    if not (a_min > 0.0 and a_max > a_min):
        raise ValueError("Require 0 < a_min < a_max.")
    a = np.linspace(a_min, a_max, int(N))
    I = np.array([invariant_I_frw(ai, H0=H0, Om=Om, Or=Or, Ode=Ode, Ok=Ok, w=w) for ai in a])

    fig, ax = plt.subplots()
    ax.plot(a, I)
    ax.axhline(I_star, linestyle="--", linewidth=1.0)
    ax.set_xlabel("scale factor a [dimensionless]")
    ax.set_ylabel("𝓘_FRW(a) [dimensionless]")
    ax.set_title("FRW invariant vs scale factor")
    ax.grid(True, which="both", ls=":")
    return fig, ax
