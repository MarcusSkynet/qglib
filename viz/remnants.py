# qgledger/viz/remnants.py
"""
Evaporation history visualization: mass vs time with IR/DCT suppression.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..remnants import integrate_evaporation_suppressed, lifetime_hawking_closedform


def plot_evaporation_history(
    M0_kg: float,
    M_rem_kg: float,
    *,
    g_eff: float = 1.0,
    p: float = 4.0,
    dt: float = 1.0,
    t_max: Optional[float] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Integrate and plot M(t) from M0 to stall at M_rem.

    Parameters
    ----------
    M0_kg : float
        Initial mass [kg].
    M_rem_kg : float
        Remnant mass [kg].
    g_eff : float
        Effective emissivity.
    p : float
        Suppression exponent.
    dt : float
        Initial integrator step [s].
    t_max : float, optional
        Cap on integration time [s].
    title : str, optional
        Title for the plot.

    Returns
    -------
    (fig, ax)
    """
    t, M = integrate_evaporation_suppressed(M0_kg, M_rem_kg, g_eff=g_eff, p=p, dt=dt, t_max=t_max)
    fig, ax = plt.subplots()
    ax.plot(t, M)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mass [kg]")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":")
    return fig, ax
