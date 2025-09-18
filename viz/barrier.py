# qgledger/viz/echoes.py
"""
Echo visualization: time series and magnitude spectra.

Notes
-----
- No seaborn. No explicit colors. One chart per function.
- You can pass precomputed spectra (f, |S|) or let the function call
  qgledger.echoes.spectrum for you from a time series.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..echoes import spectrum as _spectrum


def plot_time_series(t: np.ndarray, s: np.ndarray, *, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a real-valued time series.

    Parameters
    ----------
    t : ndarray
        Time [s].
    s : ndarray
        Signal (arbitrary units).
    title : str, optional
        Title for the axis.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude [a.u.]")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls=":")
    return fig, ax


def plot_spectrum(
    x: np.ndarray,
    fs: float,
    *,
    window: str = "hann",
    zero_pad: int = 4,
    fmax: Optional[float] = None,
    comb_freqs: Optional[Iterable[float]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compute and plot the magnitude spectrum |X(f)| of a real signal.

    Parameters
    ----------
    x : ndarray
        Time series (real-valued).
    fs : float
        Sampling rate [Hz].
    window : {"hann","hamming","rect"}
        Window type for spectral estimate.
    zero_pad : int
        Zero-padding multiple.
    fmax : float, optional
        Limit x-axis to [0, fmax].
    comb_freqs : iterable of float, optional
        Optional list of expected comb lines to annotate (vertical ticks).
    title : str, optional
        Title for the plot.

    Returns
    -------
    (fig, ax)
    """
    f, mag = _spectrum(x, fs, window=window, zero_pad=zero_pad, one_sided=True)
    fig, ax = plt.subplots()
    ax.plot(f, mag)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|X(f)| [a.u.]")
    if title:
        ax.set_title(title)
    if fmax is not None:
        ax.set_xlim(0.0, float(fmax))
    if comb_freqs is not None:
        for fr in comb_freqs:
            ax.axvline(float(fr), linestyle="--", linewidth=1.0)
    ax.grid(True, which="both", ls=":")
    return fig, ax
