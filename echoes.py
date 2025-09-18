# qgledger/echoes.py
"""
qgledger.echoes
===============
Time-domain echo synthesis and simple spectral analysis for the IR Quantum Gravity
(ledger/NPR) framework.

This module provides *detector-facing* utilities to build toy echo waveforms and their
Fourier spectra from a small set of physical/phenomenological parameters. It pairs with
:mod:`qgledger.tortoise` for echo delays derived from geometry.

Design goals
------------
- Minimal dependencies (NumPy only).
- Clear units: time [s], frequency [Hz], sampling rate fs [Hz].
- Stable numerics: optional windowing and zero-padding for clean spectral peaks.
- Honest scope: this is a pedagogical *toy* front end, not a data-analysis pipeline.

Core ideas
----------
Let s₀(t) be a base transient (e.g., a damped ringdown). An echo train models repeated,
delayed, and attenuated copies:
    s(t) = s₀(t) + R e^{iφ} s₀(t-Δt) + R² e^{i2φ} s₀(t-2Δt) + …  (N echoes)
where Δt is the roundtrip delay from the photon-sphere barrier to an effective inner wall
(outside-horizon ECO-style), R∈[0,1) the amplitude reflectivity per roundtrip, and φ an
accumulated phase shift per bounce. In practice we work with *real* waveforms; φ controls
the sign pattern (cos/sin mix) via a carrier.

Quick start
-----------
>>> from qgledger.echoes import ringdown, echo_train, spectrum
>>> import numpy as np
>>> fs, T = 4096.0, 4.0
>>> t = np.arange(0, T, 1/fs)
>>> s0 = ringdown(t, f0=150.0, tau=0.15)             # base damped sinusoid
>>> s  = echo_train(t, s0, delay=0.12, R=0.5, phi=0, N=6)
>>> f, S = spectrum(s, fs, window='hann', zero_pad=4)

See also
--------
- :func:`qgledger.tortoise.echo_delay_seconds` — get Δt from mass and ε_eff.
- :func:`expected_comb_frequencies` — theoretical line locations n / Δt.
"""

from __future__ import annotations

from typing import Tuple, Literal
import math

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise ImportError("qgledger.echoes requires NumPy.") from exc


# -----------------------------------------------------------------------------
# Windows (implemented locally to avoid scipy dependency)
# -----------------------------------------------------------------------------

def _window(N: int, kind: Literal["hann", "hamming", "rect"] = "hann") -> np.ndarray:
    """
    Generate a real window of length N.

    Parameters
    ----------
    N : int
        Window length.
    kind : {"hann","hamming","rect"}
        Window type; 'hann' is a good general-purpose choice.

    Returns
    -------
    ndarray
        Length-N window (float64).
    """
    if N <= 0:
        return np.zeros(0, dtype=float)
    n = np.arange(N, dtype=float)
    if kind == "rect":
        return np.ones(N, dtype=float)
    if kind == "hann":
        # w[n] = 0.5 - 0.5 cos(2π n/(N-1)), N>1
        if N == 1:
            return np.ones(1)
        return 0.5 - 0.5 * np.cos(2.0 * math.pi * n / (N - 1))
    if kind == "hamming":
        if N == 1:
            return np.ones(1)
        alpha = 0.54
        beta = 1.0 - alpha
        return alpha - beta * np.cos(2.0 * math.pi * n / (N - 1))
    raise ValueError("Unknown window kind. Choose 'hann', 'hamming', or 'rect'.")


# -----------------------------------------------------------------------------
# Base transients
# -----------------------------------------------------------------------------

def ringdown(t: np.ndarray, f0: float, tau: float, phase: float = 0.0, envelope: Literal["exp","gauss"] = "exp") -> np.ndarray:
    """
    Damped sinusoidal ringdown.

    Parameters
    ----------
    t : ndarray
        Time array [s].
    f0 : float
        Carrier frequency [Hz].
    tau : float
        Damping time constant [s] for EXP envelope; for GAUSS, σ = tau.
    phase : float, default 0
        Initial phase [rad].
    envelope : {"exp","gauss"}, default "exp"
        EXP: e^{-t/tau} u(t).  GAUSS: e^{-(t-t0)^2/(2 σ^2)} centered at t0 = 3σ.

    Returns
    -------
    ndarray
        Real-valued waveform s0(t).

    Notes
    -----
    This is a pedagogical base; you can replace it with any transient (chirps, etc).
    """
    t = np.asarray(t, dtype=float)
    carrier = np.cos(2.0 * math.pi * f0 * t + phase)
    if envelope == "exp":
        env = np.exp(-np.clip(t, 0.0, None) / tau)
    elif envelope == "gauss":
        sigma = tau
        t0 = 3.0 * sigma
        env = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    else:
        raise ValueError("envelope must be 'exp' or 'gauss'.")
    return env * carrier


def gaussian_pulse(t: np.ndarray, t0: float, sigma: float, amp: float = 1.0) -> np.ndarray:
    """
    Simple Gaussian pulse A exp(- (t-t0)^2 / (2 σ^2)).

    Parameters
    ----------
    t : ndarray
        Time array [s].
    t0 : float
        Pulse center [s].
    sigma : float
        Pulse width [s].
    amp : float, default 1
        Amplitude.

    Returns
    -------
    ndarray
        Real-valued pulse.
    """
    t = np.asarray(t, dtype=float)
    return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


# -----------------------------------------------------------------------------
# Echo synthesis
# -----------------------------------------------------------------------------

def echo_train(
    t: np.ndarray,
    base: np.ndarray,
    *,
    delay: float,
    R: float = 0.5,
    phi: float = 0.0,
    N: int = 5,
    taper: float = 1.0,
) -> np.ndarray:
    """
    Build an echo train from a base transient by delayed, attenuated copies.

    s(t) = Σ_{k=0}^{N} R^k cos(k φ) base(t - k Δt)   (real projection)

    Parameters
    ----------
    t : ndarray
        Time array [s], uniformly sampled.
    base : ndarray
        Base waveform s₀(t) aligned to the same t-grid (e.g., from `ringdown`).
    delay : float
        Echo spacing Δt [s].
    R : float, default 0.5
        Amplitude reflectivity per roundtrip, 0 ≤ R < 1.
    phi : float, default 0
        Phase shift per echo [rad]. Real projection uses cos(k φ).
    N : int, default 5
        Number of echoes (in addition to the prompt k=0 term).
    taper : float, default 1.0
        Optional additional multiplicative decay per echo (e.g., exp(-k/τ_e)), here
        provided as a base for a power: weight_k ← weight_k * taper^k.

    Returns
    -------
    ndarray
        Echo waveform on the same grid as `t`.

    Implementation details
    ----------------------
    Uses simple circular shifts on the discrete grid; for non-integer sample delays
    the function performs linear interpolation.
    """
    t = np.asarray(t, dtype=float)
    s0 = np.asarray(base, dtype=float)
    if t.shape != s0.shape:
        raise ValueError("t and base must have the same shape.")
    if delay <= 0.0:
        raise ValueError("delay must be positive.")
    if not (0.0 <= R < 1.0):
        raise ValueError("R must be in [0,1).")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("t must be uniformly sampled.")

    def shift_linear(x: np.ndarray, samples: float) -> np.ndarray:
        # Fractional index shift with linear interpolation; positive = shift right (delay)
        n = len(x)
        i0 = np.arange(n) - samples
        i0_floor = np.floor(i0).astype(int)
        frac = i0 - i0_floor
        i1 = i0_floor + 1
        # clip for boundary; zeros beyond edges
        w0 = np.where((i0_floor >= 0) & (i0_floor < n), 1.0 - frac, 0.0)
        w1 = np.where((i1 >= 0) & (i1 < n), frac, 0.0)
        y0 = np.where((i0_floor >= 0) & (i0_floor < n), x[i0_floor], 0.0)
        y1 = np.where((i1 >= 0) & (i1 < n), x[i1], 0.0)
        return w0 * y0 + w1 * y1

    s = np.zeros_like(s0)
    samp_delay = delay / dt
    for k in range(N + 1):
        weight = (R ** k) * (taper ** k) * math.cos(k * phi)
        if k == 0:
            s += weight * s0
        else:
            s += weight * shift_linear(s0, k * samp_delay)
    return s


# -----------------------------------------------------------------------------
# Spectral analysis
# -----------------------------------------------------------------------------

def spectrum(
    x: np.ndarray,
    fs: float,
    *,
    window: Literal["hann","hamming","rect"] = "hann",
    zero_pad: int = 4,
    one_sided: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Magnitude spectrum |X(f)| of a real time series.

    Parameters
    ----------
    x : ndarray
        Time series (real).
    fs : float
        Sampling rate [Hz].
    window : {"hann","hamming","rect"}, default "hann"
        Window applied before FFT to reduce leakage.
    zero_pad : int, default 4
        Zero-padding multiple (N_fft = zero_pad * N). Use ≥ 1.
    one_sided : bool, default True
        Return only non-negative frequencies for real input.

    Returns
    -------
    (f, mag) : (ndarray, ndarray)
        Frequencies [Hz] and magnitude spectrum (same units as x).

    Notes
    -----
    This is a simple pedagogical spectrum; no PSD normalization is applied.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return np.zeros(0), np.zeros(0)
    if fs <= 0.0:
        raise ValueError("fs must be positive.")
    if zero_pad < 1:
        zero_pad = 1
    w = _window(N, window)
    xw = x * w
    Nfft = int(zero_pad * N)
    X = np.fft.rfft(xw, n=Nfft) if one_sided else np.fft.fft(xw, n=Nfft)
    mag = np.abs(X)
    if one_sided:
        f = np.fft.rfftfreq(Nfft, d=1.0 / fs)
    else:
        f = np.fft.fftfreq(Nfft, d=1.0 / fs)
    return f, mag


def expected_comb_frequencies(delay: float, N_peaks: int = 10) -> np.ndarray:
    """
    Expected spectral line locations for an echo spacing Δt = delay.

    f_n = n / Δt,  n = 1..N_peaks.

    Parameters
    ----------
    delay : float
        Echo spacing Δt [s], delay > 0.
    N_peaks : int, default 10
        Number of harmonic peaks to list.

    Returns
    -------
    ndarray
        Frequencies [Hz] at which peaks are expected (idealized).
    """
    if delay <= 0.0:
        raise ValueError("delay must be positive.")
    n = np.arange(1, N_peaks + 1, dtype=float)
    return n / delay


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------

def echo_train_from_ringdown(
    t: np.ndarray,
    *,
    f0: float,
    tau: float,
    delay: float,
    R: float = 0.5,
    phi: float = 0.0,
    N: int = 5,
    envelope: Literal["exp","gauss"] = "exp",
) -> np.ndarray:
    """
    Build an echo train where the base is a ringdown transient.

    Parameters
    ----------
    t : ndarray
        Time grid [s].
    f0 : float
        Carrier frequency [Hz].
    tau : float
        Damping (EXP) or σ (GAUSS) [s].
    delay : float
        Echo spacing Δt [s].
    R : float, default 0.5
        Reflectivity per roundtrip.
    phi : float, default 0
        Phase shift per echo [rad].
    N : int, default 5
        Number of echoes.
    envelope : {"exp","gauss"}, default "exp"
        Base envelope type.

    Returns
    -------
    ndarray
        Echo waveform.
    """
    base = ringdown(t, f0=f0, tau=tau, envelope=envelope)
    return echo_train(t, base, delay=delay, R=R, phi=phi, N=N)


__all__ = [
    "ringdown",
    "gaussian_pulse",
    "echo_train",
    "spectrum",
    "expected_comb_frequencies",
    "echo_train_from_ringdown",
]
