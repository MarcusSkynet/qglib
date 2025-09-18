# qgledger/echo_filter.py
"""
qgledger.echo_filter
====================
Frequency-domain echo synthesis with an **arbitrary reflection** ℛ(ω).

Given a seed signal x(t), a roundtrip delay Δt, and a complex reflection ℛ(ω),
the total transfer function for N echoes is
    H_N(ω) = 1 + ℛ e^{-iωΔt} + (ℛ e^{-iωΔt})^2 + ... + (ℛ e^{-iωΔt})^N
           = (1 - (ℛ e^{-iωΔt})^{N+1}) / (1 - ℛ e^{-iωΔt}) .
For |ℛ e^{-iωΔt}|<1 this tends to the geometric sum; with |ℛ|=1 (NPR) the
finite-N form is well-defined and numerically stable.

API
---
- apply_echo_filter_fd(x, fs, delta_t, R_of_omega, n_echoes): filtered time series.
- comb_frequencies_from_dt(delta_t, n): convenience (same as expected_comb_frequencies).
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

TwoPi = 2.0 * np.pi

def _rfft_freqs(fs: float, n: int) -> np.ndarray:
    return np.fft.rfftfreq(n, d=1.0/fs) * TwoPi  # angular ω grid

def apply_echo_filter_fd(
    x: np.ndarray,
    fs: float,
    delta_t: float,
    R_of_omega: Callable[[np.ndarray], np.ndarray],
    n_echoes: int = 6,
) -> np.ndarray:
    """
    Apply the echo filter in the frequency domain using a supplied ℛ(ω).

    Parameters
    ----------
    x : ndarray
        Real time series.
    fs : float
        Sampling rate [Hz].
    delta_t : float
        Roundtrip delay Δt [s].
    R_of_omega : callable
        Vectorized function taking ω-array [rad/s] and returning ℛ(ω) (complex).
    n_echoes : int
        Number of echoes (terms) to include, N ≥ 0.

    Returns
    -------
    y : ndarray
        Real time series after echo filter.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    # FFT (one-sided for real input)
    X = np.fft.rfft(x)
    omega = _rfft_freqs(fs, n)
    R = R_of_omega(omega)
    # geometric sum H_N(ω)
    z = R * np.exp(-1j * omega * float(delta_t))
    # Avoid division by zero when z≈1 by using finite-sum form explicitly
    # H_N = sum_{k=0}^N z^k
    # Use stable Horner evaluation
    H = np.ones_like(z, dtype=np.complex128)
    zk = np.ones_like(z, dtype=np.complex128)
    for _ in range(n_echoes):
        zk *= z
        H += zk
    Y = H * X
    y = np.fft.irfft(Y, n=n)
    return y

def comb_frequencies_from_dt(delta_t: float, n: int) -> np.ndarray:
    """
    Return the first n comb-line frequencies f_k = k / Δt  (k=1..n) in Hz.
    """
    k = np.arange(1, n+1, dtype=float)
    return k / float(delta_t)
