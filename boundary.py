# qgledger/boundary.py
"""
qgledger.boundary
=================
Action-implied **Robin boundary** at the ledger and the associated complex
reflection coefficient ℛ(ω). This encodes the NPR rule (no longitudinal energy
flow across the ledger between writes) as a **lossless** boundary condition.

Core relation (linear theory)
-----------------------------
A Robin boundary for a master mode ψ_{ℓm}(t,r_*) at the ledger position r_*=r_*^L:
    (∂_{r_*} ψ) - κ_b(ω,ℓ) ψ = 0    at r_* = r_*^L
implies a frequency-domain reflection coefficient
    ℛ(ω) = (iω - κ_b) / (iω + κ_b) .
If κ_b is real and non-negative, |ℛ(ω)| = 1 (unitary, no energy flux).

API
---
- robin_reflection(omega, kappa_b): complex ℛ(ω) for scalar κ_b or callables.
- make_kappa_const(kappa0): returns κ_b(ω,ℓ)=κ0 (frequency-independent).
- make_kappa_drude(kappa0, omega_c): weak dispersion model (illustrative).
- is_unitary_R(R): check |R|≈1 within tolerance.
- phase_of_R(R): principal value phase of ℛ(ω).

Notes
-----
• This is the **IR boundary law** extracted from the ledger worldvolume term in
  the action; between writes κ_b is constant/real (lossless). During a write,
  κ_b becomes time-dependent and supplies the Clausius heat; that non-stationary
  case is not included here (would require time-domain BCs).
"""

from __future__ import annotations

from typing import Callable, Protocol, Union
import numpy as np

Array = np.ndarray
Scalar = Union[float, complex]
OmegaLike = Union[float, Array]

class KappaModel(Protocol):
    def __call__(self, omega: OmegaLike, ell: int | None = None) -> OmegaLike: ...

def robin_reflection(omega: OmegaLike, kappa_b: Union[float, KappaModel], ell: int | None = None) -> Array:
    """
    Complex reflection coefficient ℛ(ω) = (iω - κ_b) / (iω + κ_b).

    Parameters
    ----------
    omega : float or ndarray
        Angular frequency ω [rad/s], can be vector.
    kappa_b : float or callable
        Robin parameter κ_b (≥0 for unitary reflection). If callable, signature
        should be κ_b(omega, ell=None) and return array-like matching ω.
    ell : int, optional
        Multipole index (for models with ℓ-dependence).

    Returns
    -------
    ndarray (complex)
        ℛ(ω) with same shape as ω.
    """
    w = np.asarray(omega, dtype=float)
    if callable(kappa_b):
        k = np.asarray(kappa_b(w, ell), dtype=float)
    else:
        k = float(kappa_b) * np.ones_like(w)
    num = 1j * w - k
    den = 1j * w + k
    R = num / den
    return R.astype(np.complex128)

def make_kappa_const(kappa0: float) -> KappaModel:
    """
    Return a κ_b(ω,ℓ) model that is constant (lossless, frequency-independent).
    """
    def _k(omega: OmegaLike, ell: int | None = None) -> OmegaLike:
        w = np.asarray(omega, dtype=float)
        return np.full_like(w, float(kappa0))
    return _k

def make_kappa_drude(kappa0: float, omega_c: float) -> KappaModel:
    """
    A simple dispersive (lossless) Robin model:
        κ_b(ω) = κ0 / (1 + (ω/ω_c)^2) .
    This keeps κ_b ≥ 0 and yields |ℛ(ω)|=1 (no absorption).

    Parameters
    ----------
    kappa0 : float
        Low-frequency κ_b.
    omega_c : float
        Corner angular frequency [rad/s].
    """
    k0 = float(kappa0)
    oc = float(omega_c)
    def _k(omega: OmegaLike, ell: int | None = None) -> OmegaLike:
        w = np.asarray(omega, dtype=float)
        return k0 / (1.0 + (w/oc)**2)
    return _k

def is_unitary_R(R: Array, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    """
    Check ||R|-1| ≤ atol + rtol for all entries.
    """
    mag = np.abs(R)
    return np.allclose(mag, 1.0, rtol=rtol, atol=atol)

def phase_of_R(R: Array) -> Array:
    """
    Principal-value phase of ℛ (np.angle).
    """
    return np.angle(R)
