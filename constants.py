# qgledger/constants.py
"""
qgledger.constants
==================
Core physical constants and program-level universal numbers for the IR Quantum Gravity
(ledger/NPR) framework. This module centralizes numerical values (SI by default),
derives Planck units from (G, c, ħ), and exposes the universal dimensionless
constants used across the library.

Conventions
-----------
- Units are SI unless stated otherwise (meters, kilograms, seconds, kelvin).
- Entropy bookkeeping is in *nats* (1 classical bit = ln 2 nats).
- Program-universal constants:
  - alpha_H = 1/(4 ln 2) — capacity fraction S_info / S_ledger (dimensionless).
  - I_star  = 12 / alpha_H = 48 ln 2 — dimensionless ledger placement threshold.

Examples
--------
>>> from qgledger.constants import alpha_H, I_star, ell_P, A_bit
>>> alpha_H, I_star
(1/(4 ln 2), 48 ln 2)  # numerically ~ (0.36067376, 33.2710647)
>>> ell_P  # Planck length in meters (from current G, c, ħ)
1.616255e-35
>>> A_bit  # Area per irreversible classical bit [m^2]
7.2427789e-70
"""
from __future__ import annotations

import math
from typing import Final

# -----------------------------------------------------------------------------
# Mathematical helpers
# -----------------------------------------------------------------------------

#: Natural log of 2 (nats per classical bit).
LN2: Final[float] = math.log(2.0)
#: π (pi).
PI:  Final[float] = math.pi

# -----------------------------------------------------------------------------
# SI constants (CODATA and exact SI definitions)
# -----------------------------------------------------------------------------
#: Speed of light in vacuum [m/s] (exact by SI definition).
c: Final[float] = 299_792_458.0
#: Boltzmann constant [J/K] (exact by SI definition).
k_B: Final[float] = 1.380_649e-23
#: Reduced Planck constant ħ [J·s] (CODATA 2018/2022).
hbar: Final[float] = 1.054_571_817e-34
#: Newton's gravitational constant G [m^3·kg^-1·s^-2] (CODATA 2018/2022).
G: Final[float] = 6.674_30e-11

# Common astronomical helpers (kept minimal for convenience).
#: Solar mass [kg].
M_sun: Final[float] = 1.988_47e30
#: Astronomical Unit [m].
AU: Final[float] = 1.495_978_707e11
#: Parsec [m].
parsec: Final[float] = 3.085_677_581_491_367e16
#: Julian year [s] (365.25 days).
year: Final[float] = 365.25 * 24 * 3600.0

# -----------------------------------------------------------------------------
# Planck units derived from (G, c, ħ)
# -----------------------------------------------------------------------------
def planck_length(G_val: float = G, c_val: float = c, hbar_val: float = hbar) -> float:
    """
    Planck length ℓ_P = sqrt(ħ G / c^3) in meters.

    Parameters
    ----------
    G_val, c_val, hbar_val : float
        Optionally override (G, c, ħ) to explore sensitivity or alternate units.

    Returns
    -------
    float
        Planck length in meters.

    Notes
    -----
    If you override one of (G, c, ħ), ensure the trio is self-consistent;
    all Planck units below are derived from exactly these values.
    """
    return math.sqrt(hbar_val * G_val / (c_val**3))


def planck_time(G_val: float = G, c_val: float = c, hbar_val: float = hbar) -> float:
    """
    Planck time t_P = ℓ_P / c = sqrt(ħ G / c^5) in seconds.
    """
    return planck_length(G_val, c_val, hbar_val) / c_val


def planck_mass(G_val: float = G, c_val: float = c, hbar_val: float = hbar) -> float:
    """
    Planck mass m_P = sqrt(ħ c / G) in kilograms.
    """
    return math.sqrt(hbar_val * c_val / G_val)


def planck_area(G_val: float = G, c_val: float = c, hbar_val: float = hbar) -> float:
    """
    Planck area ℓ_P^2 in square meters.
    """
    Lp = planck_length(G_val, c_val, hbar_val)
    return Lp * Lp


def planck_temperature(
    G_val: float = G, c_val: float = c, hbar_val: float = hbar, k_B_val: float = k_B
) -> float:
    """
    Planck temperature T_P = sqrt(ħ c^5 / (G k_B^2)) in kelvin.
    """
    return math.sqrt(hbar_val * (c_val**5) / (G_val * (k_B_val**2)))


# Cache default Planck units using the constants above.
ell_P: Final[float] = planck_length()
t_P:   Final[float] = planck_time()
m_P:   Final[float] = planck_mass()
A_P:   Final[float] = planck_area()
T_P:   Final[float] = planck_temperature()

# -----------------------------------------------------------------------------
# Program-universal dimensionless constants
# -----------------------------------------------------------------------------
#: Capacity fraction S_info / S_ledger (dimensionless): 1/(4 ln 2).
alpha_H: Final[float] = 1.0 / (4.0 * LN2)
#: Dimensionless ledger-placement threshold: I_star = 12 / alpha_H = 48 ln 2.
I_star:  Final[float] = 12.0 / alpha_H

# -----------------------------------------------------------------------------
# Geometric budgets tied to entropy (nats)
# -----------------------------------------------------------------------------
#: Area per irreversible classical bit: A_bit = (4 ln 2) ℓ_P^2 [m^2].
A_bit_factor: Final[float] = 4.0 * LN2
A_bit:        Final[float] = A_bit_factor * A_P

# -----------------------------------------------------------------------------
# Unit-sphere areas Ω_n and tolerances
# -----------------------------------------------------------------------------
def omega_n(n: int) -> float:
    """
    Surface area of the unit n-sphere S^n: Ω_n = 2 π^{(n+1)/2} / Γ((n+1)/2).

    Parameters
    ----------
    n : int
        Dimension of the sphere surface (e.g., Ω_2 = 4π).

    Returns
    -------
    float
        Surface area of the n-sphere (unit radius).
    """
    from math import gamma
    return 2.0 * (PI ** ((n + 1) / 2.0)) / gamma((n + 1) / 2.0)


#: Relative tolerance for numerical comparisons.
RTOL: Final[float] = 1e-9
#: Absolute tolerance for numerical comparisons.
ATOL: Final[float] = 1e-12
#: Generic tolerance (legacy).
TOL:  Final[float] = 1e-12

__all__ = [
    # math
    "LN2", "PI",
    # SI constants
    "c", "k_B", "hbar", "G", "M_sun", "AU", "parsec", "year",
    # Planck units
    "ell_P", "t_P", "m_P", "A_P", "T_P",
    "planck_length", "planck_time", "planck_mass", "planck_area", "planck_temperature",
    # program constants
    "alpha_H", "I_star", "A_bit_factor", "A_bit",
    # helpers
    "omega_n", "RTOL", "ATOL", "TOL",
]
