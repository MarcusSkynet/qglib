# qgledger/__init__.py
"""
qgledger
========
A lightweight Python library implementing the IR Quantum Gravity (ledger/NPR)
machinery for black holes and cosmology:

- **Invariant placement:** locate the ledger worldvolume 𝓛 from the dimensionless
  invariant 𝓘 = 𝓚 r_A^4 with universal threshold 𝓘_* = 48 ln 2.
- **NPR boundary rules:** no longitudinal transport (T_{++}=T_{--}=0) between
  writes; Clausius balance at writes with minimal record of four classical bits.
- **Detector-facing tools:** exterior tortoise gaps, Regge–Wheeler barrier,
  echo trains and spectral combs, effective outside wall ε_eff.
- **Evaporation & remnants:** Hawking T/P and IR/DCT suppression to a finite
  remnant mass.
- **Cosmology:** flat-FRW horizon thermodynamics and the IR invariant 𝓘_FRW(a).
- **Coding:** minimal classical ECC (Hamming(7,4)) for 4-bit ledger writes.

Modules
-------
- :mod:`qgledger.constants`      — SI/Planck constants, alpha_H, I_star, A_bit
- :mod:`qglib.utils`          — numerics & unit conversions
- :mod:`qglib.geom`           — radii & invariants (4D), ledger radius r_L
- :mod:`qglib.tortoise`       — exterior r_*(r), ε inversion, echo Δt
- :mod:`qglib.barrier`        — RW barrier, peak, WKB T/R (Schwarzschild)
- :mod:`qglib.barrier_kerr`   — toy Kerr horizons, photon rings, gaps, Δt
- :mod:`qglib.echoes`         — ringdown base, echo trains, spectra
- :mod:`qglib.npr`            — NPR constraints, write thermodynamics
- :mod:`qglib.measurement`    — κ, Unruh T, ΔQ per write, time-to-write
- :mod:`qglib.remnants`       — Hawking evaporation with IR stall
- :mod:`qglib.cosmology`      — FRW H(a), R_A, T_A, S, 𝓘_FRW
- :mod:`qgledger.codes`          — 4-bit register & Hamming(7,4) ECC
- :mod:`qglib.cavity`         — map interior ledger → exterior ε_eff (echoes)
- :mod:`qglib.highD`          — Tangherlini (D≥4) invariants & r_L/r_s

Versioning
----------
The library follows simple semantic versioning (MAJOR.MINOR.PATCH). API is
young and may evolve; pin to a minor release when scripting.
"""

from __future__ import annotations

# Public version string
__version__ = "0.1.0"

# Re-exports: constants and key helpers for a friendly top-level API
from .constants import (
    # physical constants
    c, G, hbar, k_B, ell_P, LN2,
    # IR-QG constants
    alpha_H, I_star, A_bit,
    # tolerances
    RTOL, ATOL,
)

# Geometry & invariants (4D)
from .geom import (
    mass_geometric_length,
    schwarzschild_radius,
    photon_sphere_radius,
    kretschmann_schwarzschild,
    invariant_I_schwarzschild,
    ledger_radius_schwarzschild,
)

# Exterior tortoise & echo spacing
from .tortoise import (
    tortoise_coordinate,
    # delta_rstar_to_wall,
    epsilon_for_target_gap,
    echo_delay_seconds,
    echo_delay_from_target_gap,
)

# Barrier & WKB (Schwarzschild)
from .barrier import (
    V_RW,
    d2V_drstar2,
    # rw_barrier_peak,
    wkb_transmission_reflection,
)

# Echo synthesis & spectra
from .echoes import (
    ringdown,
    gaussian_pulse,
    echo_train,
    spectrum,
    expected_comb_frequencies,
    echo_train_from_ringdown,
)

# NPR & measurement
from .npr import (
    placement_residual,
    on_ledger,
    npr_satisfied,
    write_cost,
    temperature_from_surface_gravity,
    clausius_required_flux,
    clausius_check,
    write_event_diagnostics,
)
from .measurement import (
    kappa_from_acceleration,
    unruh_temperature_from_kappa,
    energy_per_write_total,
    energy_per_area_min,
    intensity_from_power_area,
    beam_area_from_waist,
    photon_energy,
    photon_flux,
    time_to_write,
    can_write_within_gate,
)

# Evaporation / remnants
from .remnants import (
    hawking_temperature,
    hawking_power,
    dMdt_hawking,
    lifetime_hawking_closedform,
    suppression_smooth,
    dMdt_suppressed,
    integrate_evaporation_suppressed,
    time_to_stall,
)

# Cosmology
from .cosmology import (
    hubble_H,
    hubble_Hdot,
    apparent_horizon_radius,
    horizon_temperature,
    horizon_area,
    horizon_entropy_nats,
    kretschmann_frw_flat,
    invariant_I_frw,
    ledger_condition_frw,
)

# Codes
from .codes import (
    sign_to_bit, bit_to_sign, pack_register, unpack_register,
    hamming74_matrices, hamming74_encode, hamming74_syndrome, hamming74_decode,
    bitflip_channel, repetition_encode, repetition_decode_majority,
    area_for_n_registers,
)

# Interior→exterior mapping (echo phenomenology)
from .cavity import (
    rstar_interior_real,
    target_gap_from_ledger,
    epsilon_from_ledger,
    roundtrip_delay_from_ledger,
)

# High-D (Tangherlini)
from .highD import (
    omega_n,
    schwarzschild_radius_tangherlini,
    mu_from_rs, mu_from_mass,
    kretschmann_tangherlini,
    invariant_I_highD,
    ledger_radius_factor,
    ledger_radius_highD_from_rs,
    ledger_radius_highD,
)

__all__ = [
    "__version__",
    # constants
    "c", "G", "hbar", "k_B", "ell_P", "LN2", "alpha_H", "I_star", "A_bit", "RTOL", "ATOL",
    # geometry
    "mass_geometric_length", "schwarzschild_radius", "photon_sphere_radius",
    "kretschmann_schwarzschild", "invariant_I_schwarzschild", "ledger_radius_schwarzschild",
    # tortoise / echoes
    "tortoise_coordinate", "delta_rstar_to_wall", "epsilon_for_target_gap",
    "echo_delay_seconds", "echo_delay_from_target_gap",
    # barrier (Schwarzschild)
    "V_RW", "d2V_drstar2", "rw_barrier_peak", "wkb_transmission_reflection",
    # echo synthesis
    "ringdown", "gaussian_pulse", "echo_train", "spectrum",
    "expected_comb_frequencies", "echo_train_from_ringdown",
    # NPR & measurement
    "placement_residual", "on_ledger", "npr_satisfied", "write_cost",
    "temperature_from_surface_gravity", "clausius_required_flux", "clausius_check",
    "write_event_diagnostics",
    "kappa_from_acceleration", "unruh_temperature_from_kappa", "energy_per_write_total",
    "energy_per_area_min", "intensity_from_power_area", "beam_area_from_waist",
    "photon_energy", "photon_flux", "time_to_write", "can_write_within_gate",
    # remnants
    "hawking_temperature", "hawking_power", "dMdt_hawking", "lifetime_hawking_closedform",
    "suppression_smooth", "dMdt_suppressed", "integrate_evaporation_suppressed", "time_to_stall",
    # cosmology
    "hubble_H", "hubble_Hdot", "apparent_horizon_radius", "horizon_temperature",
    "horizon_area", "horizon_entropy_nats", "kretschmann_frw_flat",
    "invariant_I_frw", "ledger_condition_frw",
    # codes
    "sign_to_bit", "bit_to_sign", "pack_register", "unpack_register",
    "hamming74_matrices", "hamming74_encode", "hamming74_syndrome", "hamming74_decode",
    "bitflip_channel", "repetition_encode", "repetition_decode_majority",
    "area_for_n_registers",
    # cavity mapping
    "rstar_interior_real", "target_gap_from_ledger", "epsilon_from_ledger", "roundtrip_delay_from_ledger",
    # high-D
    "omega_n", "schwarzschild_radius_tangherlini", "mu_from_rs", "mu_from_mass",
    "kretschmann_tangherlini", "invariant_I_highD", "ledger_radius_factor",
    "ledger_radius_highD_from_rs", "ledger_radius_highD",
]
