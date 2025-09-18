# gqlib/viz/__init__.py
"""
qglib.viz
============
Lightweight plotting helpers for the IR Quantum Gravity (ledger/NPR) library.

Design
------
- Uses only matplotlib (no seaborn, no styles, no custom colors).
- Each function returns (fig, ax) so callers can further customize or save.
- Plots stick to SI units and label key radii/quantities clearly.

Submodules
----------
- echoes      : time series and spectra (with optional comb markers)
- barrier     : Regge–Wheeler potential with r_s, r_ph, r_peak markers
- tortoise    : exterior r_*(r) visualization and effective wall placement
- remnants    : evaporation histories M(t) with IR suppression
- cosmology   : FRW invariant 𝓘_FRW(a) and related horizon thermodynamics
"""

from .echoes import plot_time_series, plot_spectrum
# from .barrier import plot_rw_barrier
from .tortoise import plot_rstar_exterior
from .remnants import plot_evaporation_history
from .cosmology import plot_I_FRW

__all__ = [
    "plot_time_series",
    "plot_spectrum",
    "plot_rw_barrier",
    "plot_rstar_exterior",
    "plot_evaporation_history",
    "plot_I_FRW",
]
