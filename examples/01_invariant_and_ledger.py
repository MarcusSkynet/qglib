# examples/01_invariant_and_ledger.py
"""
Place the ledger for a Schwarzschild black hole and visualize the exterior tortoise r_*(r).
Also compute the effective outside wall ε_eff that reproduces the interior ledger gap.

Outputs:
- Prints r_s, r_ph, r_L, ε_eff, Δt.
- Saves plot: invariant_and_ledger_rstar.png
"""

import numpy as np

from qgledger import (
    schwarzschild_radius, photon_sphere_radius, ledger_radius_schwarzschild,
    echo_delay_seconds, epsilon_from_ledger,
)
from qgledger.viz.tortoise import plot_rstar_exterior

# Simple solar-mass constant (kg)
M_SUN = 1.98847e30

def main():
    M = 30.0 * M_SUN

    r_s = schwarzschild_radius(M)
    r_ph = photon_sphere_radius(M)
    r_L = ledger_radius_schwarzschild(M)

    # Map interior ledger → exterior effective wall offset
    eps = epsilon_from_ledger(M, model="mixed")
    dt = echo_delay_seconds(M, eps_eff=eps)

    print(f"M = {M:.6e} kg")
    print(f"r_s = {r_s:.6e} m")
    print(f"r_ph = {r_ph:.6e} m")
    print(f"r_L = {r_L:.6e} m  (~{r_L/r_s:.4f} r_s)")
    print(f"ε_eff (from ledger) = {eps:.6e}")
    print(f"Echo roundtrip delay Δt ≈ {dt:.6e} s")

    fig, ax = plot_rstar_exterior(M, eps_eff=eps, rmax_factor=10.0)
    fig.savefig("invariant_and_ledger_rstar.png", dpi=150, bbox_inches="tight")
    print("Saved: invariant_and_ledger_rstar.png")

if __name__ == "__main__":
    main()
