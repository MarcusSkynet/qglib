# examples/04_evaporation_to_remnant.py
"""
Integrate a Hawking evaporation history with IR/DCT suppression to a finite remnant.
Plot mass vs time.

Outputs:
- Saves: evaporation_history.png
- Prints: closed-form Hawking lifetime (to zero) and numerical time-to-stall at M_rem.
"""

from qgledger import (
    lifetime_hawking_closedform, time_to_stall
)
from qgledger.viz.remnants import plot_evaporation_history

def main():
    # A primordial black hole scale (illustrative)
    M0 = 1.0e12  # kg
    M_rem = 2.0e5  # kg (choose a finite remnant)
    g_eff = 1.0
    p = 4.0

    t_hawk = lifetime_hawking_closedform(M0, g_eff=g_eff)
    t_stall = time_to_stall(M0, M_rem, g_eff=g_eff, p=p, dt=1e6)

    print(f"M0 = {M0:.3e} kg, M_rem = {M_rem:.3e} kg")
    print(f"Hawking lifetime to zero (idealized): τ_H ≈ {t_hawk:.3e} s")
    print(f"Numerical time to stall at M_rem: t_stall ≈ {t_stall:.3e} s")

    fig, ax = plot_evaporation_history(M0, M_rem, g_eff=g_eff, p=p, dt=1e6, t_max=None,
                                       title="Evaporation with IR suppression to remnant")
    fig.savefig("evaporation_history.png", dpi=150, bbox_inches="tight")
    print("Saved: evaporation_history.png")

if __name__ == "__main__":
    main()
