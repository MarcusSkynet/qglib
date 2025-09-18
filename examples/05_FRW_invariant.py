# examples/05_FRW_invariant.py
"""
Compute and plot the FRW IR invariant 𝓘_FRW(a) across a range of scale factors.
Annotate the universal threshold 𝓘_*.

Outputs:
- Saves: frw_invariant.png
- Prints: 𝓘_FRW(a=1) and 𝓘_*.
"""

from qgledger import invariant_I_frw, I_star
from qgledger.viz.cosmology import plot_I_FRW

def main():
    # ΛCDM-like parameters
    H0 = 70_000.0 / 3.085677581e22  # 70 km/s/Mpc → 1/s
    Om = 0.315
    Or = 8.5e-5
    Ode = 0.685
    w = -1.0

    # Plot over a ∈ [0.1, 2]
    fig, ax = plot_I_FRW(0.1, 2.0, H0=H0, Om=Om, Or=Or, Ode=Ode, w=w, N=400)
    fig.savefig("frw_invariant.png", dpi=150, bbox_inches="tight")
    print("Saved: frw_invariant.png")

    I_today = invariant_I_frw(1.0, H0=H0, Om=Om, Or=Or, Ode=Ode, w=w)
    print(f"𝓘_FRW(a=1) ≈ {I_today:.6f}")
    print(f"𝓘_* = {I_star:.6f}")

if __name__ == "__main__":
    main()
