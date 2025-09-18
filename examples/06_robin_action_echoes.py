# examples/06_robin_action_echoes.py
"""
Use the action-implied Robin boundary to synthesize echoes with a **unitary**
frequency-dependent reflection R(ω). Compare to constant reflectivity.

Outputs:
- Saves: robin_action_echoes.png
"""

import numpy as np
import matplotlib.pyplot as plt

from qgledger import epsilon_from_ledger, roundtrip_delay_from_ledger, ringdown
from qgledger.boundary import make_kappa_const, make_kappa_drude, robin_reflection, is_unitary_R
from qgledger.echo_filter import apply_echo_filter_fd, comb_frequencies_from_dt

M_SUN = 1.98847e30

def main():
    M = 30.0 * M_SUN
    dt = roundtrip_delay_from_ledger(M, model="mixed")
    eps = epsilon_from_ledger(M, model="mixed")
    print(f"Δt = {dt:.6e} s, ε_eff ≈ {eps:.3e}")

    # Seed ringdown
    fs = 4096.0
    T = 2.0
    t = np.arange(0.0, T, 1.0/fs)
    s0 = ringdown(t, f0_hz=150.0, tau_s=0.06, amplitude=1.0, phase0=0.0)

    # Build two Robin models:
    k_const = make_kappa_const(kappa0=200.0)           # [1/s], arbitrary illustrative scale
    k_drude = make_kappa_drude(kappa0=200.0, omega_c=800.0*2*np.pi)

    # Wrap into ℛ(ω)
    R_const = lambda w: robin_reflection(w, k_const)   # vectorized
    R_drude = lambda w: robin_reflection(w, k_drude)

    # Sanity: |R|=1 between writes
    wtest = np.linspace(0.0, 2000.0*2*np.pi, 512)
    assert is_unitary_R(R_const(wtest))
    assert is_unitary_R(R_drude(wtest))

    # Apply frequency-domain echo filter
    y_const  = apply_echo_filter_fd(s0, fs, dt, R_const, n_echoes=6)
    y_drude  = apply_echo_filter_fd(s0, fs, dt, R_drude, n_echoes=6)

    # Plot time series overlay
    fig, ax = plt.subplots()
    ax.plot(t, s0, label="seed ringdown", linewidth=1.0)
    ax.plot(t, y_const, label="echoes (Robin const κ_b)", linewidth=1.0)
    ax.plot(t, y_drude, label="echoes (Robin Drude κ_b)", linewidth=1.0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("amplitude [a.u.]")
    ax.legend(loc="upper right"); ax.grid(True, ls=":")
    fig.savefig("robin_action_echoes.png", dpi=150, bbox_inches="tight")
    print("Saved: robin_action_echoes.png")

if __name__ == "__main__":
    main()
