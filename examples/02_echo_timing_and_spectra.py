# examples/02_echo_timing_and_spectra.py
"""
Synthesize a damped ringdown + echo train using the ledger-mapped ε_eff.
Plot time series and magnitude spectrum with expected comb markers.

Outputs:
- Saves: echo_time_series.png, echo_spectrum.png
"""

import numpy as np

from qgledger import (
    epsilon_from_ledger, roundtrip_delay_from_ledger,
    ringdown, echo_train_from_ringdown, spectrum, expected_comb_frequencies
)
from qgledger.viz.echoes import plot_time_series, plot_spectrum

M_SUN = 1.98847e30

def main():
    # Source mass (choose something LIGO-like)
    M = 30.0 * M_SUN

    # Map ledger → outside wall → echo delay
    eps = epsilon_from_ledger(M, model="mixed")
    dt = roundtrip_delay_from_ledger(M, model="mixed")
    comb = expected_comb_frequencies(dt, n_lines=10)

    # Time grid
    fs = 4096.0  # Hz
    T = 2.0      # seconds
    t = np.arange(0.0, T, 1.0/fs)

    # A crude ringdown seed (choose a plausible f0, tau)
    f0 = 150.0     # Hz (illustrative)
    tau = 0.06     # s  (illustrative)
    s0 = ringdown(t, f0_hz=f0, tau_s=tau, phase0=0.0, amplitude=1.0)

    # Echo train with simple reflectivity & damping per bounce
    s = echo_train_from_ringdown(
        t, s0, delta_t=dt, r_coeff=0.5, n_echoes=6, per_echo_damping=0.8
    )

    # Plots
    fig_ts, ax_ts = plot_time_series(t, s, title="Ringdown + echo train")
    fig_ts.savefig("echo_time_series.png", dpi=150, bbox_inches="tight")

    fig_sp, ax_sp = plot_spectrum(s, fs, fmax=2000.0, comb_freqs=comb, title="Echo spectrum with comb markers")
    fig_sp.savefig("echo_spectrum.png", dpi=150, bbox_inches="tight")

    print(f"ε_eff = {eps:.3e}, Δt = {dt:.6e} s")
    print("Saved: echo_time_series.png, echo_spectrum.png")

if __name__ == "__main__":
    main()
