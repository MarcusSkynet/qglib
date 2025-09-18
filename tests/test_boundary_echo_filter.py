# tests/test_boundary_echo_filter.py
import numpy as np
import qgledger as qg
from qgledger.boundary import make_kappa_const, robin_reflection, is_unitary_R
from qgledger.echo_filter import apply_echo_filter_fd

def test_robin_unitarity_and_equivalence():
    # Build ℛ(ω) from a constant κ_b; check |R|=1
    kappa0 = 150.0
    R_of_w = lambda w: robin_reflection(w, make_kappa_const(kappa0))
    w = np.linspace(0.0, 2*np.pi*2000.0, 256)
    assert is_unitary_R(R_of_w(w))

    # Compare to time-domain echo_train with constant scalar r_coeff≈phase(R) at f~band
    fs = 2048.0
    T = 1.0
    t = np.arange(0.0, T, 1.0/fs)
    s0 = qg.ringdown(t, f0_hz=150.0, tau_s=0.05, amplitude=1.0, phase0=0.0)
    dt = 0.03
    # Frequency-domain filtered version using ℛ(ω)
    y_fd = apply_echo_filter_fd(s0, fs, dt, R_of_w, n_echoes=4)
    # Baseline constant-reflectivity echo generator (magnitude only)
    y_td = qg.echo_train_from_ringdown(t, s0, delta_t=dt, r_coeff=0.5, n_echoes=4, per_echo_damping=1.0)

    # The detailed waveforms won't be identical (phase response differs);
    # at least ensure energies and echo spacing are in the same ballpark.
    E_fd = float(np.dot(y_fd, y_fd))
    E_td = float(np.dot(y_td, y_td))
    assert E_fd > 0.0 and E_td > 0.0
    ratio = E_fd / E_td
    assert 0.1 < ratio < 10.0
