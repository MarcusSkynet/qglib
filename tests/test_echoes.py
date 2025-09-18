# tests/test_echoes.py
import numpy as np
import qgledger as qg

def test_echo_train_and_spectrum():
    fs = 2048.0
    T = 1.0
    t = np.arange(0.0, T, 1.0/fs)
    s0 = qg.ringdown(t, f0_hz=150.0, tau_s=0.05, amplitude=1.0, phase0=0.0)
    dt = 0.03
    s = qg.echo_train_from_ringdown(t, s0, delta_t=dt, r_coeff=0.5, n_echoes=4, per_echo_damping=0.8)
    f, mag = qg.spectrum(s, fs, one_sided=True)
    assert f[0] >= 0.0 and np.all(np.isfinite(mag))
    comb = qg.expected_comb_frequencies(dt, n_lines=5)
    assert len(comb) == 5 and np.all(np.diff(comb) > 0)
