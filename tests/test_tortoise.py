# tests/test_tortoise.py
import math
import qgledger as qg

def test_tortoise_monotone_and_diverges(mass_30Msun):
    rs = qg.schwarzschild_radius(mass_30Msun)
    r1 = rs * 1.001
    r2 = rs * 1.01
    t1 = qg.tortoise_coordinate(r1, rs)
    t2 = qg.tortoise_coordinate(r2, rs)
    assert t2 > t1  # monotone increasing
    # Near the horizon, r_* ~ rs ln(r/rs - 1) → -∞
    assert t1 < -100.0 * rs

def test_epsilon_inversion_roundtrip(mass_30Msun):
    rs = qg.schwarzschild_radius(mass_30Msun)
    rph = qg.photon_sphere_radius(mass_30Msun)
    eps = 1e-6
    gap = qg.delta_rstar_to_wall(rs, eps, r_ref=rph)
    eps_back = qg.epsilon_for_target_gap(rs, gap, r_ref=rph)
    rel = abs(eps_back - eps)/eps
    assert rel < 1e-6
