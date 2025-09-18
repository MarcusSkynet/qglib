# tests/test_geom.py
import math
import qgledger as qg

def test_basic_radii(mass_30Msun):
    rs = qg.schwarzschild_radius(mass_30Msun)
    rph = qg.photon_sphere_radius(mass_30Msun)
    rL = qg.ledger_radius_schwarzschild(mass_30Msun)
    assert rph == 1.5 * rs
    expected_ratio = 1.0 / (2.0 * math.sqrt(qg.LN2))
    assert abs(rL/rs - expected_ratio) < 1e-9

def test_kretschmann_and_invariant(mass_30Msun):
    rs = qg.schwarzschild_radius(mass_30Msun)
    r = 10.0 * rs
    K = qg.kretschmann_schwarzschild(r, M_geo=qg.mass_geometric_length(mass_30Msun))
    # In 4D Schwarzschild: K = 12 r_s^2 / r^6
    assert math.isfinite(K) and K > 0.0
    K_expected = 12.0 * (rs ** 2) / (r ** 6)
    assert abs((K - K_expected)/K_expected) < 1e-12
    I = qg.invariant_I_schwarzschild(r, M_geo=qg.mass_geometric_length(mass_30Msun))
    # I = K r^4 = 12 r_s^2 / r^2
    I_expected = 12.0 * (rs ** 2) / (r ** 2)
    assert abs((I - I_expected)/I_expected) < 1e-12
