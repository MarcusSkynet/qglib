# tests/test_barrier.py
import qgledger as qg

def test_barrier_peak_location_and_wkb(mass_30Msun):
    Mgeo = qg.mass_geometric_length(mass_30Msun)
    rpk, V0, Vpp = qg.rw_barrier_peak(l=2, M_geo=Mgeo)
    assert 2.5*Mgeo < rpk < 3.5*Mgeo
    assert V0 > 0.0
    T, R = qg.wkb_transmission_reflection(omega_geom=1.0/(3.0*Mgeo), l=2, M_geo=Mgeo)
    assert 0.0 <= T <= 1.0 and 0.0 <= R <= 1.0
    assert abs(T + R - 1.0) < 1e-6
