# tests/test_measurement.py
import qgledger as qg

def test_time_to_write_monotone_with_intensity():
    kappa = qg.kappa_from_acceleration(1e20)
    t1 = qg.time_to_write(F_w_per_m2=1e6, kappa=kappa)
    t2 = qg.time_to_write(F_w_per_m2=1e7, kappa=kappa)
    assert t2 < t1 and t1 > 0.0 and t2 > 0.0

def test_photon_helpers():
    E = qg.photon_energy(1e14)
    rate = qg.photon_flux(1e-3, 1e14)
    assert E > 0.0 and rate > 0.0
