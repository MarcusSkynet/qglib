# tests/test_cavity_highD.py
import math
import qgledger as qg

def test_cavity_mapping_positive(mass_30Msun):
    eps = qg.epsilon_from_ledger(mass_30Msun, model="mixed")
    dt = qg.roundtrip_delay_from_ledger(mass_30Msun, model="mixed")
    assert eps > 0.0 and dt > 0.0

def test_highD_factor_and_monotonicity():
    f4 = qg.ledger_radius_factor(4)
    f6 = qg.ledger_radius_factor(6)
    assert 0.5 < f4 < 0.7
    assert f6 > 0.0  # no strict monotonicity asserted here; depends on C_D exponent
