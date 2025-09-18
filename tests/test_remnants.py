# tests/test_remnants.py
import numpy as np
import qgledger as qg

def test_evaporation_signs_and_stall():
    M0 = 1e12
    Mrem = 2e5
    assert qg.dMdt_hawking(M0) < 0.0
    assert qg.dMdt_suppressed(Mrem, M_rem_kg=Mrem) == 0.0
    t, M = qg.integrate_evaporation_suppressed(M0, Mrem, dt=1e6)
    assert M[-1] <= Mrem + 1e-6*Mrem
    assert t[-1] > 0.0
