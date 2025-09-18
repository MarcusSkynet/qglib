# tests/test_constants.py
import math
import qgledger as qg

def test_alphaH_Istar_Abit_definitions():
    assert qg.alpha_H == 1.0 / (4.0 * qg.LN2)
    assert qg.I_star == 48.0 * qg.LN2
    # A_bit should be 4 ln2 * l_P^2 (area per bit in nats accounting)
    assert qg.A_bit == 4.0 * qg.LN2 * (qg.ell_P ** 2)
