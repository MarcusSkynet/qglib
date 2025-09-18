# tests/test_cosmology.py
import qgledger as qg

def test_frw_invariant_runs_and_positive():
    H0 = 70_000.0 / 3.085677581e22
    Om, Or, Ode, w = 0.315, 8.5e-5, 0.685, -1.0
    I = qg.invariant_I_frw(1.0, H0=H0, Om=Om, Or=Or, Ode=Ode, w=w)
    I_star = qg.I_star
    assert I > 0.0 and I_star > 0.0
    # We just check the API and positivity; absolute matching depends on units choice.
    val, star, diff = qg.ledger_condition_frw(1.0, H0=H0, Om=Om, Or=Or, Ode=Ode, w=w)
    assert val == I and star == I_star
