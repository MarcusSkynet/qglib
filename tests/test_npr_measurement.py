# tests/test_npr_measurement.py
import qgledger as qg

def test_on_ledger_and_write_cost(mass_30Msun):
    rL = qg.ledger_radius_schwarzschild(mass_30Msun)
    assert qg.on_ledger(mass_30Msun, rL)
    # Clausius energetics sanity
    a = 1e20
    kappa = qg.kappa_from_acceleration(a)
    T = qg.temperature_from_surface_gravity(kappa)
    E_write = qg.write_cost(kappa)  # total ΔQ (4 bits)
    assert T >= 0.0 and E_write > 0.0

def test_clausius_flux_check():
    kappa = qg.kappa_from_acceleration(1e20)
    Freq = qg.clausius_required_flux(kappa)  # minimal ΔQ/A per unit time → W/m²
    ok, factor = qg.clausius_check(F=Freq, kappa=kappa, gate_time_s=1.0)
    assert ok and factor >= 1.0
