# examples/00_install_check.py
"""
gqlib install check
----------------------
Quick, dependency-light sanity test for the IR QG (ledger/NPR) library.

What it does
------------
- Imports gqlib and prints version + key constants.
- Schwarzschild geometry: r_s, r_ph, r_L and expected ratio r_L/r_s.
- Echo mapping: ε_eff from ledger and Δt > 0.
- Regge–Wheeler barrier: peak near ~3 M_geo, V0>0.
- NPR: on_ledger(r_L) ≈ True.
- Measurement energetics: κ → T, ΔQ_write, ΔQ/A, t_write with simple beam.
- Remnants: closed-form Hawking lifetime + IR-suppressed time-to-stall.
- Cosmology: 𝓘_FRW(a=1) and reports 𝓘_*.
- Codes: Hamming(7,4) encode + single-bit error correction round-trip.
- High-D: prints r_L/r_s factor for D=4 and D=6.
- Viz presence (optional): checks matplotlib import; does not plot.

Run
---
$ python examples/00_install_check.py
"""

from __future__ import annotations

import math
import sys

PASS = "✅"
FAIL = "❌"

def main() -> int:
    try:
        import numpy as np  # noqa
    except Exception:
        print("Note: NumPy not found — most features require NumPy.", file=sys.stderr)

    try:
        import gqlib as qg
    except Exception as e:
        print(f"{FAIL} Import failed: {e}")
        return 1

    print(f"{PASS} gqlib imported, version: {qg.__version__}")
    print(f"   c={qg.c:.6e} m/s,  G={qg.G:.6e} SI,  ℓ_P={qg.ell_P:.6e} m")
    print(f"   alpha_H={qg.alpha_H:.6f},  𝓘_*={qg.I_star:.6f},  A_bit={qg.A_bit:.6e} m²")

    ok_all = True

    # ---------- Schwarzschild basic radii ----------
    try:
        M_SUN = 1.98847e30
        M = 30.0 * M_SUN
        rs = qg.schwarzschild_radius(M)
        rph = qg.photon_sphere_radius(M)
        rL = qg.ledger_radius_schwarzschild(M)
        ratio = rL/rs
        ratio_expected = 1.0/(2.0*math.sqrt(qg.LN2))
        print(f"{PASS} Schwarzschild: r_s={rs:.6e} m, r_ph={rph:.6e} m, r_L={rL:.6e} m; r_L/r_s={ratio:.6f} (expected ~{ratio_expected:.6f})")
        if not (0.55 < ratio < 0.65):
            print(f"{FAIL} r_L/r_s out of expected band.", file=sys.stderr)
            ok_all = False
    except Exception as e:
        print(f"{FAIL} Schwarzschild radii check failed: {e}")
        ok_all = False

    # ---------- Echo mapping (ε_eff, Δt) ----------
    try:
        eps = qg.epsilon_from_ledger(M, model="mixed")
        dt = qg.echo_delay_seconds(M, eps_eff=eps)
        cond = (eps > 0.0) and (dt > 0.0)
        print(f"{PASS if cond else FAIL} Echo mapping: ε_eff={eps:.3e}, Δt={dt:.6e} s")
        ok_all &= cond
    except Exception as e:
        print(f"{FAIL} Echo mapping failed: {e}")
        ok_all = False

    # ---------- Regge–Wheeler barrier ----------
    try:
        l = 2
        rpk, V0, Vpp = qg.rw_barrier_peak(l, M_geo=qg.mass_geometric_length(M))
        cond = (V0 > 0.0) and (2.5*qg.mass_geometric_length(M) < rpk < 3.5*qg.mass_geometric_length(M))
        print(f"{PASS if cond else FAIL} RW barrier: r_peak={rpk:.6e} m (~3M), V0={V0:.3e} 1/m², V''_*={Vpp:.3e} 1/m⁴")
        ok_all &= cond
    except Exception as e:
        print(f"{FAIL} Barrier check failed: {e}")
        ok_all = False

    # ---------- NPR placement ----------
    try:
        on = qg.on_ledger(M, rL)
        print(f"{PASS if on else FAIL} NPR placement at r_L: on_ledger={on}")
        ok_all &= bool(on)
    except Exception as e:
        print(f"{FAIL} NPR check failed: {e}")
        ok_all = False

    # ---------- Measurement energetics ----------
    try:
        a = 1.0e20  # m/s² (illustrative)
        kappa = qg.kappa_from_acceleration(a)
        T = qg.unruh_temperature_from_kappa(kappa)
        dQ = qg.energy_per_write_total(kappa)
        dQ_A = qg.energy_per_area_min(kappa)
        # Beam: P=1mW, waist=10 µm
        area = qg.beam_area_from_waist(10e-6)
        F = qg.intensity_from_power_area(1e-3, area)
        tmin = qg.time_to_write(F, kappa)
        cond = (T >= 0.0) and (dQ > 0.0) and (dQ_A > 0.0) and (tmin > 0.0)
        print(f"{PASS if cond else FAIL} Measurement: κ={kappa:.3e} 1/s, T={T:.3e} K, ΔQ={dQ:.3e} J, (ΔQ/A)_min={dQ_A:.3e} J/m², t_write≈{tmin:.3e} s")
        ok_all &= cond
    except Exception as e:
        print(f"{FAIL} Measurement check failed: {e}")
        ok_all = False

    # ---------- Evaporation & remnant ----------
    try:
        M0 = 1.0e12  # kg
        Mrem = 2.0e5
        tau0 = qg.lifetime_hawking_closedform(M0)
        tstall = qg.time_to_stall(M0, Mrem, dt=1e6)
        cond = (tau0 > 0.0) and (tstall > 0.0) and (tstall < tau0)
        print(f"{PASS if cond else FAIL} Evaporation: τ_H≈{tau0:.3e} s, t_stall≈{tstall:.3e} s")
        ok_all &= cond
    except Exception as e:
        print(f"{FAIL} Evaporation check failed: {e}")
        ok_all = False

    # ---------- Cosmology invariant ----------
    try:
        H0 = 70_000.0 / 3.085677581e22
        Om, Or, Ode, w = 0.315, 8.5e-5, 0.685, -1.0
        I_today = qg.invariant_I_frw(1.0, H0=H0, Om=Om, Or=Or, Ode=Ode, w=w)
        print(f"{PASS} Cosmology: 𝓘_FRW(a=1)={I_today:.6f},  𝓘_*={qg.I_star:.6f}")
    except Exception as e:
        print(f"{FAIL} Cosmology check failed: {e}")
        ok_all = False

    # ---------- Codes (Hamming 7,4) ----------
    try:
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        code = qg.hamming74_encode(data)
        # flip one bit
        code_flipped = code.copy()
        code_flipped[3] ^= 1
        data_dec, code_corr, corrected, detected_multi = qg.hamming74_decode(code_flipped)
        cond = corrected and not detected_multi and np.all(data_dec == data)
        print(f"{PASS if cond else FAIL} Hamming(7,4): corrected={corrected}, multi_err={detected_multi}, ok={bool(np.all(data_dec == data))}")
        ok_all &= cond
    except Exception as e:
        print(f"{FAIL} Codes check failed: {e}")
        ok_all = False

    # ---------- High-D ----------
    try:
        f4 = qg.ledger_radius_factor(4)
        f6 = qg.ledger_radius_factor(6)
        cond = (0.5 < f4 < 0.7) and (0.3 < f6 < 0.9)
        print(f"{PASS if cond else FAIL} High-D: r_L/r_s (D=4)={f4:.6f}, (D=6)={f6:.6f}")
        ok_all &= cond
    except Exception as e:
        print(f"{FAIL} High-D check failed: {e}")
        ok_all = False

    # ---------- Viz availability (optional) ----------
    try:
        import matplotlib  # noqa
        print(f"{PASS} Matplotlib present (viz helpers usable).")
    except Exception:
        print("ℹ️  Matplotlib not found — viz helpers will be unavailable (core OK).")

    print("\n" + ("ALL CHECKS PASSED 🎉" if ok_all else "Some checks failed. See ❌ above."))
    return 0 if ok_all else 2


if __name__ == "__main__":
    try:
        import numpy as np  # needed for codes check
    except Exception:
        print("❌ NumPy is required for this install check.", file=sys.stderr)
        sys.exit(1)
    sys.exit(main())
