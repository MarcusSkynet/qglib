# examples/03_measurement_thresholds.py
"""
Ledger write energetics for lab-like settings:
- Convert apparatus intensity to time-to-write using the Unruh/κ relation.
- Report energy per minimal write and minimal energy per unit area.

Outputs:
- Console print of κ, T, ΔQ_write, (ΔQ/A)_min, t_write for given power/spot.
"""

from qgledger import (
    kappa_from_acceleration, unruh_temperature_from_kappa,
    energy_per_write_total, energy_per_area_min,
    beam_area_from_waist, intensity_from_power_area, time_to_write
)

def main():
    # Choose a large proper acceleration scale (illustrative!)
    a = 1.0e20  # m/s^2
    kappa = kappa_from_acceleration(a)
    T = unruh_temperature_from_kappa(kappa)
    dQ = energy_per_write_total(kappa)
    dQ_per_A = energy_per_area_min(kappa)

    # Beam/apparatus parameters
    power_w = 1e-3     # 1 mW
    waist_m = 10e-6    # 10 microns
    area_m2 = beam_area_from_waist(waist_m)
    F = intensity_from_power_area(power_w, area_m2)
    tmin = time_to_write(F, kappa)

    print(f"a = {a:.3e} m/s^2 → κ = {kappa:.3e} 1/s → T_Unruh = {T:.3e} K")
    print(f"ΔQ_write (4 bits) = {dQ:.3e} J")
    print(f"(ΔQ/A)_min = {dQ_per_A:.3e} J/m^2")
    print(f"Beam: P = {power_w:.3e} W, waist = {waist_m:.3e} m, area = {area_m2:.3e} m^2")
    print(f"Intensity F = {F:.3e} W/m^2 → minimal write time t_write ≈ {tmin:.3e} s")

if __name__ == "__main__":
    main()
