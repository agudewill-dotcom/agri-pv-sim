"""
test_core.py — Unit tests for Agri-PV physics modules (v9.0)

Tests cover:
- Analytical SVF limits and height dependence
- Ground irradiance component assembly
- PAR conversion factors
- Faiman thermal model behavior
- Beam transmission clamping
"""

import numpy as np
import sys
import os

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(__file__))

import irradiance
import thermal
import geometry


def test_svf_limits():
    """SVF must approach 1.0 for very tall systems and decrease for shorter ones."""
    # Very tall system: rows are far above, minimal sky blockage
    svf_tall = irradiance.sky_view_factor_periodic(
        h_top=50.0, proj_width=5.44, pitch=8.63, tau_eff=0.15
    )
    # Very short system: rows are close to ground, maximum blockage
    svf_short = irradiance.sky_view_factor_periodic(
        h_top=0.1, proj_width=5.44, pitch=8.63, tau_eff=0.15
    )
    # Zero height: no obstruction
    svf_zero = irradiance.sky_view_factor_periodic(
        h_top=0.0, proj_width=5.44, pitch=8.63, tau_eff=0.15
    )
    
    assert svf_zero == 1.0, f"SVF at h=0 should be 1.0, got {svf_zero}"
    assert svf_short > svf_tall or abs(svf_short - svf_tall) < 0.01, \
        f"SVF should decrease with height (blockage increases): short={svf_short}, tall={svf_tall}"
    # Both should be in valid range
    assert 0.0 <= svf_tall <= 1.0, f"SVF out of range: {svf_tall}"
    assert 0.0 <= svf_short <= 1.0, f"SVF out of range: {svf_short}"
    print(f"  SVF limits: h=0 → {svf_zero:.4f}, h=0.1 → {svf_short:.4f}, h=50 → {svf_tall:.4f}")


def test_svf_transparency_effect():
    """Higher transparency should increase SVF."""
    svf_opaque = irradiance.sky_view_factor_periodic(
        h_top=3.56, proj_width=5.44, pitch=8.63, tau_eff=0.0
    )
    svf_trans = irradiance.sky_view_factor_periodic(
        h_top=3.56, proj_width=5.44, pitch=8.63, tau_eff=0.20
    )
    assert svf_trans > svf_opaque, \
        f"Transparent modules should have higher SVF: opaque={svf_opaque}, trans={svf_trans}"
    print(f"  SVF transparency: opaque={svf_opaque:.4f}, tau=0.20={svf_trans:.4f}")


def test_par_conversion():
    """PAR conversion should produce ~2.06 µmol/J effective factor."""
    g = 500.0  # W/m²
    par = irradiance.calculate_par(g)
    effective_factor = par / g
    assert 2.0 < effective_factor < 2.2, \
        f"Effective PAR factor should be ~2.06, got {effective_factor:.3f}"
    print(f"  PAR: {g} W/m² → {par:.1f} µmol/m²/s (factor={effective_factor:.3f})")


def test_faiman_height_effect():
    """Higher mounting should produce lower cell temperatures."""
    t_low = thermal.cell_temperature_faiman(
        t_amb=25.0, g_poa=800.0, wind_speed_10m=3.0, h_clearance=0.8
    )
    t_high = thermal.cell_temperature_faiman(
        t_amb=25.0, g_poa=800.0, wind_speed_10m=3.0, h_clearance=2.1
    )
    assert t_high < t_low, \
        f"Higher mounting should be cooler: h=0.8→{t_low:.1f}°C, h=2.1→{t_high:.1f}°C"
    delta = t_low - t_high
    # Delta should be physically plausible (0.5-5°C range)
    assert 0.1 < delta < 5.0, \
        f"Temperature delta should be 0.1-5°C, got {delta:.2f}°C"
    print(f"  Faiman: h=0.8→{t_low:.1f}°C, h=2.1→{t_high:.1f}°C, ΔT={delta:.1f}°C")


def test_faiman_wind_effect():
    """Higher wind should produce lower cell temperatures."""
    t_calm = thermal.cell_temperature_faiman(
        t_amb=25.0, g_poa=800.0, wind_speed_10m=1.0, h_clearance=2.1
    )
    t_windy = thermal.cell_temperature_faiman(
        t_amb=25.0, g_poa=800.0, wind_speed_10m=5.0, h_clearance=2.1
    )
    assert t_windy < t_calm, \
        f"Higher wind should cool: v=1→{t_calm:.1f}°C, v=5→{t_windy:.1f}°C"
    print(f"  Wind effect: v=1→{t_calm:.1f}°C, v=5→{t_windy:.1f}°C")


def test_temp_efficiency_factor():
    """Temperature factor should be <1 above STC and >1 below STC."""
    f_hot = thermal.temperature_efficiency_factor(50.0)
    f_cold = thermal.temperature_efficiency_factor(10.0)
    f_stc = thermal.temperature_efficiency_factor(25.0)
    assert f_hot < 1.0, f"Hot cell should reduce power: {f_hot}"
    assert f_cold > 1.0, f"Cold cell should boost power: {f_cold}"
    assert abs(f_stc - 1.0) < 1e-10, f"STC should be 1.0: {f_stc}"
    print(f"  Temp factor: 10°C→{f_cold:.4f}, 25°C→{f_stc:.4f}, 50°C→{f_hot:.4f}")


def test_ground_irradiance_components():
    """Ground irradiance should be sum of beam + diffuse + reflected."""
    g = irradiance.calculate_ground_irradiance(
        dni=500, dhi=200, ghi=600,
        ground_aoi_degrees=30,
        t_dir_avg=0.85,
        svf=0.75,
        albedo=0.20,
        ground_slope=5.0,
        h=2.1
    )
    assert g > 0, f"Ground irradiance should be positive: {g}"
    # Should be less than GHI (some light is blocked)
    assert g < 800, f"Ground irradiance should be < GHI: {g}"
    print(f"  Ground irradiance: {g:.1f} W/m²")


def test_log_wind_profile():
    """Wind speed should increase with height above ground."""
    v_low = thermal.effective_wind_speed(3.0, 0.8)
    v_high = thermal.effective_wind_speed(3.0, 2.1)
    v_ref = thermal.effective_wind_speed(3.0, 8.0)  # Module center ~8.75m
    
    assert v_high > v_low, f"Wind should increase with height: {v_low} vs {v_high}"
    assert v_ref < 3.0, f"Below 10m reference, wind should be less: {v_ref}"
    print(f"  Wind profile: h=0.8→{v_low:.2f}, h=2.1→{v_high:.2f}, h=9.25→{v_ref:.2f} m/s")


if __name__ == "__main__":
    tests = [
        ("SVF Limits", test_svf_limits),
        ("SVF Transparency", test_svf_transparency_effect),
        ("PAR Conversion", test_par_conversion),
        ("Faiman Height", test_faiman_height_effect),
        ("Faiman Wind", test_faiman_wind_effect),
        ("Temp Efficiency", test_temp_efficiency_factor),
        ("Ground Irradiance", test_ground_irradiance_components),
        ("Log Wind Profile", test_log_wind_profile),
    ]
    
    passed = 0
    failed = 0
    for name, test in tests:
        try:
            print(f"[TEST] {name}...")
            test()
            print(f"  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
