import numpy as np
from geometry import TableGeometry, GEOMETRY_PRESETS
import shading

def test_1_preset_12_projected_width_and_rise():
    p12 = GEOMETRY_PRESETS["Predefined Table 12°"]
    geo = TableGeometry(
        geometry_mode="Predefined Table 12°",
        tilt_deg=p12["tilt_deg"],
        clear_height_m=p12["clear_height_m"],
        surface_azimuth_deg=p12["surface_azimuth_deg"],
        table_length_m=p12["table_length_m"],
        row_pitch_m=p12["row_pitch_m"]
    )
    
    expected_proj = 5.75 * np.cos(np.radians(12.0))
    expected_rise = 5.75 * np.sin(np.radians(12.0))
    
    assert abs(geo.table_projected_width_m - expected_proj) < 1e-4
    assert abs(geo.table_vertical_rise_m - expected_rise) < 1e-4
    print("Test 1 passed: 12° projected width & rise")


def test_2_preset_12_high_edge_height():
    p12 = GEOMETRY_PRESETS["Predefined Table 12°"]
    geo = TableGeometry(
        geometry_mode="Predefined Table 12°",
        tilt_deg=p12["tilt_deg"],
        clear_height_m=p12["clear_height_m"],
        surface_azimuth_deg=p12["surface_azimuth_deg"],
        table_length_m=p12["table_length_m"],
        row_pitch_m=p12["row_pitch_m"]
    )
    
    expected_h_high = 2.70 + 5.75 * np.sin(np.radians(12.0))
    assert abs(geo.h_high_m - expected_h_high) < 1e-4
    print("Test 2 passed: 12° high edge height")


def test_3_preset_12_ground_gap():
    p12 = GEOMETRY_PRESETS["Predefined Table 12°"]
    geo = TableGeometry(
        geometry_mode="Predefined Table 12°",
        tilt_deg=p12["tilt_deg"],
        clear_height_m=p12["clear_height_m"],
        surface_azimuth_deg=p12["surface_azimuth_deg"],
        table_length_m=p12["table_length_m"],
        row_pitch_m=p12["row_pitch_m"]
    )
    
    expected_gap = 8.28 - geo.table_projected_width_m
    assert abs(geo.ground_gap_m - expected_gap) < 1e-4
    print("Test 3 passed: 12° ground gap")


def test_4_preset_15_ground_coverage_ratio():
    p15 = GEOMETRY_PRESETS["Predefined Table 15°"]
    geo = TableGeometry(
        geometry_mode="Predefined Table 15°",
        tilt_deg=p15["tilt_deg"],
        clear_height_m=p15["clear_height_m"],
        surface_azimuth_deg=p15["surface_azimuth_deg"],
        table_length_m=p15["table_length_m"],
        row_pitch_m=p15["row_pitch_m"]
    )
    
    expected_gcr = (5.63 * np.cos(np.radians(15.0))) / 8.63
    assert abs(geo.ground_coverage_ratio - expected_gcr) < 1e-4
    print("Test 4 passed: 15° ground coverage ratio")


def test_5_pitch_changes_geometry_and_spatial_grid():
    geo1 = TableGeometry(row_pitch_m=8.0)
    geo2 = TableGeometry(row_pitch_m=12.0)
    
    assert geo1.ground_gap_m != geo2.ground_gap_m
    assert geo1.ground_coverage_ratio != geo2.ground_coverage_ratio
    assert geo2.ground_gap_m > geo1.ground_gap_m
    assert geo2.ground_coverage_ratio < geo1.ground_coverage_ratio
    print("Test 5 passed: pitch scaling")


def test_6_table_length_changes_derived_geometry():
    geo1 = TableGeometry(table_length_m=5.0)
    geo2 = TableGeometry(table_length_m=6.5)
    
    assert geo2.table_projected_width_m > geo1.table_projected_width_m
    assert geo2.table_vertical_rise_m > geo1.table_vertical_rise_m
    assert geo2.h_high_m > geo1.h_high_m
    print("Test 6 passed: table length scaling")


def test_7_clearance_height_changes_h_low_and_h_high():
    geo1 = TableGeometry(clear_height_m=2.10)
    geo2 = TableGeometry(clear_height_m=3.00)
    
    assert geo1.h_low_m == 2.10
    assert geo2.h_low_m == 3.00
    assert abs((geo2.h_high_m - geo1.h_high_m) - 0.90) < 1e-4
    print("Test 7 passed: clearance height shift")


def test_8_surface_azimuth_changes_cross_row_shadow():
    geo = TableGeometry(surface_azimuth_deg=180.0) # South facing
    x_points = np.linspace(0, geo.row_pitch_m, 100)
    
    m1 = shading.calculate_spatial_mask(x_points, geo, solar_elev=30.0, solar_az=180.0, tau=0.20)
    m2 = shading.calculate_spatial_mask(x_points, geo, solar_elev=30.0, solar_az=90.0, tau=0.20)
    
    assert not np.array_equal(m1, m2)
    print("Test 8 passed: azimuth cross-row shadow shift")


def test_9_legacy_scenario_loading_fallback():
    legacy_empty_dict = {}
    geo_legacy = TableGeometry.from_dict(legacy_empty_dict)
    
    assert geo_legacy.geometry_mode == "Predefined Table 15°"
    assert geo_legacy.tilt_deg == 15.0
    assert geo_legacy.clear_height_m == 2.10
    assert geo_legacy.table_length_m == 5.63
    assert geo_legacy.row_pitch_m == 8.63
    print("Test 9 passed: legacy scenario 15° fallback")


if __name__ == "__main__":
    test_1_preset_12_projected_width_and_rise()
    test_2_preset_12_high_edge_height()
    test_3_preset_12_ground_gap()
    test_4_preset_15_ground_coverage_ratio()
    test_5_pitch_changes_geometry_and_spatial_grid()
    test_6_table_length_changes_derived_geometry()
    test_7_clearance_height_changes_h_low_and_h_high()
    test_8_surface_azimuth_changes_cross_row_shadow()
    test_9_legacy_scenario_loading_fallback()
    print("\nALL 9 GEOMETRY UNIT TESTS PASSED CLEANLY!")
