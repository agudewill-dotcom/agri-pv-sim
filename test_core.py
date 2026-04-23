import numpy as np
from geometry import calculate_derived_geometry
from shading import calculate_shadow_length, calculate_avg_direct_transmission
from irradiance import calculate_ground_irradiance

def test_geometry():
    print("Testing geometry...")
    # Tilt 20 deg, Length 5.79
    geo = calculate_derived_geometry(20.0, 5.79, 2.10)
    assert np.isclose(geo['projected_width'], 5.4409, atol=1e-3)
    assert np.isclose(geo['top_edge_height'], 4.0802, atol=1e-3)
    print("Geometry pass!")

def test_shading_flat():
    print("Testing shading (flat)...")
    # height 4.08, elevation 45 -> shadow = 4.08
    s = calculate_shadow_length(4.08, 45.0, 180.0, 0.0, 180.0)
    assert np.isclose(s, 4.08)
    print("Shading flat pass!")

def test_shading_slope():
    print("Testing shading (sloped)...")
    # height 4.08, elevation 45. Sun South (180).
    # Ground slopes UP South (aspect 180, slope 10). 
    # Shadow is North. Downhill.
    # tan(eff) = tan(45) + tan(10)*cos(0) = 1 + 0.176 = 1.176
    # x = 4.08 / 1.176 = 3.468
    # s = 3.468 / cos(10) = 3.468 / 0.984 = 3.52
    s = calculate_shadow_length(4.08, 45.0, 180.0, 10.0, 180.0)
    assert s < 4.08 # Shadow should be shorter when ground slopes "towards" the sun direction 
    print(f"Shading slope pass! (s={s:.2f})")

def test_irradiance():
    print("Testing irradiance flows...")
    # DNI 1000, DHI 100, AOI 0, t_dir 0.126, t_diff 0.75
    g = calculate_ground_irradiance(1000, 100, 0, 0.126, 0.75)
    assert g == 201.0
    print("Irradiance pass!")

if __name__ == "__main__":
    test_geometry()
    test_shading_flat()
    test_shading_slope()
    test_irradiance()
    print("\nAll Topography Core Tests Passed!")
