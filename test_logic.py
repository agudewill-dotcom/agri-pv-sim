import numpy as np
import pandas as pd
from geometry import calculate_top_height, calculate_projected_width, calculate_profile_angle, calculate_shadow_length
from transmission import calculate_avg_direct_transmission, calculate_par
import solar

def test_geometry():
    print("Testing geometry...")
    # Tilt 30 deg, sloped length 10 -> height = 2 + 5 = 7
    h = calculate_top_height(2.0, 10.0, 30.0)
    assert np.isclose(h, 7.0), f"Height fail: Expected 7.0, got {h}"
    
    # projected width = 10 * cos(30) = 8.66
    w = calculate_projected_width(10.0, 30.0)
    assert np.isclose(w, 8.6602540378), f"Width fail: Expected 8.66..., got {w}"
    
    # Profile angle check
    # 1. Due South: phi should equal elevation
    phi_south = calculate_profile_angle(45.0, 180.0, 180.0)
    assert np.isclose(phi_south, 45.0), f"Due South fail: Expected 45.0, got {phi_south}"
    
    # 2. Due East/West: phi should head towards 90
    phi_east = calculate_profile_angle(45.0, 90.0, 180.0)
    assert phi_east == 90.0, f"Due East fail: Expected 90.0, got {phi_east}"

    # Shadow lengths
    s_45 = calculate_shadow_length(7.0, 45.0)
    assert np.isclose(s_45, 7.0), f"Shadow 45 fail: Expected 7.0, got {s_45}"
    
    s_low = calculate_shadow_length(7.0, 0.1)
    assert s_low == 1e6, f"Low sun fail: Expected 1e6, got {s_low}"
    
    print("Geometry tests passed!")

def test_transmission_bounds():
    print("Testing transmission bounds...")
    # p_width=5, gap=5, pitch=10, tau=0.2
    # Case A: No shadow (s=0)
    # T = (5*0.2 + 5)/10 = 6/10 = 0.6
    t_no = calculate_avg_direct_transmission(5.0, 0.2, 5.0, 0, 10.0)
    assert np.isclose(t_no, 0.6), f"Trans no-shade fail: Expected 0.6, got {t_no}"
    
    # Case B: Full gap shaded (s=5)
    # T = (5*0.2 + 0)/10 = 1/10 = 0.1
    t_full = calculate_avg_direct_transmission(5.0, 0.2, 5.0, 5.0, 10.0)
    assert np.isclose(t_full, 0.1), f"Trans full-gap-shade fail: Expected 0.1, got {t_full}"
    
    # Case C: Extreme shadow (s=100)
    # T = (5*0.2 + 0)/10 = 0.1
    t_ext = calculate_avg_direct_transmission(5.0, 0.2, 5.0, 100.0, 10.0)
    assert np.isclose(t_ext, 0.1), f"Trans ext-shade fail: Expected 0.1, got {t_ext}"
    
    print("Transmission tests passed!")

def test_solar_logic():
    print("Testing solar logic integration...")
    times = pd.date_range("2024-06-21 12:00:00", periods=1, freq='h', tz='UTC')
    df = solar.get_solar_data(52.5, 13.4, 'UTC', times)
    
    assert 'elevation' in df.columns, "Elevation column missing"
    assert 'azimuth' in df.columns, "Azimuth column missing"
    assert 'ghi' in df.columns, "GHI column missing"
    
    # Midday summer in Berlin: elevation should be high (> 50 deg)
    val = df['elevation'].iloc[0]
    assert val > 50, f"Solar elevation fail: Expected >50, got {val}"
    print("Solar logic passed!")

if __name__ == "__main__":
    try:
        test_geometry()
        test_transmission_bounds()
        test_solar_logic()
        print("\nAll core calculation tests passed successfully!")
    except AssertionError as e:
        print(f"\nAssertion error: {e}")
    except Exception as e:
        print(f"\nUnexpected error error: {e}")
