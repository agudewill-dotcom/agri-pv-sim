import numpy as np
import pandas as pd
from geometry import calculate_derived_geometry, get_module_bounds
from transmission import calculate_avg_direct_transmission, calculate_par
import solar

def test_geometry():
    print("Testing geometry...")
    # Tilt 30 deg, sloped length 10 -> height = 2 + 5 = 7
    res = calculate_derived_geometry(30.0, length=10.0, clearance=2.0)
    h = res['top_edge_height']
    assert np.isclose(h, 7.0), f"Height fail: Expected 7.0, got {h}"
    
    # projected width = 10 * cos(30) = 8.66
    w = res['projected_width']
    assert np.isclose(w, 8.6602540378), f"Width fail: Expected 8.66..., got {w}"
    
    # get_module_bounds
    start, end = get_module_bounds(10.0, 8.0)
    assert start == 1.0, f"Bounds start fail: {start}"
    assert end == 9.0, f"Bounds end fail: {end}"
    
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
    df = solar.get_solar_position_df(52.5, 13.4, times)
    
    assert 'elevation' in df.columns, "Elevation column missing"
    assert 'azimuth' in df.columns, "Azimuth column missing"
    
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
