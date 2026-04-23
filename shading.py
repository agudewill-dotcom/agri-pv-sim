import numpy as np

def calculate_shadow_length(top_height, elevation, solar_azimuth=180.0, ground_slope=0.0, ground_aspect=180.0):
    """
    Computes shadow length on a sloped ground.
    
    elevation: Solar elevation in degrees.
    solar_azimuth: Solar azimuth in degrees.
    ground_slope: Ground slope in degrees (tilt of land).
    ground_aspect: Ground aspect in degrees (direction land faces).
    """
    if elevation <= 0.5:
        return 1e6 # Effectively infinite shadow
        
    elev_rad = np.radians(elevation)
    slope_rad = np.radians(ground_slope)
    
    # Difference in azimuth between sun and ground aspect
    az_diff_rad = np.radians(solar_azimuth - ground_aspect)
    
    # Effective slope in the direction of the shadow
    # tan(slope_eff) = tan(slope) * cos(az_diff)
    tan_slope_eff = np.tan(slope_rad) * np.cos(az_diff_rad)
    
    # Horizontal projection x of shadow:
    # x * tan(slope_eff) = H - x * tan(elev)
    # x(tan(elev) + tan(slope_eff)) = H
    # Note: tan(slope_eff) is positive if ground slopes UP in the shadow direction.
    # Shadows are cast AWAY from the sun. If sun is South, shadow is North.
    # If ground faces South (aspect=180), and sun is South (az=180), az_diff=0.
    # Shadow is cast North. Ground slopes DOWN North. 
    # So tan_slope_eff (North-down) should be used carefully.
    
    denom = np.tan(elev_rad) + tan_slope_eff
    
    if denom <= 0.01: # Sun ray parallel to or below ground slope
        return 1e6
        
    x_horiz = top_height / denom
    
    # Shadow length along the slope
    # We use the slope_rad directly for the hypotenuse if we assume the slope 
    # direction matches the shadow? For simplicity, we use the local slope magnitude.
    # But x_horiz is the N-S stretch.
    return x_horiz / np.cos(slope_rad)

def calculate_avg_direct_transmission(projected_width, tau, free_gap, shadow_length, pitch):
    """
    Compute average direct transmission across one pitch.
    """
    if pitch <= 0:
        return 0.0
        
    module_part = projected_width * tau
    gap_part = max(0, free_gap - shadow_length)
    
    t_avg = (module_part + gap_part) / pitch
    t_avg = (module_part + gap_part) / pitch
    return np.clip(t_avg, 0, 1)

def calculate_spatial_mask(x_points, top_height, clearance, length, tilt, solar_elev, solar_az, pitch, tau):
    """
    Returns transmittance array for x_points (0 to pitch).
    Simple 1D projection.
    """
    if solar_elev <= 0:
        return np.zeros_like(x_points)
        
    elev_rad = np.radians(solar_elev)
    # Horizontal shift of the shadow relative to the object
    # For a simple 2D cross-section, we look at the 'projected' solar elevation in the pitch direction
    # But let's use the shadow length logic:
    sh_len = calculate_shadow_length(top_height, solar_elev, solar_az)
    
    # Let's assume the module is centered in the pitch [0, P]
    proj_w = length * np.cos(np.radians(tilt))
    m_start = (pitch - proj_w) / 2
    m_end = m_start + proj_w
    
    # Shadow offset (simple approximation)
    # x_shadow = x_obj + top_height / tan(elev)
    # We'll use the sh_len as the displacement magnitude
    offset = sh_len * 0.8 # Empirical scaling for 1D profile
    
    transmittance = np.ones_like(x_points, dtype=float)
    
    # Check if x is within the projected shadow of any row
    # In periodic system, we check multiple row shadows
    for n in range(-1, 2):
        s_start = m_start + n*pitch + offset
        s_end = m_end + n*pitch + offset
        mask = (x_points >= s_start) & (x_points <= s_end)
        transmittance[mask] = tau
        
    return transmittance
