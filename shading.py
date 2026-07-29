import numpy as np
from geometry import TableGeometry

def calculate_shadow_length(top_height, elevation, solar_azimuth=180.0, ground_slope=0.0, ground_aspect=180.0):
    """
    Computes shadow displacement along the ground line.
    
    elevation: Solar elevation in degrees.
    solar_azimuth: Solar azimuth in degrees.
    ground_slope: Ground slope in degrees (tilt of land).
    ground_aspect: Ground aspect in degrees (direction land faces).
    """
    if elevation <= 0.5:
        return 1e6  # Effectively infinite shadow
        
    elev_rad = np.radians(elevation)
    slope_rad = np.radians(ground_slope)
    
    # Difference in azimuth between sun and ground aspect
    az_diff_rad = np.radians(solar_azimuth - ground_aspect)
    
    # Effective slope in the direction of the shadow
    tan_slope_eff = np.tan(slope_rad) * np.cos(az_diff_rad)
    
    denom = np.tan(elev_rad) + tan_slope_eff
    
    if denom <= 0.01:
        return 1e6
        
    x_horiz = top_height / denom
    return x_horiz / np.cos(slope_rad)


def calculate_cross_row_shadow_offset(height, solar_elev, solar_az, surface_az=180.0):
    """
    Calculates shadow displacement in the cross-row direction x taking into account solar elevation and azimuth relative to surface azimuth.
    """
    if solar_elev <= 0.5:
        return 1e6
        
    elev_rad = np.radians(solar_elev)
    az_diff_rad = np.radians(solar_az - surface_az)
    
    # Projection of shadow onto the cross-row normal axis
    return (height / np.tan(elev_rad)) * np.cos(az_diff_rad)


def calculate_periodic_shading_factor(projected_width, pitch_horizontal, aoi_mod, aoi_ground):
    """
    Computes the shaded fraction f of the ground in an infinite periodic array.
    f = min(1.0, (L * cos(AOI_mod)) / (P * cos(AOI_ground)))
    """
    cos_mod = np.cos(np.radians(aoi_mod))
    cos_ground = np.cos(np.radians(aoi_ground))
    
    if cos_ground <= 0.01:
        return 1.0
        
    ratio = (np.maximum(0, cos_mod) / cos_ground) * (projected_width / pitch_horizontal)
    return float(np.clip(ratio, 0.0, 1.0))


def calculate_avg_direct_transmission(shading_factor, tau_eff):
    """T_beam = (1 - f) * 1.0 + f * tau_eff"""
    return (1.0 - shading_factor) + (shading_factor * tau_eff)


def calculate_spatial_mask(x_points, geo: TableGeometry, solar_elev, solar_az, tau):
    """
    Returns transmittance array for x_points (0 to pitch) using exact TableGeometry cross-section bounds:
    Lower edge: x_low = 0, y = clear_height_m
    Upper edge: x_high = table_projected_width_m, y = h_high_m
    """
    if solar_elev <= 0.5:
        return np.zeros_like(x_points, dtype=float)
        
    elev_rad = np.radians(solar_elev)
    az_diff_rad = np.radians(solar_az - geo.surface_azimuth_deg)
    cos_az_diff = np.cos(az_diff_rad)
    
    # Shadow offsets for lower and upper edges
    offset_low = (geo.clear_height_m / np.tan(elev_rad)) * cos_az_diff
    offset_high = (geo.h_high_m / np.tan(elev_rad)) * cos_az_diff
    
    # Projected shadow bounds for a single table:
    # Lower edge at x=0 projects to x_s1 = 0 + offset_low
    # Upper edge at x=pw projects to x_s2 = pw + offset_high
    pw = geo.table_projected_width_m
    pitch = geo.row_pitch_m
    
    x_s_min = min(offset_low, pw + offset_high)
    x_s_max = max(offset_low, pw + offset_high)
    
    transmittance = np.ones_like(x_points, dtype=float)
    
    # Periodic boundary check over adjacent rows (-2 to 2)
    for n in range(-2, 3):
        row_offset = n * pitch
        s_start = row_offset + x_s_min
        s_end = row_offset + x_s_max
        mask = (x_points >= s_start) & (x_points <= s_end)
        transmittance[mask] = tau
        
    return transmittance
