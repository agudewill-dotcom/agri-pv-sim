import numpy as np
import pvlib

def calculate_incidence_angle(solar_zenith, solar_azimuth, tilt_degrees, surface_azimuth=180.0):
    """
    Computes angle of incidence on any tilted surface (module or ground).
    """
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt_degrees,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth
    )
    return aoi

def calculate_ground_irradiance(dni, dhi, ghi, ground_aoi_degrees, t_dir_avg, t_diffuse_factor, albedo=0.2, ground_slope=0.0):
    """
    G_ground = Beam + Diffuse + Ground-Reflected (Albedo)
    """
    aoi_rad = np.radians(ground_aoi_degrees)
    cos_aoi = np.cos(aoi_rad)
    
    # Direct beam on the specific ground plane orientation
    g_beam = dni * np.maximum(0, cos_aoi) * t_dir_avg
    
    # View Factors for sloped terrain
    slope_rad = np.radians(ground_slope)
    svf = (1 + np.cos(slope_rad)) / 2
    gvf = (1 - np.cos(slope_rad)) / 2
    
    # Diffuse light on ground (Sky View Factor applied)
    g_diff = dhi * t_diffuse_factor * svf
    
    # Ground Reflected (Albedo) from surrounding terrain
    g_refl = ghi * albedo * gvf
    
    return g_beam + g_diff + g_refl

def calculate_par(g_ground, factor=2.1):
    """
    PAR = G_ground * factor
    """
    return g_ground * factor
