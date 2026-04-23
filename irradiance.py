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

def calculate_ground_irradiance(dni, dhi, ground_aoi_degrees, t_dir_avg, t_diffuse):
    """
    G_ground = DNI * cos(ground_aoi) * T_dir_avg + DHI * T_diffuse
    """
    aoi_rad = np.radians(ground_aoi_degrees)
    cos_aoi = np.cos(aoi_rad)
    
    # Direct beam on the specific ground plane orientation
    g_beam = dni * np.maximum(0, cos_aoi) * t_dir_avg
    
    # Diffuse light on ground. 
    # Note: For sloped ground, the sky view factor (SVF) changes.
    # SVF = (1 + cos(slope))/2
    # But user requirement 8 was DHI * T_diffuse. 
    # We will stick to that but could optionally scale by SVF.
    g_diff = dhi * t_diffuse
    
    return g_beam + g_diff

def calculate_par(g_ground, factor=2.1):
    """
    PAR = G_ground * factor
    """
    return g_ground * factor
