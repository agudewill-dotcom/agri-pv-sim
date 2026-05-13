"""
irradiance.py — Ground Irradiance Model for Agri-PV Simulation

Computes ground-level irradiance beneath periodic PV row arrays using
physically defensible radiative geometry methods:

A) Sky View Factor — Analytical integration over one pitch period using
   elevation angles subtended by adjacent rows (Hottel crossed-strings method).
B) Ground-reflected irradiance — Isotropic terrain albedo + first-order
   cavity inter-reflection with physical backsheet reflectance.
C) PAR conversion — McCree (1972) with explicit PAR spectral fraction.

References:
    Hottel, H.C. (1954). Radiant heat transmission. McGraw-Hill.
    Duffie, J.A. & Beckman, W.A. (2013). Solar Engineering of Thermal
        Processes, 4th Ed., Ch. 2.
    McCree, K.J. (1972). "The action spectrum, absorptance and quantum yield
        of photosynthesis in crop plants." Agricultural Meteorology, 9, 191-216.
"""

import numpy as np
import pvlib


def calculate_incidence_angle(solar_zenith, solar_azimuth, tilt_degrees, surface_azimuth=180.0):
    """
    Computes angle of incidence on any tilted surface (module or ground).
    Delegates to pvlib for validated spherical trigonometry.
    """
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt_degrees,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth
    )
    return aoi


def sky_view_factor_periodic(h_top, proj_width, pitch, tau_eff,
                             h_clearance=None, n_points=100):
    """
    Analytical Sky View Factor averaged over one pitch for infinite periodic rows.

    Each module row spans from h_clearance (bottom edge) to h_top (top edge).
    The gap between the ground and h_clearance is OPEN — diffuse light passes
    freely through this gap. Only the module band (h_clearance to h_top) 
    obstructs the sky, modulated by module transparency (tau_eff).

    For a ground point at distance x from a row, the module subtends an
    angular band:
        θ_module = arctan(h_top/x) - arctan(h_clearance/x)

    Higher clearance → module at higher elevation angle → smaller subtended
    angle → more sky visible → HIGHER SVF for Agri-PV systems.

    Parameters
    ----------
    h_top : float
        Height of the top edge of the module rows above ground [m].
    proj_width : float
        Horizontal projected width of one module row [m].
    pitch : float
        Row-to-row spacing (center to center) [m].
    tau_eff : float
        Effective module transparency (0-1). Accounts for structural blockage.
    h_clearance : float or None
        Height of the bottom edge (clearance) above ground [m].
        If None, defaults to 0 (modules touching ground — conservative).
    n_points : int
        Number of integration points across pitch. Default 100.

    Returns
    -------
    float
        Pitch-averaged sky view factor (0-1).

    Notes
    -----
    Height dependence emerges naturally from the angular geometry:
    - Same-size module at higher elevation subtends a SMALLER angle
    - Larger clearance gap allows more low-angle diffuse light through
    - This correctly produces higher SVF for elevated (Agri-PV) systems

    For very tall clearance (h_clearance → ∞): SVF → 1.0 (module far away)
    For zero clearance, opaque modules: SVF depends on h_top/pitch ratio
    """
    if pitch <= 0 or h_top <= 0:
        return 1.0

    if h_clearance is None:
        h_clearance = 0.0

    # Ensure h_clearance < h_top
    h_clearance = min(h_clearance, h_top - 0.01)
    h_clearance = max(h_clearance, 0.0)

    # Sample ground positions across one pitch period
    x = np.linspace(0.01 * pitch, 0.99 * pitch, n_points)

    # For each ground point, compute angular band blocked by left and right rows.
    # The module spans from h_clearance to h_top.
    # Angular band blocked = arctan(h_top/d) - arctan(h_clearance/d)
    # The gap below h_clearance is OPEN SKY.

    # Left row at distance x
    theta_top_left = np.arctan2(h_top, x)
    theta_bot_left = np.arctan2(h_clearance, x) if h_clearance > 0 else 0.0
    band_left = theta_top_left - theta_bot_left

    # Right row at distance (pitch - x)
    d_right = pitch - x
    theta_top_right = np.arctan2(h_top, d_right)
    theta_bot_right = np.arctan2(h_clearance, d_right) if h_clearance > 0 else 0.0
    band_right = theta_top_right - theta_bot_right

    # Total angular fraction of sky blocked (out of π hemisphere)
    # Module transparency allows tau_eff fraction of light through
    f_blocked = (band_left + band_right) / np.pi * (1.0 - tau_eff)

    svf_local = 1.0 - f_blocked
    svf_avg = np.mean(np.clip(svf_local, 0.0, 1.0))

    return float(np.clip(svf_avg, 0.0, 1.0))


def ground_reflected_irradiance(ghi, albedo, svf, ground_slope_rad,
                                rho_back=0.15):
    """
    Ground-reflected irradiance from terrain and cavity inter-reflection.

    Two physically distinct components:
    1. Terrain reflection: isotropic ground-reflected radiation from
       surrounding unshaded terrain (standard Liu & Jordan model).
    2. Cavity inter-reflection: first-order bounce between ground surface
       and module undersides. Bounded by module backsheet reflectance.

    Parameters
    ----------
    ghi : float or ndarray
        Global Horizontal Irradiance [W/m²].
    albedo : float
        Ground surface reflectance (0-1).
    svf : float
        Sky view factor for the ground point (0-1).
    ground_slope_rad : float
        Ground slope angle [radians].
    rho_back : float
        Module backsheet/underside reflectance. Default 0.15 for glass-glass.
        This is a measured material property, not a tuning parameter.

    Returns
    -------
    float or ndarray
        Total ground-reflected irradiance [W/m²].
    """
    # 1. Terrain reflection (standard isotropic model)
    gvf = (1.0 - np.cos(ground_slope_rad)) / 2.0
    g_terrain = ghi * albedo * gvf

    # 2. Cavity inter-reflection (first-order approximation)
    # Light hitting ground → reflects upward (albedo fraction)
    # Fraction (1-SVF) intercepts module undersides
    # Module undersides reflect rho_back fraction back to ground
    # This is the first term of a geometric series:
    # G_cavity = GHI * albedo * (1-SVF) * rho_back
    # Full series: sum = albedo*rho_back*(1-SVF) / (1 - albedo*rho_back*(1-SVF))
    # For typical values (albedo~0.2, rho_back~0.15), higher orders are <0.5% and negligible
    g_cavity = ghi * albedo * (1.0 - svf) * rho_back

    return g_terrain + g_cavity


def calculate_ground_irradiance(dni, dhi, ghi, ground_aoi_degrees,
                                t_dir_avg, svf, albedo=0.20,
                                ground_slope=0.0, h=1.0):
    """
    Total ground irradiance under the PV array.

    G_ground = G_beam + G_diffuse + G_reflected

    Components:
        G_beam    = DNI · cos(AOI_ground) · T_beam   (direct, shadow-adjusted)
        G_diffuse = DHI · SVF                         (diffuse sky, view-factor)
        G_refl    = terrain + cavity reflections       (albedo, inter-reflection)

    Parameters
    ----------
    dni, dhi, ghi : float or ndarray
        Direct Normal, Diffuse Horizontal, Global Horizontal irradiance [W/m²].
    ground_aoi_degrees : float or ndarray
        Angle of incidence on the ground surface [degrees].
    t_dir_avg : float or ndarray
        Pitch-averaged beam transmission factor (0-1).
    svf : float
        Sky view factor for the ground (0-1).
    albedo : float
        Ground surface reflectance.
    ground_slope : float
        Ground slope [degrees].
    h : float
        Module clearance height [m]. (Retained for API compatibility;
        height effects are now captured in SVF calculation.)

    Returns
    -------
    float or ndarray
        Total ground irradiance [W/m²].
    """
    aoi_rad = np.radians(ground_aoi_degrees)
    slope_rad = np.radians(ground_slope)

    # Direct beam on ground (shadow-attenuated)
    g_beam = dni * np.maximum(0, np.cos(aoi_rad)) * t_dir_avg

    # Diffuse sky irradiance (view-factor weighted)
    # Additional slope correction for diffuse
    svf_slope = (1.0 + np.cos(slope_rad)) / 2.0
    g_diffuse = dhi * svf * svf_slope

    # Ground-reflected irradiance (terrain + cavity)
    g_reflected = ground_reflected_irradiance(ghi, albedo, svf, slope_rad)

    return g_beam + g_diffuse + g_reflected


def calculate_par(g_ground, f_par=0.45, mccree_factor=4.57):
    """
    Convert broadband ground irradiance to Photosynthetically Active Radiation.

    PAR [µmol/m²/s] = G_ground [W/m²] × f_PAR × McCree_factor

    The PAR waveband (400-700nm) contains approximately 45% of total solar
    energy. Within this band, the average photon energy corresponds to
    4.57 µmol photons per Joule.

    Combined effective factor: 0.45 × 4.57 = 2.057 µmol/J (broadband)

    Parameters
    ----------
    g_ground : float or ndarray
        Broadband ground irradiance [W/m²].
    f_par : float
        Fraction of broadband irradiance in PAR waveband (400-700nm).
        Default 0.45 (typical clear-sky, per McCree 1972).
    mccree_factor : float
        Photon flux conversion for PAR-band radiation [µmol/J].
        Default 4.57 (McCree 1972).

    Returns
    -------
    float or ndarray
        PAR flux [µmol/m²/s].

    Notes
    -----
    This is a standard approximation. The actual PAR fraction varies with
    solar elevation, cloud cover, and atmospheric conditions. For a
    comparative model, a constant f_PAR is a widely accepted simplification.

    Reference: McCree (1972), Agricultural Meteorology, 9, 191-216.
    """
    return g_ground * f_par * mccree_factor
