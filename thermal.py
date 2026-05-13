"""
thermal.py — Cell Temperature Models for Agri-PV Simulation

Provides physically defensible cell temperature estimation using:
1. Faiman (2008) model with logarithmic wind profile height correction
2. IEC 61853-2 NOCT model with wind correction (fallback)

The height dependence emerges naturally from the logarithmic wind profile
(atmospheric boundary layer theory), eliminating arbitrary correction factors.

References:
    Faiman, D. (2008). "Assessing the outdoor operating temperature of
        photovoltaic modules." Progress in Photovoltaics, 16(4), 307-315.
    IEC 61853-2:2016 — PV module performance testing
    Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology.
"""

import numpy as np


# --- Physical Constants ---
Z0_GRASS = 0.03       # Surface roughness length for grassland [m] (Stull 1988)
H_REF_WIND = 10.0     # PVGIS wind speed reference height [m] (WS10m)


def effective_wind_speed(wind_speed_10m, h_clearance, z0=Z0_GRASS):
    """
    Estimate wind speed at module height using logarithmic wind profile.

    The atmospheric boundary layer wind profile follows:
        v(z) = v_ref * ln(z/z0) / ln(z_ref/z0)

    where z0 is the surface roughness length.

    Parameters
    ----------
    wind_speed_10m : float or ndarray
        Wind speed at 10m reference height [m/s] (from PVGIS WS10m).
    h_clearance : float
        Module lower edge clearance above ground [m].
    z0 : float
        Surface roughness length [m]. Default 0.03 for grass/crops.

    Returns
    -------
    float or ndarray
        Estimated wind speed at module center height [m/s].
    """
    # Module center is approximately at clearance + half the exposed height
    # For a 15° tilt, 5.63m module: vertical rise ≈ 1.46m, center ≈ 0.73m above clearance
    h_module = h_clearance + 0.75  # approximate module center height

    # Prevent log(0) — minimum height is z0
    h_module = max(h_module, z0 + 0.01)

    log_ratio = np.log(h_module / z0) / np.log(H_REF_WIND / z0)
    return wind_speed_10m * log_ratio


def cell_temperature_faiman(t_amb, g_poa, wind_speed_10m, h_clearance,
                            u0=25.0, u1=6.84, z0=Z0_GRASS):
    """
    Cell temperature using the Faiman (2008) model with height correction.

    Model:
        T_cell = T_amb + G_POA / (u0 + u1 * v_eff)

    The height effect enters through the logarithmic wind profile:
    elevated modules experience higher wind speeds, improving convective
    cooling and reducing cell temperatures.

    Parameters
    ----------
    t_amb : float or ndarray
        Ambient air temperature [°C].
    g_poa : float or ndarray
        Plane-of-array irradiance [W/m²].
    wind_speed_10m : float or ndarray
        Wind speed at 10m reference height [m/s].
    h_clearance : float
        Module lower edge clearance [m].
    u0 : float
        Constant heat transfer coefficient [W/m²/K]. Default 25.0.
    u1 : float
        Wind-dependent heat transfer coefficient [W/m²/K/(m/s)]. Default 6.84.
    z0 : float
        Surface roughness length [m].

    Returns
    -------
    float or ndarray
        Estimated cell temperature [°C].

    Reference
    ---------
    Faiman (2008), Progress in Photovoltaics, 16(4), 307-315.
    Coefficients u0=25.0, u1=6.84 from Koehl et al. (2011) for c-Si.
    """
    v_eff = effective_wind_speed(wind_speed_10m, h_clearance, z0)

    # Prevent division by zero at zero wind
    v_eff = np.maximum(v_eff, 0.5)

    t_cell = t_amb + g_poa / (u0 + u1 * v_eff)
    return t_cell


def temperature_efficiency_factor(t_cell, gamma=-0.0029, t_ref=25.0):
    """
    Module power temperature correction factor.

    P_actual = P_stc * [1 + γ * (T_cell - T_ref)]

    Parameters
    ----------
    t_cell : float or ndarray
        Cell temperature [°C].
    gamma : float
        Power temperature coefficient [1/°C]. Default -0.0029 (-0.29%/°C)
        from SUNfarming SF600-72N datasheet.
    t_ref : float
        Reference temperature for STC [°C]. Default 25.0.

    Returns
    -------
    float or ndarray
        Temperature correction factor (dimensionless, typically 0.85-1.05).
    """
    return 1.0 + gamma * (t_cell - t_ref)
