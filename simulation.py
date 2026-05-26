"""
simulation.py — Shared simulation engine for the multi-page Agri-PV dashboard.

Provides cached simulation results and derived KPIs that are shared across all pages.
All heavy computation happens once and is stored in st.session_state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import solar
import geometry
import irradiance
import thermal
import requests


# --- TOPOGRAPHY ---
@st.cache_data
def get_topo(lat, lon):
    """Fetch satellite topography (slope, aspect) from OpenTopoData SRTM30m."""
    d = 0.0005 
    locations = f"{lat},{lon}|{lat+d},{lon}|{lat-d},{lon}|{lat},{lon+d}|{lat},{lon-d}"
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={locations}"
    try:
        r = requests.get(url, timeout=5).json()['results']
        z_c, z_n, z_s, z_e, z_w = r[0]['elevation'], r[1]['elevation'], r[2]['elevation'], r[3]['elevation'], r[4]['elevation']
        dist = 111320 * d
        slope = np.degrees(np.arctan(np.sqrt(((z_e-z_w)/(2*dist))**2 + ((z_n-z_s)/(2*dist))**2)))
        aspect = (np.degrees(np.arctan2((z_e-z_w)/(2*dist), (z_n-z_s)/(2*dist))) + 360) % 360
        return round(slope, 1), round(aspect, 1)
    except:
        return 0.0, 180.0


# --- PHYSICS ENGINE v9.0 ---
@st.cache_data
def run_v9_physics(lat, lon, yr, l, h, p, gs, ga, tau, block, tilt, albedo=0.20):
    """
    v9.0 Physics Engine — Refactored for scientific defensibility.
    
    Changes from v8.3:
    - SVF: Analytical view factor integration (Hottel crossed-strings)
    - Thermal: Faiman (2008) model with log wind profile (uses PVGIS wind data)
    - PAR: Explicit McCree decomposition (f_PAR × McCree factor)
    - Albedo: Physical backsheet reflectance (rho_back=0.15)
    - Removed all arbitrary correction factors
    """
    df = solar.fetch_pvgis_hourly(lat, lon, yr, yr)
    sp = solar.get_solar_position_df(lat, lon, df.index)
    df = pd.concat([df, sp], axis=1)
    
    # Ensure wind speed column exists (PVGIS provides WS10m)
    if 'wind_speed' not in df.columns:
        df['wind_speed'] = 1.0  # Fallback if not available
    
    # --- A) GEOMETRY ---
    geo = geometry.calculate_derived_geometry(tilt, length=l, clearance=h)
    pw = geo['projected_width']
    h_top = geo['top_edge_height']
    
    # --- B) BEAM TRANSMISSION (AOI-ratio method) ---
    tau_eff = max(0, (pw - block) / pw) * tau if pw > 0 else 0
    aoi_mod = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], tilt, 180.0)
    aoi_ground = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], gs, ga)
    
    def get_t_beam(aoi_m, aoi_g):
        cos_g = np.cos(np.radians(aoi_g))
        if cos_g <= 0.01: return tau_eff
        cos_m = max(0, np.cos(np.radians(aoi_m)))
        f_int = (l * cos_m) / (p * cos_g)
        return 1.0 - min(1.0, f_int) * (1.0 - tau_eff)
        
    df['t_avg'] = [get_t_beam(m, g) for m, g in zip(aoi_mod, aoi_ground)]
    
    # --- C) DIFFUSE SKY — Analytical View Factor (Hottel) ---
    svf_f = irradiance.sky_view_factor_periodic(h_top, pw, p, tau_eff, h_clearance=h)
    
    # --- D) GROUND IRRADIANCE (beam + diffuse + reflected) ---
    aoi = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], gs, ga)
    df['g_g'] = df.apply(lambda r: irradiance.calculate_ground_irradiance(
        r['dni'], r['dhi'], r['ghi'], aoi.loc[r.name], r['t_avg'], svf_f, albedo, gs, h
    ), axis=1)
    
    # --- E) PAR (McCree 1972: f_PAR=0.45, factor=4.57 µmol/J) ---
    df['par'] = irradiance.calculate_par(df['g_g'])
    
    # --- F) THERMAL MODEL — Faiman (2008) with log wind profile ---
    aoi_mod = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], tilt, 180.0)
    svf_m = (1.0 + np.cos(np.radians(tilt))) / 2.0
    gvf_m = (1.0 - np.cos(np.radians(tilt))) / 2.0
    g_poa = df['dni'] * np.maximum(0, np.cos(np.radians(aoi_mod))) + df['dhi'] * svf_m + df['ghi'] * albedo * gvf_m
    df['g_poa'] = g_poa
    
    # Cell temperature via Faiman model (height enters through wind profile)
    df['t_cell'] = thermal.cell_temperature_faiman(
        df['temp_air'], g_poa, df['wind_speed'], h
    )
    
    # Temperature correction factor (γ = -0.29%/°C from SF600-72N datasheet)
    df['temp_factor'] = thermal.temperature_efficiency_factor(df['t_cell'])
    
    return df


def compute_derived_metrics(res_a, res_s):
    """
    Compute all derived KPIs from raw simulation DataFrames.
    Returns a dict of metrics shared across pages.
    """
    # Irradiance sums (kWh/m²)
    va = res_a['g_g'].sum() / 1000.0
    vs = res_s['g_g'].sum() / 1000.0
    vo = res_a['ghi'].sum() / 1000.0
    
    # PAR sums (mol/m²)
    pa = (res_a['par'] * 3600).sum() / 1e6
    ps = (res_s['par'] * 3600).sum() / 1e6
    po = (res_a['ghi'] * 2.1 * 3.6).sum() / 1e6  # Open field PAR approximation
    
    # Open-field PAR via McCree: GHI(W/m²) * f_PAR(0.45) * 4.57(µmol/J) → µmol/m²/s
    # Annual sum: sum(PPFD * 3600) / 1e6 → mol/m²
    par_open_field = (res_a['ghi'] * 0.45 * 4.57 * 3600).sum() / 1e6
    
    # Remaining PAR (%)
    remaining_par_pct = (pa / par_open_field) * 100.0 if par_open_field > 0 else 0.0
    
    # Monthly PAR sums for Agri-PV (mol/m²)
    monthly_par_agri = []
    for m in range(1, 13):
        mask = res_a.index.month == m
        m_par = (res_a.loc[mask, 'par'] * 3600).sum() / 1e6
        monthly_par_agri.append(m_par)
    
    # Monthly PAR open field
    monthly_par_open = []
    for m in range(1, 13):
        mask = res_a.index.month == m
        m_par = (res_a.loc[mask, 'ghi'] * 0.45 * 4.57 * 3600).sum() / 1e6
        monthly_par_open.append(m_par)
    
    # Specific Yield (kWh/kWp)
    ya_spec = (res_a['g_poa'] * res_a['temp_factor']).sum() / 1000.0
    ys_spec = (res_s['g_poa'] * res_s['temp_factor']).sum() / 1000.0
    y_bonus = ya_spec - ys_spec
    
    # Temperature statistics (daylight hours only)
    ta_cell = res_a['t_cell'][res_a['ghi'] > 50].mean()
    ts_cell = res_s['t_cell'][res_s['ghi'] > 50].mean()
    delta_t = ts_cell - ta_cell
    temp_bonus_pct = (ya_spec / ys_spec - 1.0) * 100.0 if ys_spec > 0 else 0.0
    
    # Spatial PAR variability (simplified from hourly data)
    # Use coefficient of variation of monthly remaining PAR ratios
    monthly_ratios = []
    for i in range(12):
        if monthly_par_open[i] > 0:
            monthly_ratios.append(monthly_par_agri[i] / monthly_par_open[i])
    cv_par = float(np.std(monthly_ratios) / np.mean(monthly_ratios)) if monthly_ratios else 0.15
    
    return {
        'va': va, 'vs': vs, 'vo': vo,
        'pa': pa, 'ps': ps, 'po': po,
        'par_open_field': par_open_field,
        'remaining_par_pct': remaining_par_pct,
        'monthly_par_agri': monthly_par_agri,
        'monthly_par_open': monthly_par_open,
        'ya_spec': ya_spec, 'ys_spec': ys_spec, 'y_bonus': y_bonus,
        'ta_cell': ta_cell, 'ts_cell': ts_cell,
        'delta_t': delta_t, 'temp_bonus_pct': temp_bonus_pct,
        'cv_par': cv_par,
    }


# --- SIDEBAR CONFIG (shared across pages) ---
ALBEDO_PRESETS = {
    "Green Grass (Agri-PV Standard)": 0.20,
    "Dry Soil / Tilled Field": 0.15,
    "Sand / Light Soil": 0.28,
    "Fresh Snow": 0.75,
    "Custom High-Reflectance": 0.40,
}

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def render_sidebar_and_run():
    """
    Render the shared sidebar configuration and run the simulation.
    Returns (config_dict, res_a, res_s, metrics) or uses session_state cache.
    """
    if 's' not in st.session_state:
        st.session_state.s = 0.0
    if 'a' not in st.session_state:
        st.session_state.a = 180.0
    
    st.sidebar.title("⚡ Simulation Setup")
    st.sidebar.markdown("**Project Site Coordinates**")
    c1, c2 = st.sidebar.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, 52.52, format="%.4f",
                          help="Geographic latitude of the project site.")
    lon = c2.number_input("Longitude", -180.0, 180.0, 13.40, format="%.4f",
                          help="Geographic longitude of the project site.")
    
    if st.sidebar.button("Fetch Satellite Topography"):
        st.session_state.s, st.session_state.a = get_topo(lat, lon)
        st.sidebar.success("Terrain Applied")
    
    st.sidebar.subheader("🌍 Terrain & Topography")
    use_manual = st.sidebar.toggle("Manual Terrain Override", value=False,
                                    help="Enable to manually set slope/aspect.")
    
    if not use_manual:
        g_slope = st.session_state.s
        g_aspect = st.session_state.a
        st.sidebar.info(f"**Satellite Active:** {g_slope}° Slope | {g_aspect}° Aspect")
    else:
        g_slope = st.sidebar.slider("Manual Site Slope (°)", 0.0, 20.0, st.session_state.s,
                                     help="The inclination of the ground surface.")
        g_aspect = st.sidebar.slider("Manual Site Aspect (°)", 0, 360, int(st.session_state.a),
                                      help="Compass direction the slope faces.")
    
    st.sidebar.divider()
    tau = st.sidebar.slider("Module Transparency (τ)", 0.0, 1.0, 0.20,
                             help="Fraction of light passing through the module.")
    
    ground_type = st.sidebar.selectbox(
        "Ground Surface Type",
        options=list(ALBEDO_PRESETS.keys()),
        help="Determines the albedo (reflection coefficient) of the ground."
    )
    albedo = ALBEDO_PRESETS[ground_type]
    
    pitch = st.sidebar.number_input("Design Pitch (m)", 5.0, 15.0, 8.63,
                                     help="Horizontal distance between adjacent module rows.")
    
    # Run simulation
    res_a = run_v9_physics(lat, lon, 2020, 5.63, 2.10, pitch, g_slope, g_aspect, tau, 0.81, 15, albedo)
    res_s = run_v9_physics(lat, lon, 2020, 5.63, 0.80, pitch, g_slope, g_aspect, tau, 0.81, 15, albedo)
    metrics = compute_derived_metrics(res_a, res_s)
    
    config = {
        'lat': lat, 'lon': lon,
        'g_slope': g_slope, 'g_aspect': g_aspect,
        'tau': tau, 'albedo': albedo, 'pitch': pitch,
    }
    
    return config, res_a, res_s, metrics


@st.cache_data
def compute_spatial_annual_par(
    lat, lon, g_slope, g_aspect, tau, albedo, pitch, n_points=11
):
    """
    Rigorous, vectorized 1D spatial simulation of ground irradiance and PAR
    across the pitch period [0, pitch] using v9.0 physical formulas.
    Cached based on simulation parameters.
    """
    # Fetch base hourly data (uses streamlit cache internally)
    res_a = run_v9_physics(lat, lon, 2020, 5.63, 2.10, pitch, g_slope, g_aspect, tau, 0.81, 15, albedo)
    
    x_points = np.linspace(0.0, pitch, n_points)
    
    # Pre-extract solar position arrays (length 8760)
    elev = res_a['elevation'].values
    az = res_a['azimuth'].values
    dni = res_a['dni'].values
    dhi = res_a['dhi'].values
    ghi = res_a['ghi'].values
    
    # Calculate geometric details
    geo = geometry.calculate_derived_geometry(15.0, length=5.63, clearance=2.10)
    proj_w = geo['projected_width']
    h_top = geo['top_edge_height']
    m_start = (pitch - proj_w) / 2
    m_end = m_start + proj_w
    
    # Effective beam transparency
    tau_eff = max(0, (proj_w - 0.81) / proj_w) * tau if proj_w > 0 else 0
    svf_f = irradiance.sky_view_factor_periodic(h_top, proj_w, pitch, tau_eff, h_clearance=2.10)
    
    # Calculate aoi_ground for all hours
    aoi_ground = irradiance.calculate_incidence_angle(res_a['zenith'], res_a['azimuth'], g_slope, g_aspect).values
    cos_g = np.cos(np.radians(aoi_ground))
    
    # Uniform diffuse and reflected ground irradiance components (8760, )
    g_diff = dhi * svf_f * (1.0 + np.cos(np.radians(g_slope))) / 2.0
    g_refl = ghi * albedo * (1.0 - np.cos(np.radians(g_slope))) / 2.0 + ghi * albedo * (1.0 - svf_f) * 0.15
    
    # Output arrays
    spatial_par_annual = np.zeros(n_points)
    spatial_par_monthly = np.zeros((n_points, 12))
    
    # Vectorized loop over points to ensure efficiency
    # For each point x, we calculate the hourly beam transmission mask
    for idx, x in enumerate(x_points):
        # Hourly shadow offset (8760, )
        # Using vectorized shadow length:
        sh_len = np.zeros_like(elev)
        valid = elev > 0.5
        elev_rad = np.radians(elev[valid])
        slope_rad = np.radians(g_slope)
        az_diff_rad = np.radians(az[valid] - g_aspect)
        tan_slope_eff = np.tan(slope_rad) * np.cos(az_diff_rad)
        denom = np.tan(elev_rad) + tan_slope_eff
        
        # Avoid division by zero
        denom_mask = denom > 0.01
        sh_len[valid] = np.where(denom_mask, h_top / np.where(denom_mask, denom, 1.0) / np.cos(slope_rad), 1e6)
        sh_len[~valid] = 1e6
        
        # Check if x is in shadow for each hour
        # Periodicity: check rows -1, 0, 1
        is_shaded = np.zeros(len(elev), dtype=bool)
        for n in range(-1, 2):
            s_start = m_start + n * pitch + sh_len
            s_end = m_end + n * pitch + sh_len
            is_shaded |= (x >= s_start) & (x <= s_end)
            
        # Beam transmission mask at x
        t_beam = np.where(is_shaded, tau, 1.0)
        # Handle night
        t_beam[elev <= 0] = 0.0
        
        # Beam irradiance at x
        g_beam = dni * np.maximum(0.0, cos_g) * t_beam
        
        # Total ground irradiance at x
        g_ground = g_beam + g_diff + g_refl
        
        # Convert to PAR (McCree) -> µmol/m²/s
        par_hourly = irradiance.calculate_par(g_ground)
        
        # Annual cumulative PAR in mol/m²
        spatial_par_annual[idx] = (par_hourly * 3600.0).sum() / 1e6
        
        # Monthly cumulative PAR in mol/m²
        for m in range(1, 13):
            month_mask = res_a.index.month == m
            spatial_par_monthly[idx, m - 1] = (par_hourly[month_mask] * 3600.0).sum() / 1e6
            
    return x_points, spatial_par_annual, spatial_par_monthly
