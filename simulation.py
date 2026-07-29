"""
simulation.py — Shared simulation engine for the multi-page Agri-PV dashboard.

Provides cached simulation results and derived KPIs that are shared across all pages.
All heavy computation happens once and is stored in st.session_state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import solar
import geometry
from geometry import TableGeometry, GEOMETRY_PRESETS
import irradiance
import thermal
import shading


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


# --- DYNAMIC CROSS-SECTION PREVIEW GRAPHIC ---
def create_cross_section_preview(geo: TableGeometry) -> go.Figure:
    """
    Renders a dynamic cross-section preview based strictly on physical TableGeometry.
    """
    fig = go.Figure()
    
    pw = geo.table_projected_width_m
    pitch = geo.row_pitch_m
    lh = geo.clear_height_m
    h_high = geo.h_high_m
    gap = geo.ground_gap_m
    
    x_max = pitch + pw * 0.35
    
    # Ground line (y = 0)
    fig.add_trace(go.Scatter(
        x=[-0.5, x_max], y=[0, 0],
        mode='lines', line=dict(color='#64748b', width=3),
        name='Ground Level', showlegend=False
    ))
    
    # Table 1 (Active)
    fig.add_trace(go.Scatter(
        x=[0, pw], y=[lh, h_high],
        mode='lines+markers',
        line=dict(color='#0284c7', width=5),
        marker=dict(size=6, color='#0369a1'),
        name='PV Table 1 (Active)', showlegend=False
    ))
    # Posts for Table 1
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[0, lh],
        mode='lines', line=dict(color='#94a3b8', width=2, dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[pw, pw], y=[0, h_high],
        mode='lines', line=dict(color='#94a3b8', width=2, dash='dot'),
        showlegend=False
    ))
    
    # Table 2 (Ghost Row)
    fig.add_trace(go.Scatter(
        x=[pitch, pitch + pw], y=[lh, h_high],
        mode='lines+markers',
        line=dict(color='#94a3b8', width=4, dash='dash'),
        marker=dict(size=5, color='#64748b'),
        name='PV Table 2 (Ghost)', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[pitch, pitch], y=[0, lh],
        mode='lines', line=dict(color='#cbd5e1', width=1.5, dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[pitch + pw, pitch + pw], y=[0, h_high],
        mode='lines', line=dict(color='#cbd5e1', width=1.5, dash='dot'),
        showlegend=False
    ))
    
    # Dimension Annotations
    fig.add_annotation(
        x=0, y=lh / 2,
        text=f"LH: {lh:.2f}m",
        showarrow=True, arrowhead=2, ax=-35, ay=0,
        font=dict(size=10, color="#0369a1")
    )
    fig.add_annotation(
        x=pw, y=h_high / 2,
        text=f"H_high: {h_high:.2f}m",
        showarrow=True, arrowhead=2, ax=40, ay=0,
        font=dict(size=10, color="#0369a1")
    )
    fig.add_annotation(
        x=pw / 2, y=(lh + h_high) / 2 + 0.35,
        text=f"L={geo.table_length_m:.2f}m @ {geo.tilt_deg:.1f}°",
        showarrow=False,
        font=dict(size=11, color="#0f172a", weight="bold")
    )
    if gap > 0:
        fig.add_annotation(
            x=pw + gap / 2, y=lh / 2,
            text=f"Gap: {gap:.2f}m",
            showarrow=False,
            font=dict(size=10, color="#16a34a", weight="bold"),
            bgcolor="rgba(220, 252, 231, 0.8)", bordercolor="#16a34a", borderwidth=1, borderpad=2
        )
    fig.add_annotation(
        x=pitch / 2, y=-0.4,
        text=f"Pitch: {pitch:.2f}m",
        showarrow=False,
        font=dict(size=10, color="#1e293b", weight="bold")
    )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(range=[-0.8, x_max + 0.2], showgrid=True, title="Cross-Section Width x [m]", zeroline=False),
        yaxis=dict(range=[-0.6, max(4.5, h_high + 0.6)], showgrid=True, title="Height y [m]", zeroline=False),
        plot_bgcolor="#f8fafc"
    )
    return fig


# --- PHYSICS ENGINE v9.0 ---
@st.cache_data
def run_v9_physics(lat, lon, yr, geo_dict: dict, gs, ga, tau, block, albedo=0.20):
    """
    v9.0 Physics Engine — Refactored to accept TableGeometry.
    """
    geo = TableGeometry.from_dict(geo_dict)
    
    df = solar.fetch_pvgis_hourly(lat, lon, yr, yr)
    sp = solar.get_solar_position_df(lat, lon, df.index)
    df = pd.concat([df, sp], axis=1)
    
    if 'wind_speed' not in df.columns:
        df['wind_speed'] = 1.0
        
    pw = geo.table_projected_width_m
    h_top = geo.h_high_m
    tilt = geo.tilt_deg
    clearance = geo.clear_height_m
    pitch = geo.row_pitch_m
    azimuth = geo.surface_azimuth_deg
    
    # Effective structural loss adjustment
    tau_struct = tau * (1.0 - geo.structural_loss_percent / 100.0)
    tau_eff = max(0, (pw - block) / pw) * tau_struct if pw > 0 else 0
    
    aoi_mod = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], tilt, azimuth)
    aoi_ground = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], gs, ga)
    
    def get_t_beam(aoi_m, aoi_g):
        cos_g = np.cos(np.radians(aoi_g))
        if cos_g <= 0.01: return tau_eff
        cos_m = max(0, np.cos(np.radians(aoi_m)))
        f_int = (geo.table_length_m * cos_m) / (pitch * cos_g)
        return 1.0 - min(1.0, f_int) * (1.0 - tau_eff)
        
    df['t_avg'] = [get_t_beam(m, g) for m, g in zip(aoi_mod, aoi_ground)]
    
    # Analytical View Factor (Hottel crossed-strings)
    svf_f = irradiance.sky_view_factor_periodic(h_top, pw, pitch, tau_eff, h_clearance=clearance)
    
    # Ground irradiance
    aoi = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], gs, ga)
    df['g_g'] = df.apply(lambda r: irradiance.calculate_ground_irradiance(
        r['dni'], r['dhi'], r['ghi'], aoi.loc[r.name], r['t_avg'], svf_f, albedo, gs, clearance
    ), axis=1)
    
    # PAR (McCree 1972)
    df['par'] = irradiance.calculate_par(df['g_g'])
    
    # Thermal model (Faiman 2008)
    svf_m = (1.0 + np.cos(np.radians(tilt))) / 2.0
    gvf_m = (1.0 - np.cos(np.radians(tilt))) / 2.0
    g_poa = df['dni'] * np.maximum(0, np.cos(np.radians(aoi_mod))) + df['dhi'] * svf_m + df['ghi'] * albedo * gvf_m
    df['g_poa'] = g_poa
    
    df['t_cell'] = thermal.cell_temperature_faiman(
        df['temp_air'], g_poa, df['wind_speed'], clearance
    )
    df['temp_factor'] = thermal.temperature_efficiency_factor(df['t_cell'])
    
    return df


def compute_derived_metrics(res_a, res_s):
    """Compute derived KPIs shared across pages."""
    va = res_a['g_g'].sum() / 1000.0
    vs = res_s['g_g'].sum() / 1000.0
    vo = res_a['ghi'].sum() / 1000.0
    
    pa = (res_a['par'] * 3600).sum() / 1e6
    ps = (res_s['par'] * 3600).sum() / 1e6
    po = (res_a['ghi'] * 2.1 * 3.6).sum() / 1e6
    
    par_open_field = (res_a['ghi'] * 0.45 * 4.57 * 3600).sum() / 1e6
    remaining_par_pct = (pa / par_open_field) * 100.0 if par_open_field > 0 else 0.0
    
    monthly_par_agri = []
    for m in range(1, 13):
        mask = res_a.index.month == m
        m_par = (res_a.loc[mask, 'par'] * 3600).sum() / 1e6
        monthly_par_agri.append(m_par)
    
    monthly_par_open = []
    for m in range(1, 13):
        mask = res_a.index.month == m
        m_par = (res_a.loc[mask, 'ghi'] * 0.45 * 4.57 * 3600).sum() / 1e6
        monthly_par_open.append(m_par)
    
    ya_spec = (res_a['g_poa'] * res_a['temp_factor']).sum() / 1000.0
    ys_spec = (res_s['g_poa'] * res_s['temp_factor']).sum() / 1000.0
    y_bonus = ya_spec - ys_spec
    
    ta_cell = res_a['t_cell'][res_a['ghi'] > 50].mean()
    ts_cell = res_s['t_cell'][res_s['ghi'] > 50].mean()
    delta_t = ts_cell - ta_cell
    temp_bonus_pct = (ya_spec / ys_spec - 1.0) * 100.0 if ys_spec > 0 else 0.0
    
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
    Returns (config_dict, res_a, res_s, metrics).
    """
    if 's' not in st.session_state:
        st.session_state.s = 0.0
    if 'a' not in st.session_state:
        st.session_state.a = 180.0
    
    if 'lat' not in st.session_state:
        st.session_state.lat = 52.5200
    if 'lon' not in st.session_state:
        st.session_state.lon = 13.4000
    
    st.sidebar.title("Simulation Setup")
    st.sidebar.markdown("**Project Site Coordinates**")
    
    coord_input = st.sidebar.text_input(
        "Coordinates (Lat, Lon)",
        value=f"{st.session_state.lat:.4f}, {st.session_state.lon:.4f}",
        help="Copy-paste coordinates directly (e.g. '52.5200, 13.4000')"
    )
    
    try:
        if ',' in coord_input:
            parts = coord_input.split(',')
            if len(parts) == 2:
                st.session_state.lat = float(parts[0].strip())
                st.session_state.lon = float(parts[1].strip())
    except ValueError:
        st.sidebar.error("Invalid coordinate format. Please use 'latitude, longitude' (e.g., 52.52, 13.40)")
        
    lat = st.session_state.lat
    lon = st.session_state.lon
    
    if st.sidebar.button("Fetch Satellite Topography"):
        st.session_state.s, st.session_state.a = get_topo(lat, lon)
        st.sidebar.success("Terrain Applied")
    
    st.sidebar.subheader("Terrain & Topography")
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
    
    # SECTION: PV TABLE GEOMETRY (CROSS-SECTION BASED)
    st.sidebar.subheader("PV-Tischgeometrie nach Schnitt")
    
    geom_mode = st.sidebar.selectbox(
        "Geometrie-Modus",
        options=["Predefined Table 12°", "Predefined Table 15°", "Custom Table Geometry"],
        key="geom_mode_select"
    )
    
    if geom_mode in GEOMETRY_PRESETS:
        preset_data = GEOMETRY_PRESETS[geom_mode]
        def_tilt = preset_data["tilt_deg"]
        def_lh = preset_data["clear_height_m"]
        def_len = preset_data["table_length_m"]
        def_pitch = preset_data["row_pitch_m"]
        def_az = preset_data["surface_azimuth_deg"]
        def_loss = preset_data.get("structural_loss_percent", 0.0)
        source_label = preset_data["source_label"]
    else:
        def_tilt = st.session_state.get("geo_tilt", 12.0)
        def_lh = st.session_state.get("geo_lh", 2.70)
        def_len = st.session_state.get("geo_len", 5.75)
        def_pitch = st.session_state.get("geo_pitch", 8.28)
        def_az = st.session_state.get("geo_az", 180.0)
        def_loss = st.session_state.get("geo_loss", 0.0)
        source_label = "Custom Table Geometry"

    tilt_deg = st.sidebar.number_input(
        "Neigung des PV-Tisches [°]",
        min_value=0.0, max_value=45.0, value=float(def_tilt), step=0.5, key="geo_tilt_input"
    )
    clear_height_m = st.sidebar.number_input(
        "Lichte Höhe LH / Unterkante Modul [m]",
        min_value=0.5, max_value=8.0, value=float(def_lh), step=0.05, key="geo_lh_input"
    )
    table_length_m = st.sidebar.number_input(
        "Tischlänge entlang Modulfläche [m]",
        min_value=0.5, max_value=30.0, value=float(def_len), step=0.05, key="geo_len_input"
    )
    row_pitch_m = st.sidebar.number_input(
        "Reihenabstand / Pitch [m]",
        min_value=1.0, max_value=50.0, value=float(def_pitch), step=0.05, key="geo_pitch_input"
    )
    surface_azimuth_deg = st.sidebar.number_input(
        "Azimut der Modulfläche [°]",
        min_value=0.0, max_value=360.0, value=float(def_az), step=1.0, key="geo_az_input"
    )
    structural_loss_percent = st.sidebar.number_input(
        "Struktureller Lichtverlust [%]",
        min_value=0.0, max_value=30.0, value=float(def_loss), step=0.1, key="geo_loss_input"
    )
    
    geo = TableGeometry(
        geometry_mode=geom_mode,
        tilt_deg=tilt_deg,
        clear_height_m=clear_height_m,
        surface_azimuth_deg=surface_azimuth_deg,
        table_length_m=table_length_m,
        row_pitch_m=row_pitch_m,
        structural_loss_percent=structural_loss_percent,
        source_label=source_label
    )
    
    # Validation Warnings
    if geo.ground_gap_m < 0:
        st.sidebar.error("Der Reihenabstand ist kleiner als die horizontale Projektion des PV-Tisches. Die Tischprojektionen überlappen.")
    elif geo.ground_gap_m < 1.0:
        st.sidebar.warning("Sehr geringer freier Gap zwischen den Tischprojektionen. Bewirtschaftung und Lichtverteilung können kritisch sein.")
        
    if geo.ground_coverage_ratio > 0.75:
        st.sidebar.warning("Sehr hoher Ground Coverage Ratio. Die Lichtverteilung kann stark eingeschränkt und heterogen sein.")
    if geo.clear_height_m < 1.5:
        st.sidebar.warning("Niedrige lichte Höhe. Durchlüftung, Bewirtschaftung und Arbeitshöhe können kritisch sein.")
    if geo.tilt_deg > 30:
        st.sidebar.warning("Hohe Neigung. Schattenlänge, Oberkante und Windangriffsfläche prüfen.")
    if geo.surface_azimuth_deg < 135 or geo.surface_azimuth_deg > 225:
        st.sidebar.info("Die Modulfläche ist nicht südorientiert. Schatten- und Ertragsprofil unterscheiden sich deutlich vom Standard-Südaufbau.")

    # Dynamic Cross-Section Preview
    fig_preview = create_cross_section_preview(geo)
    st.sidebar.plotly_chart(fig_preview, use_container_width=True)

    # Geometry KPI Cards
    st.sidebar.markdown("#### Geometrische Kennwerte (Schnitt)")
    gc1, gc2 = st.sidebar.columns(2)
    gc1.metric("Horiz. Projektion", f"{geo.table_projected_width_m:.2f} m")
    gc2.metric("Höhenversatz", f"{geo.table_vertical_rise_m:.2f} m")
    
    gc3, gc4 = st.sidebar.columns(2)
    gc3.metric("Oberkante Modul", f"{geo.h_high_m:.2f} m")
    gc4.metric("Freier Gap", f"{geo.ground_gap_m:.2f} m")
    
    gc5, gc6 = st.sidebar.columns(2)
    gc5.metric("Ground Coverage (GCR)", f"{geo.ground_coverage_ratio*100:.1f}%")
    gc6.metric("Reihen-Pitch", f"{geo.row_pitch_m:.2f} m")

    if geom_mode == "Predefined Table 12°":
        st.sidebar.caption("Referenz Schnitt 12°: LH 2,70 m | Tischlänge ca. 5,75 m | horizontale Projektion ca. 5,62 m | Pitch ca. 8,28 m | freier Gap ca. 2,63 m.")

    st.sidebar.divider()
    tau = st.sidebar.slider("Module Transparency (τ)", 0.0, 1.0, 0.20,
                             help="Fraction of light passing through the module.")
    
    ground_type = st.sidebar.selectbox(
        "Ground Surface Type",
        options=list(ALBEDO_PRESETS.keys()),
        help="Determines the albedo (reflection coefficient) of the ground."
    )
    albedo = ALBEDO_PRESETS[ground_type]
    
    # Run simulation
    geo_dict = geo.to_dict()
    geo_s_dict = geo.to_dict()
    geo_s_dict["clear_height_m"] = 0.80  # Standard PV reference height
    
    res_a = run_v9_physics(lat, lon, 2020, geo_dict, g_slope, g_aspect, tau, 0.81, albedo)
    res_s = run_v9_physics(lat, lon, 2020, geo_s_dict, g_slope, g_aspect, tau, 0.81, albedo)
    metrics = compute_derived_metrics(res_a, res_s)
    
    config = {
        'lat': lat, 'lon': lon,
        'g_slope': g_slope, 'g_aspect': g_aspect,
        'tau': tau, 'albedo': albedo, 'pitch': geo.row_pitch_m,
        'azimuth': geo.surface_azimuth_deg, 'tilt': geo.tilt_deg, 'height': geo.clear_height_m,
        'geometry': geo_dict
    }
    
    return config, res_a, res_s, metrics


@st.cache_data
def compute_spatial_annual_par(
    lat, lon, g_slope, g_aspect, tau, albedo, geo_dict: dict, n_points=200
):
    """
    Rigorous, vectorized 1D spatial simulation of ground irradiance and PAR
    across the pitch period [0, row_pitch_m] using exact TableGeometry.
    """
    geo = TableGeometry.from_dict(geo_dict)
    pitch = geo.row_pitch_m
    pw = geo.table_projected_width_m
    
    res_a = run_v9_physics(lat, lon, 2020, geo_dict, g_slope, g_aspect, tau, 0.81, albedo)
    
    x_points = np.linspace(0.0, pitch, n_points)
    
    elev = res_a['elevation'].values
    az = res_a['azimuth'].values
    dni = res_a['dni'].values
    dhi = res_a['dhi'].values
    ghi = res_a['ghi'].values
    
    tau_struct = tau * (1.0 - geo.structural_loss_percent / 100.0)
    tau_eff = max(0, (pw - 0.81) / pw) * tau_struct if pw > 0 else 0
    svf_f = irradiance.sky_view_factor_periodic(geo.h_high_m, pw, pitch, tau_eff, h_clearance=geo.clear_height_m)
    
    aoi_ground = irradiance.calculate_incidence_angle(res_a['zenith'], res_a['azimuth'], g_slope, g_aspect).values
    cos_g = np.cos(np.radians(aoi_ground))
    
    g_diff = dhi * svf_f * (1.0 + np.cos(np.radians(g_slope))) / 2.0
    g_refl = ghi * albedo * (1.0 - np.cos(np.radians(g_slope))) / 2.0 + ghi * albedo * (1.0 - svf_f) * 0.15
    
    spatial_par_annual = np.zeros(n_points)
    spatial_par_monthly = np.zeros((n_points, 12))
    
    for idx, x in enumerate(x_points):
        # Calculate hourly transmission mask using shading.calculate_spatial_mask
        # For efficiency, compute mask for each hour where elevation > 0.5
        t_beam = np.ones(len(elev), dtype=float)
        valid = elev > 0.5
        
        if np.any(valid):
            elev_rad = np.radians(elev[valid])
            az_diff_rad = np.radians(az[valid] - geo.surface_azimuth_deg)
            cos_az_diff = np.cos(az_diff_rad)
            
            offset_low = (geo.clear_height_m / np.tan(elev_rad)) * cos_az_diff
            offset_high = (geo.h_high_m / np.tan(elev_rad)) * cos_az_diff
            
            x_s_min = np.minimum(offset_low, pw + offset_high)
            x_s_max = np.maximum(offset_low, pw + offset_high)
            
            # Periodic rows -2 to 2
            is_shaded = np.zeros(np.sum(valid), dtype=bool)
            for n in range(-2, 3):
                row_offset = n * pitch
                s_start = row_offset + x_s_min
                s_end = row_offset + x_s_max
                is_shaded |= (x >= s_start) & (x <= s_end)
                
            t_beam[valid] = np.where(is_shaded, tau_struct, 1.0)
            
        t_beam[~valid] = 0.0
        
        g_beam = dni * np.maximum(0.0, cos_g) * t_beam
        g_ground = g_beam + g_diff + g_refl
        
        par_hourly = irradiance.calculate_par(g_ground)
        spatial_par_annual[idx] = (par_hourly * 3600.0).sum() / 1e6
        
        for m in range(1, 13):
            month_mask = res_a.index.month == m
            spatial_par_monthly[idx, m - 1] = (par_hourly[month_mask] * 3600.0).sum() / 1e6
            
    return x_points, spatial_par_annual, spatial_par_monthly
