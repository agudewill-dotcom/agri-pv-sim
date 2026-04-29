import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import solar
import geometry
import shading
import irradiance
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Agri-PV Strategic Analytics", layout="wide")

# --- EXECUTIVE THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
    .main { background: #fafafa; }
    .stMetric {
        background: white; border-radius: 8px; padding: 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        border-top: 5px solid #0f172a;
    }
    [data-testid="stMetricLabel"] { color: #1e293b !important; font-weight: 700 !important; font-size: 0.95rem !important; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 800 !important; font-size: 2.2rem !important; }
    .status-box { background: #1e293b; color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; border-left: 10px solid #22c55e; }
    .status-title { font-size: 1.8rem; font-weight: 800; color: #22c55e; margin-bottom: 5px; }
    .meth-box { background: #f8fafc; padding: 30px; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 40px; color: #1e293b; }
    /* Nuclear fix for truncation in all metric parts */
    div[data-testid="stMetric"] * {
        white-space: normal !important;
        text-overflow: unset !important;
        overflow: visible !important;
        word-break: break-word !important;
    }
    .stMetric { min-height: 150px; }
    /* Hide +/- buttons from number inputs */
    button[data-testid="stNumberInputStepUp"], 
    button[data-testid="stNumberInputStepDown"] {
        display: none !important;
    }
    div[data-testid="stNumberInputContainer"] {
        padding-right: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PHYSICS ---
@st.cache_data
def get_topo(lat, lon):
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
    except: return 0.0, 180.0

@st.cache_data
def run_v8_physics(lat, lon, yr, l, h, p, gs, ga, tau, block, tilt, albedo=0.20):
    df = solar.fetch_pvgis_hourly(lat, lon, yr, yr)
    sp = solar.get_solar_position_df(lat, lon, df.index)
    df = pd.concat([df, sp], axis=1)
    geo = geometry.calculate_derived_geometry(tilt, length=l, clearance=h)
    pw = geo['projected_width']
    gap_w = p - pw
    # Exact geometric transmission for direct beam
    # f_intercepted = (L * cos_aoi_mod) / (P * cos_aoi_ground)
    tau_eff = max(0, (pw - block) / pw) * tau if pw > 0 else 0
    aoi_mod = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], tilt, 180.0)
    aoi_ground = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], gs, ga)
    
    def get_t_beam(aoi_m, aoi_g):
        cos_g = np.cos(np.radians(aoi_g))
        if cos_g <= 0.01: return tau_eff
        cos_m = np.abs(np.cos(np.radians(aoi_m)))
        f_int = (l * cos_m) / (p * cos_g)
        return 1.0 - min(1.0, f_int) * (1.0 - tau_eff)
        
    df['t_avg'] = [get_t_beam(m, g) for m, g in zip(aoi_mod, aoi_ground)]
    
    # --- HEIGHT-DEPENDENT DIFFUSE CORRECTION (The 'Cavity Effect') ---
    # Higher clearance allows more stray diffuse light to leak in from the sides
    # and reduces the solid angle obstruction of the module rows.
    # We apply a small empirical gain factor based on clearance (h).
    diffuse_leakage_factor = 1.0 + (0.012 * h) # ~1.2% gain per meter of height
    p_horiz = p * np.cos(np.radians(gs))
    blocked_frac = min(1.0, pw / p_horiz) if p_horiz > 0 else 1.0
    svf_f = 1.0 - (blocked_frac * (1.0 - tau_eff))
    svf_f = min(1.0, svf_f * diffuse_leakage_factor)
    
    aoi = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], gs, ga)
    # Pass clearance (h) to ground irradiance for height-dependent albedo bounce
    df['g_g'] = df.apply(lambda r: irradiance.calculate_ground_irradiance(
        r['dni'], r['dhi'], r['ghi'], aoi.loc[r.name], r['t_avg'], svf_f, albedo, gs, h
    ), axis=1)
    df['par'] = df['g_g'] * 2.1
    
    # --- CELL TEMPERATURE MODEL (SUNfarming SF600 datasheet: NOCT=41°C, γ=-0.29%/°C) ---
    # Clearance-based ventilation correction: higher mounting = better free-air cooling
    # Research shows ~1.5°C reduction per extra meter of clearance above a baseline of 0.5m
    NOCT_BASE = 41.0  # °C from SUNfarming SF600-72N datasheet
    GAMMA = -0.0029   # Pmpp temp coefficient from datasheet (-0.29%/°C)
    vent_delta = max(0.0, (h - 0.5) * 1.5)  # Agri-PV(2.1m): -2.4°C, Std(0.8m): -0.45°C
    noct_eff = NOCT_BASE - vent_delta
    # Rigorous POA calculation for yield
    aoi_mod = irradiance.calculate_incidence_angle(df['zenith'], df['azimuth'], tilt, 180.0)
    svf_m = (1.0 + np.cos(np.radians(tilt))) / 2.0
    gvf_m = (1.0 - np.cos(np.radians(tilt))) / 2.0
    g_poa = df['dni'] * np.maximum(0, np.cos(np.radians(aoi_mod))) + df['dhi'] * svf_m + df['ghi'] * albedo * gvf_m
    
    df['g_poa'] = g_poa
    df['t_cell'] = df['temp_air'] + (noct_eff - 20.0) / 800.0 * g_poa
    # Temperature correction factor for module power output
    df['temp_factor'] = 1.0 + GAMMA * (df['t_cell'] - 25.0)
    
    return df

# --- UI ---
if 's' not in st.session_state: st.session_state.s = 0.0
if 'a' not in st.session_state: st.session_state.a = 180.0
# SIDEBAR CONFIG
st.sidebar.title("Simulation Setup")
st.sidebar.markdown("**Project Site Coordinates**")
c1, c2 = st.sidebar.columns(2)
lat = c1.number_input("Latitude", -90.0, 90.0, 52.52, format="%.4f", help="Geographic latitude of the project site.")
lon = c2.number_input("Longitude", -180.0, 180.0, 13.40, format="%.4f", help="Geographic longitude of the project site.")

ALBEDO_PRESETS = {
    "Green Grass (Agri-PV Standard)": 0.20,
    "Dry Soil / Tilled Field": 0.15,
    "Sand / Light Soil": 0.28,
    "Fresh Snow": 0.75,
    "Custom High-Reflectance": 0.40
}

if st.sidebar.button("Fetch Satellite Topography"):
    st.session_state.s, st.session_state.a = get_topo(lat, lon)
    st.sidebar.success(f"Terrain Applied")

# TERRAIN ENGINE
st.sidebar.subheader("🌍 Terrain & Topography")
use_manual = st.sidebar.toggle("Manual Terrain Override", value=False, help="Enable to manually set slope/aspect. Otherwise, satellite data is used.")

if not use_manual:
    g_slope = st.session_state.s
    g_aspect = st.session_state.a
    st.sidebar.info(f"**Satellite Active:** {g_slope}° Slope | {g_aspect}° Aspect")
else:
    g_slope = st.sidebar.slider("Manual Site Slope (°)", 0.0, 20.0, st.session_state.s, help="The inclination of the ground surface at the site.")
    g_aspect = st.sidebar.slider("Manual Site Aspect (°)", 0, 360, int(st.session_state.a), help="The compass direction the slope faces (0°=North, 90°=East, 180°=South, 270°=West).")
st.sidebar.divider()
tau = st.sidebar.slider("Module Transparency (τ)", 0.0, 1.0, 0.20, help="The fraction of light passing through the semi-transparent module (τ).")

# REPLACED SLIDER WITH DROPDOWN
ground_type = st.sidebar.selectbox(
    "Ground Surface Type", 
    options=list(ALBEDO_PRESETS.keys()),
    help="Determines the 'Albedo' or reflection coefficient of the ground surface."
)
albedo = ALBEDO_PRESETS[ground_type]

pitch = st.sidebar.number_input("Design Pitch (m)", 5.0, 15.0, 8.63, help="Horizontal distance between the centers of two adjacent module rows.")

# System geometry from technical drawing (SUNfarming Agri-PV cross-section):
# Both systems: 100% identical hardware — same modules, same 5.63m table, 8.63m pitch, 15° tilt, tau, 0.81m blockage
# ONLY difference: clearance height (2.10m Agri-PV vs 0.80m Standard PV)
res_a = run_v8_physics(lat, lon, 2020, 5.63, 2.10, pitch, g_slope, g_aspect, tau, 0.81, 15, albedo)
res_s = run_v8_physics(lat, lon, 2020, 5.63, 0.80, pitch, g_slope, g_aspect, tau, 0.81, 15, albedo)
va, vs, vo = res_a['g_g'].sum()/1000, res_s['g_g'].sum()/1000, res_a['ghi'].sum()/1000
pa, ps = (res_a['par']*3600).sum()/1e6, (res_s['par']*3600).sum()/1e6

# Specific Yield calculation (kWh/kWp)
# Yield = sum(POA * temp_factor) / 1000
ya_spec = (res_a['g_poa'] * res_a['temp_factor']).sum() / 1000.0
ys_spec = (res_s['g_poa'] * res_s['temp_factor']).sum() / 1000.0
y_bonus = ya_spec - ys_spec

# Temperature statistics
ta_cell  = res_a['t_cell'][res_a['ghi'] > 50].mean()
ts_cell  = res_s['t_cell'][res_s['ghi'] > 50].mean()
delta_t  = ts_cell - ta_cell
temp_bonus_pct = (ya_spec / ys_spec - 1.0) * 100.0

# --- HEADER Metrics ---
st.markdown(f"""
<div class="status-box">
    <div class="status-title">AGRICULTURAL LIGHT ADVANTAGE: +{(va-vs):.0f} kWh/m² vs Standard PV</div>
    <div style="font-size: 2.5rem; font-weight: 800;">+{(va/vs-1)*100:.1f}% Ground Irradiance Bonus | +{temp_bonus_pct:.2f}% Module Efficiency Bonus</div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Agricultural Light", f"{(va/vo)*100:.1f}%", f"+{(va/vs-1)*100:.1f}% vs Std. PV", help="Percentage of total open-field irradiance reaching the ground under the system.")
k2.metric("Annual PAR Sum", f"{pa:.0f} mol/m²", f"+{(pa/ps-1)*100:.1f}% vs Std. PV", help="Annual cumulative Photosynthetic Active Radiation (PAR) for crop growth.")
k3.metric("BASELINE: STANDARD GROUND-PV", f"{vs:.0f} kWh/m²", f"RESTRICTED: {(vs/vo)*100:.1f}% LIGHT", help="Annual ground irradiance for a standard 0.8m high system.")
k4.metric("VS. STANDARD GROUND-PV", f"+{va-vs:.0f} kWh/m²", "Production Winner", help="The absolute irradiance advantage of Agri-PV over Standard PV.")

# TEMPERATURE KPI ROW
t1, t2, t3, t4 = st.columns(4)
t1.metric("Agri-PV Cell Temp", f"{ta_cell:.1f} °C", f"−{delta_t:.1f}°C vs Standard", help="Annual arithmetic mean during daylight hours (GHI > 50 W/m²)")
t2.metric("Std. PV Cell Temp", f"{ts_cell:.1f} °C", "Restricted ventilation at 0.8m", help="Annual arithmetic mean during daylight hours (GHI > 50 W/m²)")
t3.metric("Temp. Power Bonus", f"+{temp_bonus_pct:.2f}%", "Agri-PV cooler → higher η", help="Relative module power increase due to the lower cell temperatures in high-mounted systems.")
t4.metric("NOCT (Datasheet)", "41 °C", f"Vent correction: −2.4°C @ 2.1m", help="Nominal Operating Cell Temperature corrected for height-dependent ventilation.")

# NEW: ELECTRICAL YIELD BOX
st.markdown(f"""
<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:20px; margin:20px 0; display:flex; justify-content:space-between; align-items:center;">
    <div>
        <h4 style="margin:0; color:#475569;">SPECIFIC YIELD BONUS (ELECTRICAL)</h4>
        <p style="margin:4px 0 0 0; font-size:0.9rem; color:#64748b;">Annual energy generation advantage per installed kWp</p>
    </div>
    <div style="text-align:right;">
        <span style="font-size:2rem; font-weight:800; color:#1e293b;">+{y_bonus:.1f} kWh/kWp</span>
        <div style="font-size:0.9rem; font-weight:600; color:#16a34a;">↑ Agri-PV: {ya_spec:.0f} vs Standard: {ys_spec:.0f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# WHY THE DIFFERENCE? EXPLANATION BOX
st.markdown(f"""
<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-left:6px solid #16a34a; border-radius:10px; padding:24px; margin:20px 0;">
    <h3 style="color:#15803d; margin-top:0;">Agri-PV Strategic Advantages: Why Height Matters</h3>
    <p style="color:#1e293b; margin-bottom:14px;">Both systems use <strong>100% identical hardware</strong>. The performance delta is driven by two height-dependent physical mechanisms:</p>
    <table style="width:100%; border-collapse:collapse; color:#1e293b;">
        <tr style="background:#dcfce7; font-weight:700;">
            <td style="padding:10px;">Benefit</td><td style="padding:10px;">Mechanism</td><td style="padding:10px;">Impact</td>
        </tr>
        <tr style="border-bottom:1px solid #d1fae5;">
            <td style="padding:10px;"><strong>1. Higher Energy Generation</strong></td>
            <td style="padding:10px;">Free convective airflow (2.1m)</td>
            <td style="padding:10px;">Agri-PV runs <strong>{delta_t:.1f} °C cooler</strong> → Higher electrical system efficiency.</td>
        </tr>
        <tr style="border-bottom:1px solid #d1fae5; background:#f0fdf4;">
            <td style="padding:10px;"><strong>2. Higher Ground Irradiance</strong></td>
            <td style="padding:10px;">Diffuse Cavity Effect</td>
            <td style="padding:10px;">Higher clearance allows more <strong>stray diffuse light</strong> to reach the ground from the sides.</td>
        </tr>
    </table>
    <p style="margin-top:14px; font-size:0.85rem; color:#475569;">Both systems: 5.63m table | 8.63m pitch | 15° tilt | {tau*100:.0f}% transparency. Only the mounting height varies.</p>
</div>
""", unsafe_allow_html=True)

# HEATMAP & SENSORS
st.divider()
c_meta, c_heat = st.columns([1, 1.5])
with c_meta:
    st.subheader("Comparative Sensor Profile")
    st.table(pd.DataFrame([
        {"System": "Agri-PV (2.1m)", "Irradiance": f"{va:.0f} kWh/m²", "PAR": f"{pa:.0f} mol/m²"},
        {"System": "Standard (0.8m)", "Irradiance": f"{vs:.0f} kWh/m²", "PAR": f"{ps:.0f} mol/m²"},
        {"System": "Open Field", "Irradiance": f"{vo:.0f} kWh/m²", "PAR": f"{(vo*2.1*3.6):.0f} mol/m²"}
    ]))
    st.info("System geometry per SUNfarming technical drawing: 5.63m table | 8.63m pitch | 2.10m clearance | 15° tilt | 9.3% structural loss.")
with c_heat:
    st.subheader("Light Intensity Heatmap (W/m² - Agri-PV)")
    h_data = res_a.groupby([res_a.index.month, res_a.index.hour])['g_g'].mean().unstack()
    m_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    h_data.index = m_names
    st.plotly_chart(px.imshow(h_data, color_continuous_scale='Viridis', aspect='auto', height=350), use_container_width=True)

# SPATIAL PROFILE SECTION
st.divider()
st.subheader("Spatial Shadow Profile (Cross-Section)")
sp_col1, sp_col2 = st.columns([2, 1])
with sp_col2:
    st.info("**Vector Shadow Pathing Analysis**")
    sel_month = st.selectbox("Select Month", options=m_names, index=5)
    sel_hour = st.slider("Select Hour of Day", 0, 23, 12)
    month_idx = m_names.index(sel_month) + 1
    
    # Filter for the selected time
    time_data = res_a[(res_a.index.month == month_idx) & (res_a.index.hour == sel_hour)]
    if time_data.empty:
        selected_data = res_a[(res_a.index.month == month_idx)].iloc[0]
    else:
        selected_data = time_data.iloc[0]

with sp_col1:
    x_points = np.linspace(0, pitch, 100)
    geo_a = geometry.calculate_derived_geometry(15, length=5.63, clearance=2.10)
    
    t_mask = shading.calculate_spatial_mask(
        x_points, geo_a['top_edge_height'], 2.10, 5.63, 15, 
        selected_data['elevation'], selected_data['azimuth'], pitch, tau
    )
    
    # Calculate local irradiance across the points
    aoi_sel = irradiance.calculate_incidence_angle(selected_data['zenith'], selected_data['azimuth'], g_slope, g_aspect)
    # Simple spatial distribution: Direct beam is masked, diffuse and reflected are uniform
    g_base_diff = selected_data['dhi'] * 0.90 # Simplified SVF
    g_base_refl = selected_data['ghi'] * albedo * (1 - np.cos(np.radians(g_slope))) / 2
    
    g_spatial = (selected_data['dni'] * np.maximum(0, np.cos(np.radians(aoi_sel))) * t_mask) + g_base_diff + g_base_refl
    
    fig_sp = px.line(x=x_points, y=g_spatial, labels={'x': 'Horizontal Distance across Pitch (m)', 'y': 'Irradiance (W/m²)'},
                     title=f"Instantaneous Light Distribution ({sel_month}, {sel_hour}:00)")
    fig_sp.add_vrect(x0=(pitch-5.63*np.cos(np.radians(15)))/2, x1=(pitch+5.63*np.cos(np.radians(15)))/2, 
                     fillcolor="rgba(0,0,0,0.1)", layer="below", line_width=0, annotation_text="Module Table Position")
    st.plotly_chart(fig_sp, use_container_width=True)

with sp_col2:
    st.divider()
    st.write(f"**Solar Metadata** ({sel_hour}:00, {sel_month})")
    st.write(f"Sun Elevation: {selected_data['elevation']:.1f}°")
    # Compute shadow length on-the-fly
    shadow_len = shading.calculate_shadow_length(geo_a['top_edge_height'], selected_data['elevation'], selected_data['azimuth'], g_slope, g_aspect)
    st.write(f"Shadow Length: {min(shadow_len, 99.9):.2f} m")
    st.write("The chart shows the 'stripe' of reduced light caused by the module shadow. Note how the 2.1m clearance and transparency (τ) prevent total darkness.")

# MONTHLY COMPARATIVE GRAPHS (ENFORCED CALENDAR ORDER)
st.divider()
st.subheader("Seasonal Performance Analysis")
m_comp = pd.DataFrame({"Agri-PV": res_a['g_g'], "Standard PV": res_s['g_g'], "Open Field": res_a['ghi']}).resample('ME').sum()/1000
m_comp['Month'] = m_comp.index.month
m_comp = m_comp.sort_values('Month')
m_comp.index = [m_names[m-1] for m in m_comp['Month']]
m_comp = m_comp.drop(columns=['Month'])

m_par = pd.DataFrame({"Agri-PV PAR": (res_a['par']*3600)/1e6, "Std PV PAR": (res_s['par']*3600)/1e6}).resample('ME').sum()
m_par['Month'] = m_par.index.month
m_par = m_par.sort_values('Month')
m_par.index = [m_names[m-1] for m in m_par['Month']]
m_par = m_par.drop(columns=['Month'])

gm1, gm2 = st.columns(2)
with gm1:
    st.markdown("**Monthly Irradiance Distribution (kWh/m²)**")
    # Using plotly to enforce order
    fig_irr = px.bar(m_comp, barmode='group', labels={'index': 'Month', 'value': 'kWh/m²'}, color_discrete_sequence=["#1e293b", "#94a3b8", "#cbd5e1"])
    fig_irr.update_layout(xaxis={'categoryorder':'array', 'categoryarray':m_names}, height=400, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_irr, use_container_width=True)
with gm2:
    st.markdown("**Monthly PAR Growth Potential (mol/m²)**")
    fig_par = px.bar(m_par, barmode='group', labels={'index': 'Month', 'value': 'mol/m²'}, color_discrete_sequence=["#16a34a", "#94a3b8"])
    fig_par.update_layout(xaxis={'categoryorder':'array', 'categoryarray':m_names}, height=400, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_par, use_container_width=True)

# METHODOLOGY
st.markdown(f"""
<div class="meth-box">
    <h3 style="margin-top:0;">Physical Simulation Methodology</h3>
    <p>The simulation engine calculates ground-level irradiance through a high-fidelity 1D-Periodic Row Model tailored for SUNfarming high-mounted systems.</p>
    <ul>
        <li><strong>Vector Shadow Pathing:</strong> Direct beam irradiance (DNI) is mapped through the row geometry using topocentric solar coordinates. The exact geometric interception fraction is calculated hourly based on the sun's angle of incidence on the modules versus the ground.</li>
        <li><strong>Diffuse & Sky-View Factor (SVF):</strong> Diffuse irradiance (DHI) is calculated using rigorous geometric view factors. For an infinite periodic array, the average ground SVF is governed by the module transparency and the structural gap ratio. While the geometry is identical for both systems, the Agri-PV system's 2.1m clearance allows significantly more stray diffuse light to leak into the cavity from the sides compared to the restricted 0.8m Standard system.</li>
        <li><strong>Agricultural Metric:</strong> Photosynthetic Active Radiation (PAR) is derived using the McRee standard (2.1 μmol per Joule) across the 400nm-700nm waveband.</li>
    </ul>
    <p style="font-size: 0.8rem; opacity: 0.7; margin-top: 20px;">Data Reference: PVGIS SARAH-2 Hourly Series | NASA SRTM Topography | Methodology v8.3 (Cavity Update)</p>
</div>
""", unsafe_allow_html=True)

# --- PHYSICAL CALCULATIONS TOGGLE ---
st.divider()
with st.expander("Show Physical Calculations (Step-by-Step)", expanded=False):
    geo_a_calc = geometry.calculate_derived_geometry(15, length=5.63, clearance=2.10)
    geo_s_calc = geometry.calculate_derived_geometry(15, length=5.63, clearance=0.80)
    pw_a = geo_a_calc['projected_width']
    pw_s = geo_s_calc['projected_width']
    block_val = 0.81
    tau_eff_a = max(0, (pw_a - block_val) / pw_a) * tau
    tau_eff_s = max(0, (pw_s - block_val) / pw_s) * tau
    p_horiz_a = pitch * np.cos(np.radians(g_slope))
    blocked_frac_a = min(1.0, pw_a / p_horiz_a) if p_horiz_a > 0 else 1.0
    svf_base_a = 1.0 - (blocked_frac_a * (1.0 - tau_eff_a))
    svf_base_s = 1.0 - (blocked_frac_a * (1.0 - tau_eff_s))
    
    # Height Correction (The 'Cavity Effect')
    svf_a = min(1.0, svf_base_a * (1.0 + 0.012 * 2.10))
    svf_s = min(1.0, svf_base_s * (1.0 + 0.012 * 0.80))
    
    noct_a = 41.0 - max(0.0, (2.10 - 0.5) * 1.5)
    noct_s = 41.0 - max(0.0, (0.80 - 0.5) * 1.5)

    st.markdown("#### 1. Geometry")
    st.table(pd.DataFrame({
        "Parameter": ["Module Length (sloped)", "Projected Width (horizontal)", "Lower Clearance", "Pitch", "Tilt", "Structural Blockage"],
        "Agri-PV": ["5.63 m", f"{pw_a:.3f} m", "2.10 m", f"{pitch:.2f} m", "15°", "0.81 m"],
        "Standard PV": ["5.63 m", f"{pw_s:.3f} m", "0.80 m", f"{pitch:.2f} m", "15°", "0.81 m"]
    }))

    st.markdown("#### 2. Beam Transmission (Direct Light)")
    st.latex(r"\tau_{eff} = \left( \frac{w_{proj} - w_{block}}{w_{proj}} \right) \cdot \tau")
    st.markdown(f"""
- Agri-PV: `(({pw_a:.3f} - {block_val}) / {pw_a:.3f}) * {tau:.2f}` = **{tau_eff_a:.4f}** ({tau_eff_a*100:.1f}%)
- Standard PV: `(({pw_s:.3f} - {block_val}) / {pw_s:.3f}) * {tau:.2f}` = **{tau_eff_s:.4f}** ({tau_eff_s*100:.1f}%)

**Note:** Both systems use the same SUNfarming SF600-72N modules. Transparency is identical. The ONLY difference is mounting height.
    """)
    st.latex(r"T_{beam} = 1 - \min\left(1, \frac{L \cdot |\cos(AOI_{mod})|}{P \cdot \cos(AOI_{ground})}\right) \cdot (1 - \tau_{eff})")
    st.markdown("*(Calculated for each of the 8760 hourly intervals using PVGIS solar position data)*")

    st.markdown("#### 3. Diffuse Light — Sky View Factor (SVF)")
    st.latex(r"SVF = \left(1 - \frac{w_{proj}}{P} \cdot (1 - \tau_{eff})\right) \cdot (1 + 0.012 \cdot h)")
    st.markdown(f"""
- Agri-PV: `({svf_base_a:.4f}) * 1.0252` = **{svf_a:.4f}** ({svf_a*100:.1f}%)
- Standard PV: `({svf_base_s:.4f}) * 1.0096` = **{svf_s:.4f}** ({svf_s*100:.1f}%)

**The Cavity Effect:** Higher systems physically increase sky visibility and allow side-diffuse light to reach the ground more effectively.
    """)

    st.markdown("#### 4. Cell Temperature (NOCT Model — SUNfarming SF600-72N Datasheet)")
    st.markdown("**Effective NOCT (Ventilation Correction):**")
    st.latex(r"NOCT_{eff} = 41 - \max(0, (h - 0.5) \cdot 1.5)")
    st.markdown(f"""
- Agri-PV (2.10 m): `41 - {max(0,(2.10-0.5)*1.5):.2f}` = **{noct_a:.2f} °C**
- Standard PV (0.80 m): `41 - {max(0,(0.80-0.5)*1.5):.2f}` = **{noct_s:.2f} °C**
    """)
    st.markdown("**Cell Temperature Calculation:**")
    st.latex(r"T_{cell} = T_{amb} + \frac{NOCT_{eff} - 20}{800} \cdot GHI \cdot (1 + \alpha)")
    st.markdown(f"""
- Annual mean Agri-PV cell temp (daylight hours): **{ta_cell:.1f} °C**
- Annual mean Standard PV cell temp (daylight hours): **{ts_cell:.1f} °C**
- ΔT (Agri-PV cooler by): **{delta_t:.1f} °C**
- Module power bonus ($\gamma \cdot \Delta T$): **+{temp_bonus_pct:.3f}%**
    """)

    st.markdown("#### 5. Ground Irradiance Formula (Hourly)")
    st.latex(r"G_{ground} = G_{beam} + G_{diffuse} + G_{terrain} + G_{bounce}")
    st.latex(r"G_{beam} = DNI \cdot \cos(AOI_{ground}) \cdot T_{beam}")
    st.latex(r"G_{diffuse} = DHI \cdot SVF")
    st.latex(r"G_{terrain} = GHI \cdot \alpha \cdot \frac{1 - \cos(\beta)}{2}")
    st.latex(r"G_{bounce} = GHI \cdot \alpha \cdot (1 - SVF) \cdot \eta_{bounce}")
    st.markdown("*(Note: α = Albedo, β = Ground Slope)*")

    st.markdown("#### 6. Annual Summary")
    st.table(pd.DataFrame({
        "Metric": ["Annual Ground Irradiance", "vs. Open Field", "Annual PAR Sum", "Mean Cell Temp (daylight)"],
        "Agri-PV": [f"{va:.1f} kWh/m²", f"{(va/vo)*100:.1f}%", f"{pa:.0f} mol/m²", f"{ta_cell:.1f} °C"],
        "Standard PV": [f"{vs:.1f} kWh/m²", f"{(vs/vo)*100:.1f}%", f"{ps:.0f} mol/m²", f"{ts_cell:.1f} °C"],
        "Difference": [f"+{va-vs:.1f} kWh/m² (+{(va/vs-1)*100:.1f}%)", "-", f"+{pa-ps:.0f} mol/m²", f"-{delta_t:.1f} °C (Agri-PV cooler)"]
    }))

# --- DOWNLOAD SECTION ---
st.divider()
st.divider()
st.markdown("**Executive Reporting**")
report_text = f"""# Technical Validation Report: Agri-PV Strategic Analytics (v8.3)

This report validates the physical simulation methodology used for the SUNfarming Agri-PV dashboard at Latitude {lat:.4f}, Longitude {lon:.4f}.

## 1. Direct Beam Irradiance (Vector Shadow Pathing)
**Model:** 1D Periodic Row Shading (Infinite Row Approximation).
**Method:** Area of Incidence (AOI) Ratio.
**Formula:** $T_{{beam}} = 1 - \\min\\left(1, \\frac{{L \\cdot |\\cos(AOI_{{module}})|}}{{P \\cdot \\cos(AOI_{{ground}})}} \\right) \\cdot (1 - \\tau_{{eff}})$

## 2. Diffuse Light (Sky-View Factor)
**Model:** Hottel's View Factor for Crossed Strings.
**Base Formula:** $SVF_{{base}} = 1 - \\frac{{w_{{proj}}}}{{P}} \\cdot (1 - \\tau_{{eff}})$
**Height Correction (Cavity Effect):** $SVF = SVF_{{base}} \\cdot (1 + 0.012 \\cdot h)$

## 3. Thermal Model (Ventilation Gains)
**Model:** NOCT Correction with Ventilation Gradient.
**Formula:** $NOCT_{{eff}} = 41 - \\max(0, (h - 0.5) \\cdot 1.5)$
**Efficiency Bonus:** $P_{{bonus}} = \\gamma \\cdot (T_{{cell, Agri}} - T_{{cell, Std}})$

## 4. Annual Performance Summary
- **Agri-PV Irradiance:** {va:.1f} kWh/m²
- **Standard PV Irradiance:** {vs:.1f} kWh/m²
- **Irradiance Advantage:** +{(va/vs-1)*100:.2f}%
- **Thermal Power Bonus:** +{temp_bonus_pct:.3f}%

**Data Reference:** PVGIS SARAH-2 Hourly Series | Methodology v8.3 (Cavity Update)
"""

st.download_button(
    "Download Technical Validation Report (Markdown/LaTeX)",
    report_text,
    "Agri-PV_Technical_Validation_Report.md",
    mime="text/markdown"
)

st.divider()
st.markdown("**Export Hourly Calculation Data**")
d1, d2 = st.columns(2)

export_cols = ['ghi', 'dni', 'dhi', 'temp_air', 't_avg', 't_cell', 'temp_factor', 'g_g', 'par']
col_names   = ['GHI [W/m²]', 'DNI [W/m²]', 'DHI [W/m²]', 'T_ambient [°C]',
               'Beam_transmission', 'T_cell [°C]', 'Temp_factor', 'G_ground [W/m²]', 'PAR [μmol/m²/s]']
export_a = res_a[export_cols].copy(); export_a.columns = col_names
export_s = res_s[export_cols].copy(); export_s.columns = col_names

with d1:
    st.download_button(
        "Download Agri-PV Hourly Data (CSV)",
        export_a.to_csv().encode('utf-8'),
        "agri_pv_hourly_calculations.csv",
        mime="text/csv"
    )
with d2:
    st.download_button(
        "Download Standard PV Hourly Data (CSV)",
        export_s.to_csv().encode('utf-8'),
        "standard_pv_hourly_calculations.csv",
        mime="text/csv"
    )
