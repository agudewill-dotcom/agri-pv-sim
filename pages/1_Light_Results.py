import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import geometry
import shading
import irradiance
import simulation

st.set_page_config(
    page_title="Agri-PV Light Results",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #fafafa;
    }
    h1, h2, h3, .title-outfit {
        font-family: 'Outfit', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 6px solid #60a5fa;
    }
    .meth-box {
        background: #f8fafc;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Retrieve simulation data from session_state
if 'metrics' not in st.session_state:
    st.info("🔄 Running initial simulation... Please stand by.")
    config, res_a, res_s, metrics = simulation.render_sidebar_and_run()
    st.session_state['config'] = config
    st.session_state['res_a'] = res_a
    st.session_state['res_s'] = res_s
    st.session_state['metrics'] = metrics
else:
    config = st.session_state['config']
    res_a = st.session_state['res_a']
    res_s = st.session_state['res_s']
    metrics = st.session_state['metrics']

st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-weight:800;">🌾 High-Fidelity Light Results</h2>
    <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem;">Spatial Shadow Paths, 2D Irradiance Distributions, and Analytical View Factors</p>
</div>
""", unsafe_allow_html=True)

m_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# SECTION 1: SENSORS & HEATMAPS
c_meta, c_heat = st.columns([1, 1.5])

with c_meta:
    st.subheader("📊 Comparative Sensor Profile")
    st.markdown("Annual cumulative ground irradiance and PAR potential under the system:")
    st.table(pd.DataFrame([
        {"System": "Agri-PV (2.10m)", "Irradiance": f"{metrics['va']:.0f} kWh/m²", "PAR": f"{metrics['pa']:.0f} mol/m²"},
        {"System": "Standard (0.80m)", "Irradiance": f"{metrics['vs']:.0f} kWh/m²", "PAR": f"{metrics['ps']:.0f} mol/m²"},
        {"System": "Open Field", "Irradiance": f"{metrics['vo']:.0f} kWh/m²", "PAR": f"{metrics['par_open_field']:.0f} mol/m²"}
    ]))
    st.info(f"**Structural Blockage Factor:** 0.81m row blockage (SUNfarming rack geometry). Pitch: {config['pitch']:.2f}m. Modules: SF600-72N.")

with c_heat:
    st.subheader("🗺️ Light Intensity Heatmap (W/m² - Agri-PV)")
    # Generate diurnal/seasonal average irradiance matrix
    h_data = res_a.groupby([res_a.index.month, res_a.index.hour])['g_g'].mean().unstack()
    h_data.index = m_names
    
    fig_heat = px.imshow(
        h_data,
        color_continuous_scale='Viridis',
        labels=dict(x="Hour of Day", y="Month", color="W/m²"),
        aspect='auto',
        height=320
    )
    fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# SECTION 2: SPATIAL VECTOR SHADOW PROFILES
st.subheader("📐 Spatial Shadow Profile (Cross-Section)")
sp_col1, sp_col2 = st.columns([2, 1])

with sp_col2:
    st.markdown("**Vector Shadow Pathing Controller**")
    sel_month = st.selectbox("Select Target Month", options=m_names, index=5)
    sel_hour = st.slider("Select Hour of Day (Local)", 0, 23, 12)
    month_idx = m_names.index(sel_month) + 1
    
    # Filter solar position for selected period
    time_data = res_a[(res_a.index.month == month_idx) & (res_a.index.hour == sel_hour)]
    if time_data.empty:
        selected_data = res_a[res_a.index.month == month_idx].iloc[0]
    else:
        selected_data = time_data.iloc[0]

with sp_col1:
    x_points = np.linspace(0, config['pitch'], 100)
    geo_a = geometry.calculate_derived_geometry(15, length=5.63, clearance=2.10)
    
    t_mask = shading.calculate_spatial_mask(
        x_points, geo_a['top_edge_height'], 2.10, 5.63, 15, 
        selected_data['elevation'], selected_data['azimuth'], config['pitch'], config['tau']
    )
    
    # Direct beam calculation
    aoi_sel = irradiance.calculate_incidence_angle(selected_data['zenith'], selected_data['azimuth'], config['g_slope'], config['g_aspect'])
    cos_g = np.cos(np.radians(aoi_sel))
    
    # Uniform diffuse and reflected components
    g_base_diff = selected_data['dhi'] * irradiance.sky_view_factor_periodic(
        geo_a['top_edge_height'], geo_a['projected_width'], config['pitch'], 
        max(0, (geo_a['projected_width'] - 0.81)/geo_a['projected_width']) * config['tau'], h_clearance=2.10
    ) * (1.0 + np.cos(np.radians(config['g_slope']))) / 2.0
    
    g_base_refl = selected_data['ghi'] * config['albedo'] * (1.0 - np.cos(np.radians(config['g_slope']))) / 2.0
    
    # Spatial irradiance distribution
    g_spatial = (selected_data['dni'] * np.maximum(0.0, cos_g) * t_mask) + g_base_diff + g_base_refl
    
    fig_sp = px.line(
        x=x_points, y=g_spatial,
        labels={'x': 'Horizontal Distance across Pitch (m)', 'y': 'Irradiance (W/m²)'},
        title=f"Instantaneous Light Distribution ({sel_month}, {sel_hour}:00)"
    )
    
    # Add visual bounds for the module
    m_start = (config['pitch'] - geo_a['projected_width']) / 2
    m_end = m_start + geo_a['projected_width']
    fig_sp.add_vrect(
        x0=m_start, x1=m_end, 
        fillcolor="rgba(0,0,0,0.1)", layer="below", line_width=0, 
        annotation_text="Module Table Position"
    )
    fig_sp.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_sp, use_container_width=True)

with sp_col2:
    st.divider()
    st.markdown(f"**Solar Metadata ({sel_hour}:00 {sel_month})**")
    st.write(f"☀️ Sun Elevation: **{selected_data['elevation']:.1f}°**")
    sh_len = shading.calculate_shadow_length(geo_a['top_edge_height'], selected_data['elevation'], selected_data['azimuth'], config['g_slope'], config['g_aspect'])
    st.write(f"📏 Shadow Length: **{min(sh_len, 99.9):.2f} m**")
    st.write("The cross-section visualizes the module's cast shadow stripe. Transparency (τ) and high elevated spacing prevent total ground darkness.")

# SECTION 3: EXPANDABLE CALCULATIONS
st.divider()
with st.expander("📝 Show Step-by-Step Physics Calculations"):
    geo_a_calc = geometry.calculate_derived_geometry(15, length=5.63, clearance=2.10)
    geo_s_calc = geometry.calculate_derived_geometry(15, length=5.63, clearance=0.80)
    pw_a = geo_a_calc['projected_width']
    pw_s = geo_s_calc['projected_width']
    h_top_a = geo_a_calc['top_edge_height']
    h_top_s = geo_s_calc['top_edge_height']
    block_val = 0.81
    tau_eff_a = max(0, (pw_a - block_val) / pw_a) * config['tau']
    tau_eff_s = max(0, (pw_s - block_val) / pw_s) * config['tau']
    
    svf_a = irradiance.sky_view_factor_periodic(h_top_a, pw_a, config['pitch'], tau_eff_a, h_clearance=2.10)
    svf_s = irradiance.sky_view_factor_periodic(h_top_s, pw_s, config['pitch'], tau_eff_s, h_clearance=0.80)
    
    st.markdown("#### 1. System Geometry")
    st.table(pd.DataFrame({
        "Parameter": ["Module Sloped Length", "Projected Horizontal Width", "Lower Mounting Clearance", "Top Edge Height", "Row Pitch", "Blockage Width"],
        "Agri-PV (2.10m)": ["5.63 m", f"{pw_a:.3f} m", "2.10 m", f"{h_top_a:.3f} m", f"{config['pitch']:.2f} m", "0.81 m"],
        "Standard PV (0.80m)": ["5.63 m", f"{pw_s:.3f} m", "0.80 m", f"{h_top_s:.3f} m", f"{config['pitch']:.2f} m", "0.81 m"]
    }))

    st.markdown("#### 2. Beam Transmission (Direct Light Interception)")
    st.latex(r"\tau_{eff} = \left( \frac{w_{proj} - w_{block}}{w_{proj}} \right) \cdot \tau")
    st.markdown(f"""
- **Agri-PV (2.10m):** `(({pw_a:.3f} - {block_val}) / {pw_a:.3f}) * {config['tau']:.2f}` = **{tau_eff_a:.4f}** ({tau_eff_a*100:.1f}%)
- **Standard PV (0.80m):** `(({pw_s:.3f} - {block_val}) / {pw_s:.3f}) * {config['tau']:.2f}` = **{tau_eff_s:.4f}** ({tau_eff_s*100:.1f}%)
    """)
    st.latex(r"T_{beam} = 1 - \min\left(1, \frac{L \cdot \max(0, \cos(AOI_{mod}))}{P \cdot \cos(AOI_{ground})}\right) \cdot (1 - \tau_{eff})")

    st.markdown("#### 3. Diffuse Light — Analytical Periodic Sky View Factor")
    st.latex(r"SVF = \frac{1}{P} \int_0^P \left[ 1 - \frac{\arctan(H/x) + \arctan(H/(P-x))}{\pi} \cdot (1 - \tau_{eff}) \right] dx")
    st.markdown(f"""
- **Agri-PV (H={h_top_a:.2f}m):** SVF = **{svf_a:.4f}** ({svf_a*100:.1f}%)
- **Standard PV (H={h_top_s:.2f}m):** SVF = **{svf_s:.4f}** ({svf_s*100:.1f}%)
- **Scientific Significance:** High clearance height allows stray diffuse light to enter from adjacent row gaps. No arbitrary constants — height dependency emerges directly from Hottel integration.
    """)

    st.markdown("#### 4. Total Ground Irradiance Formulation")
    st.latex(r"G_{ground} = G_{beam} + G_{diffuse} + G_{reflected}")
    st.latex(r"G_{beam} = DNI \cdot \cos(AOI_{ground}) \cdot T_{beam}")
    st.latex(r"G_{diffuse} = DHI \cdot SVF \cdot \frac{1 + \cos(\beta)}{2}")
    st.latex(r"G_{reflected} = GHI \cdot \alpha \cdot \frac{1 - \cos(\beta)}{2} + GHI \cdot \alpha \cdot (1 - SVF) \cdot \rho_{back}")
    st.markdown("*(α = ground albedo, β = site slope, ρ_back = 0.15 backsheet reflectance)*")
