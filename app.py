import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
from datetime import datetime
import requests

import importlib
import solar
import geometry
import shading
import irradiance
import spatial
import thermal

importlib.reload(geometry)
importlib.reload(shading)
importlib.reload(irradiance)
importlib.reload(spatial)
importlib.reload(thermal)

import simulation
importlib.reload(simulation)

from geometry import TableGeometry, GEOMETRY_PRESETS
from crop_profiles import CROP_REGISTRY
from crop_scoring import evaluate_all_crops, evaluate_crop
from medicinal_crop_suitability import evaluate_all_medicinal_crops, MED_CROP_REGISTRY, MED_SOURCES_REGISTRY
from meadow_suitability import evaluate_all_meadow_species, MEADOW_REGISTRY
from crop_suitability import SOURCES_REGISTRY

st.set_page_config(page_title="Agri-PV Strategic Analytics", layout="wide")

# --- EXECUTIVE DESIGN SYSTEM (CSS) ---
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
    
    /* Premium Metric Panel Style */
    .stMetric {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        border-top: 5px solid #0f172a;
        min-height: 160px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    }
    
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        line-height: 1.2;
    }
    
    /* Nuclear fix for metric truncation in all responsive sizes */
    div[data-testid="stMetric"] * {
        white-space: normal !important;
        text-overflow: unset !important;
        overflow: visible !important;
        word-break: break-word !important;
    }
    
    /* Hide default +/- buttons from Streamlit number inputs */
    button[data-testid="stNumberInputStepUp"], 
    button[data-testid="stNumberInputStepDown"] {
        display: none !important;
    }
    div[data-testid="stNumberInputContainer"] {
        padding-right: 10px !important;
    }
    
    /* Section containers & boxes */
    .status-box {
        background: #1e293b;
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 10px solid #22c55e;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .status-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #22c55e;
        margin-bottom: 5px;
    }
    
    .meth-box {
        background: #f8fafc;
        padding: 30px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-top: 40px;
        color: #1e293b;
    }
    
    .info-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
    }
    
    .yield-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .din-box {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .law-box {
        background-color: #fff1f2;
        border-left: 6px solid #f43f5e;
        border-radius: 8px;
        padding: 20px 24px;
        color: #9f1239;
        margin-top: 30px;
    }
    
    /* Header gradients per tab */
    .header-overview {
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 6px solid #94a3b8;
    }
    
    .header-light {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 6px solid #60a5fa;
    }
    
    .header-crops {
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 6px solid #a7f3d0;
    }
    
    .header-elec {
        background: linear-gradient(135deg, #7c2d12 0%, #ea580c 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 6px solid #ffedd5;
    }
    
    .header-din {
        background: linear-gradient(135deg, #334155 0%, #64748b 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        border-left: 6px solid #cbd5e1;
    }
    
    /* Crop card layouts */
    .crop-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border-left: 6px solid #cbd5e1;
    }
    
    .crop-card-sehr-gut { border-left-color: #059669; }
    .crop-card-geeignet { border-left-color: #10b981; }
    .crop-card-grenzwertig { border-left-color: #f59e0b; }
    .crop-card-nicht { border-left-color: #ef4444; }
    
    .badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    
    .badge-sehr-gut { background-color: #d1fae5; color: #065f46; }
    .badge-geeignet { background-color: #d1fae5; color: #047857; }
    .badge-grenzwertig { background-color: #fef3c7; color: #92400e; }
    .badge-nicht { background-color: #fee2e2; color: #991b1b; }
    
    .limiting-box {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        margin-top: 10px;
        border-left: 3px solid #64748b;
    }
    
    /* --- TAB CONTROLLER PRESET OVERRIDES --- */
    button[data-testid="stTab"] {
        color: #475569 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        border: none !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    button[data-testid="stTab"]:hover {
        color: #1e3a8a !important;
        background-color: rgba(30, 58, 138, 0.05) !important;
        border-radius: 8px !important;
    }
    
    button[data-testid="stTab"][aria-selected="true"] {
        color: #1e3a8a !important;
        font-weight: 800 !important;
        border-bottom: 3px solid #1e3a8a !important;
        background-color: rgba(30, 58, 138, 0.08) !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    button[data-testid="stTab"] p {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        margin: 0 !important;
    }
    
    div[role="tablist"] {
        border-bottom: 2px solid #e2e8f0 !important;
        padding-bottom: 5px !important;
        margin-bottom: 25px !important;
        gap: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- SIMULATION ENGINE RUN ---
config, res_a, res_s, metrics = simulation.render_sidebar_and_run()
st.session_state['config'] = config
st.session_state['res_a'] = res_a
st.session_state['res_s'] = res_s
st.session_state['metrics'] = metrics

# Compute crop suitability results (Reproductive)
crop_results = evaluate_all_crops(
    par_ann=metrics['pa'],
    par_ref=metrics['par_open_field'],
    monthly_par=metrics['monthly_par_agri'],
    cv_par=metrics['cv_par'],
    has_hourly=True
)
st.session_state['crop_results'] = crop_results

# Compute crop suitability results (Biomass)
import biomass_suitability
site_context = biomass_suitability.SiteContext(hot_dry_index=0.5, water_stress_risk=0.5, humidity_disease_index=0.2)
crop_results_bio = biomass_suitability.evaluate_all_biomass(metrics, res_a, site_context)
st.session_state['crop_results_bio'] = crop_results_bio

# Compute medicinal crop suitability results
crop_results_med = evaluate_all_medicinal_crops(
    annual_PAR_agri=metrics['pa'],
    annual_PAR_openfield=metrics['par_open_field'],
    monthly_PAR_agri=metrics['monthly_par_agri'],
    monthly_PAR_openfield=metrics['monthly_par_open'],
    cv_PAR=metrics['cv_par'],
    hourly_par=res_a['par']
)
st.session_state['crop_results_med'] = crop_results_med

# Compute meadow crop suitability results
crop_results_meadow = evaluate_all_meadow_species(
    annual_PAR_agri=metrics['pa'],
    annual_PAR_openfield=metrics['par_open_field'],
    monthly_PAR_agri=metrics['monthly_par_agri'],
    monthly_PAR_openfield=metrics['monthly_par_open'],
    cv_PAR=metrics['cv_par']
)
st.session_state['crop_results_meadow'] = crop_results_meadow

ENGLISH_CROP_NOTES = {
    "luzerne": "Lucerne is relatively shade-tolerant and well-suited for Agri-PV systems with moderate shading.",
    "wintergerste": "Winter barley shows stable yields in field trials under Agri-PV at ≥ 60% PAR availability.",
    "winterroggen": "Winter rye is robust and relatively shade-tolerant; ear development in spring is sensitive to light limitation.",
    "triticale": "Triticale (wheat-rye hybrid) shows similar shade tolerance to winter rye.",
    "winterweizen": "Winter wheat is the best-studied crop under Agri-PV. Yield losses are likely at < 65% PAR.",
    "dinkel": "Spelt is evaluated as a proxy of the winter wheat group. Direct Agrivoltaic trial data is missing.",
    "einkorn": "Einkorn is evaluated as a proxy of the winter wheat group. Direct Agrivoltaic trial data is missing.",
    "emmer": "Emmer is evaluated as a proxy of the winter wheat group. Direct Agrivoltaic trial data is missing.",
    "hafer": "Oats are evaluated as a proxy of the winter wheat group. Direct Agrivoltaic trial data is missing.",
    "schwarzhafer": "Black oat is highly light-demanding. Strong yield losses are likely under standard shading layouts.",
    "mais": "Maize is a C4 plant with extremely high light requirements. Not recommended for shaded layouts under panels."
}

def translate_class(val):
    mapping = {
        "sehr gut geeignet": "Highly Suitable",
        "geeignet": "Suitable",
        "grenzwertig": "Marginal",
        "nicht empfohlen": "Not Recommended",
        "geeignet als hauptkultur": "Suitable as Main Crop",
        "geeignet als sonderkultur / blühstreifen": "Suitable as Special Crop / Flower Strip",
        "nur als sonderkultur mit abnehmer-/feldnachweis": "Special Crop Only (Trial Req.)",
        "nur mit agronomischer prüfung": "Only with Agronomic Trial"
    }
    return mapping.get(val.lower(), val)

def translate_limiting(val):
    if not val: return "None"
    mapping = {
        "jahres-par nicht ausreichend": "Insufficient Annual PAR",
        "par in der kritischen phase zu niedrig": "Insufficient PAR in Critical Phase",
        "zu heterogene lichtverteilung": "Heterogeneous Light Distribution",
        "lichtverteilung über die wachstumsmonate ungünstig": "Unfavorable Seasonal Light Distribution",
        "keine": "None"
    }
    return mapping.get(val.lower(), val)

def translate_confidence(val):
    mapping = {
        "hoch": "High",
        "mittel": "Medium",
        "niedrig": "Low"
    }
    return mapping.get(val.lower(), val)

# Define metrics variables for easy reference
va, vs, vo = metrics['va'], metrics['vs'], metrics['vo']
pa, ps = metrics['pa'], metrics['ps']
ya_spec, ys_spec, y_bonus = metrics['ya_spec'], metrics['ys_spec'], metrics['y_bonus']
ta_cell, ts_cell, delta_t, temp_bonus_pct = metrics['ta_cell'], metrics['ts_cell'], metrics['delta_t'], metrics['temp_bonus_pct']


# --- NAVIGATION TABS ---
tab_overview, tab_light, tab_spatial, tab_crops, tab_elec, tab_report = st.tabs([
    "Executive Summary", 
    "Light Results", 
    "Spatial Heatmaps",
    "Crop & Vegetation Compatibility",
    "Electrical & Thermal", 
    "Report & Downloads"
])


# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY
# ==============================================================================
with tab_overview:
    st.markdown("""
    <div class="header-overview">
        <h2 style="margin:0; font-weight:800; color:white;">Strategic Overview Dashboard</h2>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; color:white;">Physics-based technical validation comparing elevated mounting heights (2.10m) against standard configurations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="status-box">
        <div class="status-title">AGRICULTURAL LIGHT ADVANTAGE: +{(va-vs):.0f} kWh/m² vs Standard PV</div>
        <div style="font-size: 2.5rem; font-weight: 800;">+{(va/vs-1)*100:.1f}% Ground Irradiance Bonus | +{temp_bonus_pct:.2f}% Module Efficiency Bonus</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Baseline & Agri-PV light metrics row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Agricultural Light", f"{(va/vo)*100:.1f}%", f"+{(va/vs-1)*100:.1f}% vs Std. PV", help="Percentage of total open-field irradiance reaching the ground under the system.")
    k2.metric("Annual PAR Sum", f"{pa:.0f} mol/m² ({(metrics['pa']/metrics['par_open_field'])*100:.1f}%)", f"+{(pa/ps-1)*100:.1f}% vs Std. PV", help="Annual cumulative Photosynthetic Active Radiation (PAR) for crop growth.")
    k3.metric("BASELINE: STANDARD GROUND-PV", f"{vs:.0f} kWh/m²", f"RESTRICTED: {(vs/vo)*100:.1f}% LIGHT", help="Annual ground irradiance for a standard 0.8m high system.")
    k4.metric("VS. STANDARD GROUND-PV", f"+{va-vs:.0f} kWh/m²", help="The absolute irradiance advantage of Agri-PV over Standard PV.")
    
    # Temperature statistics row
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Agri-PV Cell Temp", f"{ta_cell:.1f} °C", f"−{delta_t:.1f}°C vs Standard", help="Annual arithmetic mean during daylight hours (GHI > 50 W/m²)")
    t2.metric("Std. PV Cell Temp", f"{ts_cell:.1f} °C", "Restricted ventilation at 0.8m", help="Annual arithmetic mean during daylight hours (GHI > 50 W/m²)")
    t3.metric("Temp. Power Bonus", f"+{temp_bonus_pct:.2f}%", "Agri-PV cooler → higher η", help="Relative module power increase due to the lower cell temperatures in high-mounted systems.")
    t4.metric("Thermal Model", "Faiman (2008)", f"Wind-corrected, log profile", help="Cell temperature via Faiman model with logarithmic wind profile height correction using PVGIS 10m wind data.")
    
    # Specific yield calculation block
    st.markdown(f"""
    <div class="yield-box">
        <div>
            <h4 style="margin:0; color:#475569; font-family:'Outfit';">SPECIFIC YIELD BONUS (ELECTRICAL)</h4>
            <p style="margin:4px 0 0 0; font-size:0.9rem; color:#64748b;">Annual energy generation advantage per installed kWp</p>
        </div>
        <div style="text-align:right;">
            <span style="font-size:2rem; font-weight:800; color:#1e293b;">+{y_bonus:.1f} kWh/kWp <span style="font-size:1.2rem; color:#16a34a; font-weight:600;">(+{temp_bonus_pct:.2f}%)</span></span>
            <div style="font-size:0.9rem; font-weight:600; color:#16a34a;">↑ Agri-PV: {ya_spec:.0f} vs Standard: {ys_spec:.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Highlight table
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
        <p style="margin-top:14px; font-size:0.85rem; color:#475569;">Both systems: 5.63m table | {config['pitch']:.2f}m pitch | 15° tilt | {config['tau']*100:.0f}% transparency. Only the mounting height varies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Physical Simulation Methodology summary
    st.markdown(f"""
    <div class="meth-box">
        <h3 style="margin-top:0;">Physical Simulation Methodology (v9.0)</h3>
        <p><strong>Model Classification:</strong> Physics-based comparative simulation for early-stage Agri-PV system evaluation. This tool estimates <em>relative</em> performance differences between elevated and standard-height PV configurations. It is not intended as a bankable yield model.</p>
        <ul>
            <li><strong>Direct Beam (AOI-Ratio):</strong> Hourly beam transmission computed via geometric interception fraction using topocentric solar coordinates. The ratio of module-projected to ground-projected areas determines the shaded fraction per pitch period.</li>
            <li><strong>Diffuse Sky (Analytical SVF):</strong> Pitch-averaged sky view factor computed by integrating the elevation angles subtended by adjacent rows across one pitch period (Hottel crossed-strings method). No empirical height corrections — the geometry produces height dependence naturally via arctan(H/d).</li>
            <li><strong>Ground Reflection:</strong> Isotropic terrain albedo (Liu & Jordan) plus first-order cavity inter-reflection bounded by module backsheet reflectance (ρ_back = 0.15).</li>
            <li><strong>Thermal Model:</strong> Faiman (2008) cell temperature model with height-dependent wind speed via logarithmic atmospheric boundary layer profile. Uses PVGIS WS10m wind data. No arbitrary NOCT corrections.</li>
            <li><strong>PAR:</strong> McCree (1972) conversion: G × f_PAR × 4.57 µmol/J, where f_PAR = 0.45 is the broadband-to-PAR spectral fraction.</li>
        </ul>
        <p style="font-size:0.85rem; margin-top:15px;"><strong>Limitations:</strong> 2D cross-section (infinite row assumption), isotropic diffuse sky (no circumsolar), steady-state thermal model, constant PAR spectral fraction.</p>
        <p style="font-size: 0.8rem; opacity: 0.7; margin-top: 10px;">Data: PVGIS SARAH-2 Hourly Series | NASA SRTM Topography | Methodology v9.0</p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: LIGHT RESULTS
# ==============================================================================
with tab_light:
    st.markdown("""
    <div class="header-light">
        <h2 style="margin:0; font-weight:800; color:white;">High-Fidelity Light Results</h2>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; color:white;">Spatial Shadow Paths, 2D Irradiance Distributions, and Analytical View Factors</p>
    </div>
    """, unsafe_allow_html=True)
    
    m_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Comparative sensor profiles & heatmap columns
    c_meta, c_heat = st.columns([1, 1.5])
    
    with c_meta:
        st.subheader("Comparative Sensor Profile")
        st.markdown("Annual cumulative ground irradiance and PAR potential under the system:")
        st.table(pd.DataFrame([
            {"System": "Agri-PV (2.10m)", "Irradiance": f"{va:.0f} kWh/m²", "PAR": f"{pa:.0f} mol/m²"},
            {"System": "Standard (0.80m)", "Irradiance": f"{vs:.0f} kWh/m²", "PAR": f"{ps:.0f} mol/m²"},
            {"System": "Open Field", "Irradiance": f"{vo:.0f} kWh/m²", "PAR": f"{metrics['par_open_field']:.0f} mol/m²"}
        ]))
        st.info(f"**Structural Blockage Factor:** 0.81m row blockage (SUNfarming rack geometry). Pitch: {config['pitch']:.2f}m. Modules: SF600-72N.")
        
    with c_heat:
        st.subheader("Light Intensity Heatmap (W/m² - Agri-PV)")
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
    
    # Vector Shadow Paths
    st.subheader("Spatial Shadow Profile (Cross-Section)")
    sp_col1, sp_col2 = st.columns([2, 1])
    
    with sp_col2:
        st.markdown("**Vector Shadow Pathing Controller**")
        sel_month = st.selectbox("Select Target Month", options=m_names, index=5, key="light_month")
        sel_hour = st.slider("Select Hour of Day (Local)", 0, 23, 12, key="light_hour")
        month_idx = m_names.index(sel_month) + 1
        
        time_data = res_a[(res_a.index.month == month_idx) & (res_a.index.hour == sel_hour)]
        if time_data.empty:
            selected_data = res_a[res_a.index.month == month_idx].iloc[0]
        else:
            selected_data = time_data.iloc[0]
            
    with sp_col1:
        geo_inst = TableGeometry.from_dict(config.get('geometry', {}))
        x_points = np.linspace(0, geo_inst.row_pitch_m, 100)
        
        t_mask = shading.calculate_spatial_mask(
            x_points, geo_inst, selected_data['elevation'], selected_data['azimuth'], config['tau']
        )
        
        aoi_sel = irradiance.calculate_incidence_angle(selected_data['zenith'], selected_data['azimuth'], config['g_slope'], config['g_aspect'])
        cos_g = np.cos(np.radians(aoi_sel))
        
        pw_val = geo_inst.table_projected_width_m
        tau_eff_val = max(0, (pw_val - 0.81)/pw_val) * config['tau'] if pw_val > 0 else 0
        g_base_diff = selected_data['dhi'] * irradiance.sky_view_factor_periodic(
            geo_inst.h_high_m, pw_val, geo_inst.row_pitch_m, 
            tau_eff_val, h_clearance=geo_inst.clear_height_m
        ) * (1.0 + np.cos(np.radians(config['g_slope']))) / 2.0
        
        g_base_refl = selected_data['ghi'] * config['albedo'] * (1.0 - np.cos(np.radians(config['g_slope']))) / 2.0
        
        g_spatial = (selected_data['dni'] * np.maximum(0.0, cos_g) * t_mask) + g_base_diff + g_base_refl
        
        fig_sp = px.line(
            x=x_points, y=g_spatial,
            labels={'x': 'Horizontal Distance across Pitch (m)', 'y': 'Irradiance (W/m²)'},
            title=f"Instantaneous Light Distribution ({sel_month}, {sel_hour}:00)"
        )
        
        m_start = (geo_inst.row_pitch_m - geo_inst.table_projected_width_m) / 2
        m_end = m_start + geo_inst.table_projected_width_m
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
        st.write(f"Sun Elevation: **{selected_data['elevation']:.1f}°**")
        sh_len = shading.calculate_shadow_length(geo_inst.h_high_m, selected_data['elevation'], selected_data['azimuth'], config['g_slope'], config['g_aspect'])
        st.write(f"Shadow Length: **{min(sh_len, 99.9):.2f} m**")
        st.write("The cross-section visualizes the module's cast shadow stripe. Transparency (τ) and high elevated spacing prevent total ground darkness.")
        
    # Expandable formulas
    st.divider()
    with st.expander("Show Step-by-Step Physics Calculations"):
        geo_a_calc = geo_inst
        geo_s_dict = geo_inst.to_dict()
        geo_s_dict["clear_height_m"] = 0.80
        geo_s_calc = TableGeometry.from_dict(geo_s_dict)
        
        pw_a = geo_a_calc.table_projected_width_m
        pw_s = geo_s_calc.table_projected_width_m
        h_top_a = geo_a_calc.h_high_m
        h_top_s = geo_s_calc.h_high_m
        block_val = 0.81
        tau_eff_a = max(0, (pw_a - block_val) / pw_a) * config['tau'] if pw_a > 0 else 0
        tau_eff_s = max(0, (pw_s - block_val) / pw_s) * config['tau'] if pw_s > 0 else 0
        
        svf_a = irradiance.sky_view_factor_periodic(h_top_a, pw_a, geo_a_calc.row_pitch_m, tau_eff_a, h_clearance=geo_a_calc.clear_height_m)
        svf_s = irradiance.sky_view_factor_periodic(h_top_s, pw_s, geo_s_calc.row_pitch_m, tau_eff_s, h_clearance=0.80)
        
        st.markdown("#### 1. System Geometry")
        st.table(pd.DataFrame({
            "Parameter": ["Module Sloped Length", "Projected Horizontal Width", "Lower Mounting Clearance", "Top Edge Height", "Row Pitch", "Blockage Width"],
            "Agri-PV": [f"{geo_a_calc.table_length_m:.2f} m", f"{pw_a:.3f} m", f"{geo_a_calc.clear_height_m:.2f} m", f"{h_top_a:.3f} m", f"{geo_a_calc.row_pitch_m:.2f} m", "0.81 m"],
            "Standard PV (0.80m)": [f"{geo_s_calc.table_length_m:.2f} m", f"{pw_s:.3f} m", "0.80 m", f"{h_top_s:.3f} m", f"{geo_s_calc.row_pitch_m:.2f} m", "0.81 m"]
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


# ==============================================================================

# ==============================================================================
# TAB 2.5: SPATIAL HEATMAPS
# ==============================================================================
with tab_spatial:
    st.markdown('''
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid #38bdf8;">
        <h2 style="margin:0; font-weight:800; color:white;">Spatial Light and PAR Heatmaps</h2>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; color:white;">2D Spatio-Temporal Shading Matrix across 8760 hours.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        res_sel = st.selectbox("Grid Resolution (m)", [1.0, 0.5, 0.25], index=1)
        st.info("Higher resolution = slower rendering.")
        
    with st.spinner("Calculating 2D Matrix (millions of data points)..."):
        layers, kpis = spatial.compute_spatial_grid_2d(config, res_a, resolution=res_sel, field_length=20.0)
    
    st.subheader("Key Spatial Statistics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean Remaining PAR", f"{kpis['mean_rem']:.1f}%")
    k2.metric("CV (Heterogeneity)", f"{kpis['cv_rem']*100:.1f}%")
    k3.metric("Shadow Frequency", f"{kpis['mean_shadow_freq']:.1f}%")
    k4.metric("Area < 50% PAR", f"{kpis['below_50_pct']:.1f}%")
    
    st.divider()
    
    fig_spatial_dict = {}
    
    def plot_heatmap(z_matrix, title, colorscale="Viridis", zmin=None, zmax=None):
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=layers['X'][0],
            y=layers['Y'][:, 0],
            colorscale=colorscale,
            zmin=zmin, zmax=zmax
        ))
        
        for rect in layers['pv_rects']:
            fig.add_shape(type="rect",
                x0=rect['x0'], y0=rect['y0'], x1=rect['x1'], y1=rect['y1'],
                line=dict(color="rgba(255, 255, 255, 0.8)", width=2),
                fillcolor="rgba(0, 0, 0, 0.5)"
            )
            
        fig.update_layout(
            title=title,
            xaxis_title="Distance parallel to rows (m)",
            yaxis_title="Distance across rows (m)",
            height=450,
            margin=dict(l=70, r=20, t=65, b=65)
        )
        return fig
        
    hm1, hm2, hm3 = st.tabs(["Remaining PAR", "Shadow Frequency", "PAR Loss"])
    
    with hm1:
        fig_hm_rem = plot_heatmap(layers['rem_par'], "Annual Remaining PAR (%)", "Viridis", 20, 100)
        st.plotly_chart(fig_hm_rem, use_container_width=True)
        fig_spatial_dict['rem_par'] = fig_hm_rem
        
    with hm2:
        fig_hm_freq = plot_heatmap(layers['shadow_freq'], "Direct Shadow Frequency (% of daylight hours)", "Plasma")
        st.plotly_chart(fig_hm_freq, use_container_width=True)
        fig_spatial_dict['shadow_freq'] = fig_hm_freq
        
    with hm3:
        fig_hm_loss = plot_heatmap(layers['par_loss'], "Annual PAR Loss (%)", "Reds", 0, 80)
        st.plotly_chart(fig_hm_loss, use_container_width=True)
        fig_spatial_dict['par_loss'] = fig_hm_loss
        
    st.markdown("### Seasonal PAR")
    sc1, sc2 = st.columns(2)
    with sc1:
        fig_spr = plot_heatmap(layers['par_spring'], "Spring PAR", "YlGnBu")
        st.plotly_chart(fig_spr, use_container_width=True)
        fig_spatial_dict['spring'] = fig_spr
        
        fig_aut = plot_heatmap(layers['par_autumn'], "Autumn PAR", "YlGnBu")
        st.plotly_chart(fig_aut, use_container_width=True)
        fig_spatial_dict['autumn'] = fig_aut
    with sc2:
        fig_sum = plot_heatmap(layers['par_summer'], "Summer PAR", "YlGnBu")
        st.plotly_chart(fig_sum, use_container_width=True)
        fig_spatial_dict['summer'] = fig_sum
        
        fig_win = plot_heatmap(layers['par_winter'], "Winter PAR", "YlGnBu")
        st.plotly_chart(fig_win, use_container_width=True)

# ==============================================================================
# TAB 4: CROP & VEGETATION COMPATIBILITY (UNIFIED)
# ==============================================================================
with tab_crops:
    st.markdown("""
    <div class="header-crops">
        <h2 style="margin:0; font-weight:800; color:white;">Crop & Vegetation Compatibility Engine</h2>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; color:white;">Integrated Agrivoltaic Suitability Modeling across Arable Crops, Medicinal & Special Crops, and Meadow Species</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "**IMPORTANT NOTICE:** Agronomic suitability is evaluated based on species-specific PAR thresholds, DLI integrals, "
        "microclimate indicators, and Ellenberg/Landolt ecological light values. Shading compatibility describes light availability "
        "and does not replace site-specific agronomic planning or formal DIN SPEC 91434 review."
    )

    st.divider()

    # Category Sub-tabs
    sub_arable, sub_medicinal, sub_meadow = st.tabs([
        "Arable Crops (Ackerbau)",
        "Medicinal & Special Crops (Arznei- & Sonderkulturen)",
        "Wet Meadow & Floodplain Species (Feuchtwiesen & Grünland)"
    ])

    # --- SUBTAB 1: ARABLE CROPS ---
    with sub_arable:
        st.subheader("Arable Crops Suitability Ranking (Balkendiagramm)")
        st.markdown("Comparison of suitability scores (0–100%) across all arable & agricultural crops in database:")
        
        arable_chart_data = []
        for r in crop_results:
            crop = CROP_REGISTRY[r.crop_id]
            lbl = translate_class(r.classification)
            color = "#059669" if "suitable" in lbl.lower() else ("#d97706" if "marginal" in lbl.lower() else "#dc2626")
            arable_chart_data.append({
                "Crop": f"{crop.name_en} ({crop.name_de})",
                "Score (%)": r.score * 100.0,
                "Class": lbl,
                "Color": color
            })
        df_arable_chart = pd.DataFrame(arable_chart_data).sort_values("Score (%)", ascending=True)

        chart_h = max(450, len(df_arable_chart) * 24)
        fig_arable_bar = px.bar(
            df_arable_chart, y="Crop", x="Score (%)", color="Class", orientation="h",
            title=f"All Arable Crops Suitability Scores ({len(df_arable_chart)} Species)", text_auto=".1f",
            color_discrete_map={"Suitable": "#059669", "Highly Suitable": "#047857", "Marginal": "#d97706", "Not Recommended": "#dc2626"}
        )
        fig_arable_bar.add_vline(x=80, line_dash="dash", line_color="#047857", annotation_text="Target Threshold (80%)")
        fig_arable_bar.add_vline(x=65, line_dash="dot", line_color="#d97706", annotation_text="Minimum Threshold (65%)")
        fig_arable_bar.update_layout(height=chart_h, margin=dict(l=0, r=20, t=40, b=0), xaxis=dict(range=[0, 105]))
        st.plotly_chart(fig_arable_bar, use_container_width=True)

        st.subheader("Arable Crops Suitability Matrix")
        df_arable = []
        for r in crop_results:
            crop = CROP_REGISTRY[r.crop_id]
            df_arable.append({
                "Crop Name (EN)": crop.name_en,
                "Crop Name (DE)": crop.name_de,
                "Crop Group": crop.crop_group.replace("_", " ").title(),
                "Score": f"{r.score*100:.1f}%",
                "Suitability Class": translate_class(r.classification),
                "Evidence Tier": r.evidence_tier,
                "Confidence": translate_confidence(r.confidence),
                "Limiting Factor": r.limiting_factor.replace("_", " ").title()
            })
        st.dataframe(pd.DataFrame(df_arable), use_container_width=True)

        # Interactive Plant Explorer
        st.divider()
        st.subheader("Interactive Plant Light Profile Explorer")
        st.markdown("Select any plant from the database to view its specific monthly light (DLI) diagram & agronomic details:")

        sel_cid = st.selectbox(
            "Select Plant / Crop:",
            options=[r.crop_id for r in crop_results],
            format_func=lambda cid: f"{CROP_REGISTRY[cid].name_en} ({CROP_REGISTRY[cid].name_de})",
            key="sel_arable_crop"
        )

        sel_r = next((r for r in crop_results if r.crop_id == sel_cid), crop_results[0])
        sel_crop = CROP_REGISTRY[sel_cid]

        days_in_m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        m_dli_agri = [metrics['monthly_par_agri'][m-1] / days_in_m[m-1] for m in range(1, 13)]
        m_dli_open = [metrics['monthly_par_open'][m-1] / days_in_m[m-1] for m in range(1, 13)]

        dli_min_val = getattr(sel_crop, 'DLI_min', 18.0)
        dli_target_val = getattr(sel_crop, 'DLI_target', 24.0)

        fig_single_plant = go.Figure()
        fig_single_plant.add_trace(go.Bar(
            x=m_names, y=m_dli_agri, name="Agri-PV DLI (mol/m²/d)",
            marker_color="#0284c7"
        ))
        fig_single_plant.add_trace(go.Bar(
            x=m_names, y=m_dli_open, name="Open-Field DLI (mol/m²/d)",
            marker_color="#94a3b8"
        ))
        fig_single_plant.add_hline(
            y=dli_target_val, line_dash="dash", line_color="#10b981",
            annotation_text=f"Target DLI ({dli_target_val:.1f} mol/m²/d)"
        )
        fig_single_plant.add_hline(
            y=dli_min_val, line_dash="dot", line_color="#f59e0b",
            annotation_text=f"Min DLI ({dli_min_val:.1f} mol/m²/d)"
        )
        fig_single_plant.update_layout(
            title=f"Monthly Daily Light Integral (DLI) Profile for {sel_crop.name_en} ({sel_crop.name_de})",
            xaxis_title="Month", yaxis_title="Daily Light Integral (mol/m²/d)",
            barmode="group", height=380, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_single_plant, use_container_width=True)

        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:6px solid #0284c7; border-radius:10px; padding:20px; margin-top:15px;">
            <h3 style="margin:0; color:#0f172a;">{sel_crop.name_en} <span style="font-size:1rem; color:#64748b; font-weight:normal;">({sel_crop.name_de})</span></h3>
            <div style="font-weight:700; color:#0284c7; margin-top:4px;">Suitability Score: {sel_r.score*100:.1f}% ({translate_class(sel_r.classification)})</div>
            <p style="margin-top:8px; color:#334155;">{ENGLISH_CROP_NOTES.get(sel_cid, sel_crop.notes_de)}</p>
            <div style="font-size:0.88rem; color:#475569;">
                <strong>Limiting Factor:</strong> {sel_r.limiting_factor.replace("_", " ").upper()} | 
                <strong>Evidence Tier:</strong> {sel_r.evidence_tier} ({translate_confidence(sel_r.confidence).upper()} CONFIDENCE)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- SUBTAB 2: MEDICINAL CROPS ---
    with sub_medicinal:
        st.subheader("Medicinal & Special Crops Light Availability (Balkendiagramm)")
        st.markdown(f"Comparison of annual relative PAR (%) and critical phase PAR (%) across all {len(crop_results_med)} medicinal species:")

        med_chart_data = []
        for r in crop_results_med:
            med_chart_data.append({
                "Species": r.crop_name,
                "Annual rPAR (%)": r.r_ann * 100.0,
                "Critical Phase rPAR (%)": r.r_crit * 100.0,
                "Class": translate_class(r.suitability_class)
            })
        df_med_chart = pd.DataFrame(med_chart_data)

        fig_med_bar = px.bar(
            df_med_chart, x="Species", y=["Annual rPAR (%)", "Critical Phase rPAR (%)"],
            barmode="group", title=f"Medicinal & Special Crops PAR Availability ({len(crop_results_med)} Species)", text_auto=".1f"
        )
        fig_med_bar.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0), yaxis=dict(range=[0, 110]))
        st.plotly_chart(fig_med_bar, use_container_width=True)

        st.subheader("Medicinal & Special Crops Suitability Matrix")
        df_med = []
        for r in crop_results_med:
            df_med.append({
                "Crop Name": r.crop_name,
                "Botanical Name": r.botanical_name,
                "Use Type": r.use_type.title(),
                "Annual rPAR": f"{r.r_ann*100:.1f}%",
                "Critical rPAR": f"{r.r_crit*100:.1f}%",
                "Suitability Class": translate_class(r.suitability_class),
                "Homogeneity": r.homogeneity_class.title(),
                "Limiting Factor": r.limiting_factor
            })
        st.dataframe(pd.DataFrame(df_med), use_container_width=True)

        # Interactive Medicinal Selector
        st.divider()
        st.subheader("Interactive Medicinal Species Light Profile Explorer")
        sel_med_id = st.selectbox(
            "Select Medicinal / Special Crop:",
            options=[r.crop_id for r in crop_results_med],
            format_func=lambda cid: f"{MED_CROP_REGISTRY[cid].display_name} ({MED_CROP_REGISTRY[cid].botanical_name})",
            key="sel_med_crop"
        )
        sel_med_r = next((r for r in crop_results_med if r.crop_id == sel_med_id), crop_results_med[0])
        sel_med_crop = MED_CROP_REGISTRY[sel_med_id]

        fig_single_med = go.Figure()
        fig_single_med.add_trace(go.Bar(
            x=m_names, y=[metrics['monthly_par_agri'][m-1] / days_in_m[m-1] for m in range(1, 13)],
            name="Agri-PV DLI (mol/m²/d)", marker_color="#059669"
        ))
        fig_single_med.add_trace(go.Bar(
            x=m_names, y=[metrics['monthly_par_open'][m-1] / days_in_m[m-1] for m in range(1, 13)],
            name="Open-Field DLI (mol/m²/d)", marker_color="#94a3b8"
        ))
        fig_single_med.add_hline(
            y=sel_med_crop.DLI_min, line_dash="dash", line_color="#10b981",
            annotation_text=f"Min DLI Threshold ({sel_med_crop.DLI_min:.1f} mol/m²/d)"
        )
        fig_single_med.update_layout(
            title=f"Monthly Daily Light Integral (DLI) Profile for {sel_med_crop.display_name}",
            xaxis_title="Month", yaxis_title="Daily Light Integral (mol/m²/d)",
            barmode="group", height=380, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_single_med, use_container_width=True)

    # --- SUBTAB 3: WET MEADOW & FLOODPLAIN SPECIES ---
    with sub_meadow:
        st.subheader("Wet Meadow & Floodplain Species Suitability Scores (Balkendiagramm)")
        st.markdown(f"Comparison of overall suitability scores (0–100) across all {len(crop_results_meadow)} meadow species:")

        meadow_chart_data = []
        for r in crop_results_meadow:
            meadow_chart_data.append({
                "Species": f"{r.display_name} (L={r.ellenberg_L})",
                "Score": r.score,
                "Light Suitability": r.light_class,
                "Zone": r.zone_hint
            })
        df_meadow_chart = pd.DataFrame(meadow_chart_data).sort_values("Score", ascending=True)

        chart_m_h = max(450, len(df_meadow_chart) * 24)
        fig_meadow_bar = px.bar(
            df_meadow_chart, y="Species", x="Score", color="Light Suitability", orientation="h",
            title=f"Wet Meadow Species Suitability Scores ({len(df_meadow_chart)} Species)", text_auto=".1f"
        )
        fig_meadow_bar.update_layout(height=chart_m_h, margin=dict(l=0, r=0, t=40, b=0), xaxis=dict(range=[0, 105]))
        st.plotly_chart(fig_meadow_bar, use_container_width=True)

        st.subheader("Wet Meadow & Floodplain Species Suitability Matrix")
        df_meadow = []
        for r in crop_results_meadow:
            df_meadow.append({
                "Species Name": r.display_name,
                "Botanical Name": r.botanical_name,
                "Group": r.species_group.title(),
                "Ellenberg L": r.ellenberg_L,
                "Ellenberg F": r.ellenberg_F,
                "Ellenberg N": r.ellenberg_N,
                "Score": f"{r.score:.1f}",
                "Light Suitability": r.light_class,
                "Hydrology Suitability": r.hydro_class,
                "Recommended Zone": r.zone_hint
            })
        st.dataframe(pd.DataFrame(df_meadow), use_container_width=True)

    st.divider()

    # SECTION 4: PREMIUM CELL-LEVEL SPATIAL CROP EXPLORER
    st.subheader("Cell-Level Spatial Suitability Explorer (Micro-Climate)")
    st.markdown("""
    Since the module rows shade some parts of the ground more than others, crop suitability changes across the pitch period.
    This Rigorous Spatial simulation models suitability at 11 separate cells across the row pitch period (from row-to-row spacing).
    """)

    # Run 1D spatial simulation
    x_points, spatial_par_annual, spatial_par_monthly = simulation.compute_spatial_annual_par(
        config['lat'], config['lon'], config['g_slope'], config['g_aspect'], 
        config['tau'], config['albedo'], config['geometry'], n_points=11
    )

    # Compute suitability at each cell
    spatial_scores = {crop_id: [] for crop_id in CROP_REGISTRY.keys()}
    for idx in range(11):
        cell_ann = spatial_par_annual[idx]
        cell_monthly = list(spatial_par_monthly[idx])
        
        # Evaluate each crop at cell idx
        for crop_id, crop in CROP_REGISTRY.items():
            res = evaluate_crop(
                crop, cell_ann, metrics['par_open_field'], cell_monthly, cv_par=0.0, has_hourly=True
            )
            spatial_scores[crop_id].append(res.score)
            
    # Plot Spatial Suitability Profile
    fig_spatial = go.Figure()

    # Plot top crops
    for crop_id in ["luzerne", "weizen", "hafer", "mais"]:
        crop = CROP_REGISTRY[crop_id]
        fig_spatial.add_trace(go.Scatter(
            x=x_points, 
            y=spatial_scores[crop_id],
            mode='lines+markers',
            name=crop.name_en,
            line=dict(width=3),
            marker=dict(size=7)
        ))
        
    # Add module shading box
    proj_w = 5.63 * np.cos(np.radians(15))
    m_start = (config['pitch'] - proj_w) / 2
    m_end = m_start + proj_w
    fig_spatial.add_vrect(
        x0=m_start, x1=m_end, 
        fillcolor="rgba(0,0,0,0.06)", layer="below", line_width=0, 
        annotation_text="Module Table"
    )

    fig_spatial.update_layout(
        title="Spatial Suitability Score across Row Pitch (0m = row gap center, middle = directly under module)",
        xaxis_title="Horizontal Distance across Row Pitch (m)",
        yaxis_title="Suitability Score (0-1)",
        yaxis=dict(range=[0, 1.05]),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_spatial, use_container_width=True)

    st.info("""
    **Agronomic Insights from Spatial Profile:** 
    Cereals like Wheat show high suitability in the row gap center (left & right) but drop significantly directly under the modules (shaded zone).
    Lucerne remains highly robust and suited across the entire row pitch. Maize is fully unsuited regardless of location.
    """)

    st.divider()

    # SECTION 5: TOP RECOMMENDED CROP DETAIL CARDS
    st.subheader("Detailed Crop Recommendation Cards")
    st.markdown("Detailed breakdown of the top recommended crops:")

    # Group crops by classification
    rec_crops = [r for r in crop_results if "geeignet" in r.classification.lower()]
    if not rec_crops:
        rec_crops = crop_results[:3]  # fallback to top 3 if none suitable
        
    cols = st.columns(len(rec_crops[:3]))

    for idx, r in enumerate(rec_crops[:3]):
        crop = CROP_REGISTRY[r.crop_id]
        
        # Color coding classes matching new labels
        class_lower = r.classification.lower()
        if "hauptkultur" in class_lower or "sonderkultur" in class_lower or "geeignet" in class_lower:
            card_class = "crop-card-geeignet"
            badge_class = "badge-geeignet"
        elif "prüfung" in class_lower or "grenzwertig" in class_lower:
            card_class = "crop-card-grenzwertig"
            badge_class = "badge-grenzwertig"
        else:
            card_class = "crop-card-nicht"
            badge_class = "badge-nicht"
            
        # Warning banner if evidence_tier == 'C'
        warning_html = ""
        if r.evidence_tier == 'C':
            warning_html = (
                f'<div style="background-color:#fffbeb; border-left:3px solid #d97706; padding:8px 12px; border-radius:4px; font-size:0.8rem; color:#b45309; margin-top:10px; line-height:1.35;">'
                f'For this crop, no reliable species-specific Agri-PV PAR curve is available. Evaluation is performed as a proxy based on light preference, crop group, and site-specific PAR.'
                f'</div>'
            )
            
        with cols[idx]:
            st.markdown(f"""
<div class="crop-card {card_class}">
<h3 style="margin:0; color:#0f172a;">{crop.name_en}</h3>
<div style="font-size:0.9rem; color:#64748b; font-style:italic; margin-bottom:8px;">(German: {crop.name_de})</div>
<div class="badge {badge_class}">{translate_class(r.classification)}</div>
<div style="margin-top:12px; font-size:1.6rem; font-weight:800; color:#0f172a;">Score: {r.score*100:.1f}%</div>
<p style="font-size:0.9rem; color:#334155; margin-top:8px; line-height:1.4;">{ENGLISH_CROP_NOTES.get(r.crop_id, r.notes_de)}</p>
{warning_html}
<div class="limiting-box">
<strong>Limiting Factor:</strong> {r.limiting_factor.replace("_", " ").upper()}<br/>
<strong>Evidence Tier:</strong> {r.evidence_tier} ({translate_confidence(r.confidence).upper()} CONFIDENCE)
</div>
</div>
            """, unsafe_allow_html=True)
            
            # Expander details
            with st.expander(f"View Agronomic Details for {crop.name_en}"):
                st.markdown(f"**Evidence Literature Sources:**")
                for src in r.sources:
                    st.markdown(f"- *{src}*")
                    
                st.markdown(f"**Component Scoring Breakdown:**")
                c_scores = r.component_scores
                st.write(f"- Annual PAR Adequacy (A): **{c_scores['A']*100:.1f}%**")
                st.write(f"- Seasonal PAR Sum (S): **{c_scores['S']*100:.1f}%**")
                st.write(f"- Critical Phase DLI (C): **{c_scores['C']*100:.1f}%**")
                st.write(f"- Spatial Homogeneity (H): **{c_scores['H']*100:.1f}%**")
                
                # Show growing calendar months
                st.markdown(f"**Growing Season Calendar:**")
                calendar_str = " | ".join([f"**{m_names[m-1]}**" if m in crop.growing_months else f"{m_names[m-1]}" for m in range(1, 13)])
                st.markdown(f"Months (active in bold): {calendar_str}")
                
                st.markdown(f"**Critical Light Sensitivity Window:**")
                crit_str = " | ".join([f"**{m_names[m-1]}**" if m in crop.critical_months else f"{m_names[m-1]}" for m in range(1, 13)])
                st.markdown(f"Months (critical in bold): {crit_str}")
                
    st.markdown("""
    <div style="background-color:#eff6ff; border-left:6px solid #3b82f6; padding:18px 24px; border-radius:8px; margin-top:20px;">
        <strong style="color:#1e3a8a;">Strategic Recommendation:</strong><br/>
        For fixed-tilt high-clearance systems (Category II under DIN SPEC 91434) with row pitches ≥ 8m, <strong>Lucerne</strong> 
        and robust C3 cereals (such as <strong>Oats</strong> and <strong>Spelt</strong>) represent the most reliable agricultural choice. 
        They maintain robust yields under partial shading and show high spatial homogeneity across the layout.
    </div>
    """, unsafe_allow_html=True)



# ==============================================================================
# TAB 5: ELECTRICAL & THERMAL RESULTS
# ==============================================================================
with tab_elec:
    st.markdown("""
    <div class="header-elec">
        <h2 style="margin:0; font-weight:800; color:white;">Electrical & Thermal Modeling</h2>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; color:white;">Specific Yield Calculations, Faiman Convective Ventilation, and Cell Temperature Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="yield-box">
        <div>
            <h3 style="margin:0; color:#1e293b; font-family:'Outfit';">SPECIFIC YIELD BONUS (ELECTRICAL)</h3>
            <p style="margin:4px 0 0 0; font-size:0.95rem; color:#64748b;">Annual energy generation advantage per installed kWp due to heightened mounting (2.10m vs 0.80m)</p>
        </div>
        <div style="text-align:right;">
            <span style="font-size:2.4rem; font-weight:800; color:#0f172a;">+{metrics['y_bonus']:.1f} kWh/kWp <span style="font-size:1.4rem; color:#10b981; font-weight:700;">(+{metrics['temp_bonus_pct']:.2f}%)</span></span>
            <div style="font-size:0.95rem; font-weight:600; color:#10b981; margin-top:2px;">↑ Agri-PV: {metrics['ya_spec']:.0f} vs Standard: {metrics['ys_spec']:.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Convective Cooling metrics row
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.metric("Agri-PV Cell Temp", f"{metrics['ta_cell']:.1f} °C", f"−{metrics['delta_t']:.1f}°C vs Standard",
                  help="Annual arithmetic mean cell temperature during daylight hours (GHI > 50 W/m²).")
    with t_col2:
        st.metric("Std. PV Cell Temp", f"{metrics['ts_cell']:.1f} °C", "Restricted ventilation",
                  help="Mean cell temperature for a standard 0.80m high ground-mounted system.")
    with t_col3:
        st.metric("Thermal Model", "Faiman (2008)", "Wind-corrected, log profile",
                  help="Rigorous cell temperature model using Koehl coefficients (u₀ = 25.0, u₁ = 6.84).")
    with t_col4:
        st.metric("Log Wind Speed", f"Height corrected", "Uses PVGIS WS10m series",
                  help="Wind speed at the module center is computed using the logarithmic boundary layer profile.")
                  
    st.divider()
    
    # Seasonal performance analysis
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
        fig_irr = px.bar(
            m_comp, 
            barmode='group', 
            labels={'index': 'Month', 'value': 'kWh/m²'}, 
            color_discrete_sequence=["#1e293b", "#94a3b8", "#cbd5e1"]
        )
        fig_irr.update_layout(xaxis={'categoryorder':'array', 'categoryarray':m_names}, height=380, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_irr, use_container_width=True)
        
    with gm2:
        st.markdown("**Monthly PAR Growth Potential (mol/m²)**")
        fig_par = px.bar(
            m_par, 
            barmode='group', 
            labels={'index': 'Month', 'value': 'mol/m²'}, 
            color_discrete_sequence=["#16a34a", "#94a3b8"]
        )
        fig_par.update_layout(xaxis={'categoryorder':'array', 'categoryarray':m_names}, height=380, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_par, use_container_width=True)
        
    st.divider()
    
    # Log wind profile explanation
    st.subheader("Logarithmic Wind Profile & Faiman Convective Cooling")
    st.markdown("""
    Mounted height doesn't just dictate shadow pathing; it directly affects the module's thermal balance.
    Under the Faiman (2008) thermal model, module cell temperature is calculated as:
    """)
    st.latex(r"T_{cell} = T_{ambient} + \frac{G_{POA}}{u_0 + u_1 \cdot v_{eff}}")
    st.latex(r"v_{eff} = v_{10m} \cdot \frac{\ln(h_{mod} / z_0)}{\ln(10 / z_0)}")
    
    ws_mean = res_a['wind_speed'][res_a['ghi'] > 50].mean()
    v_eff_a = thermal.effective_wind_speed(ws_mean, 2.10)
    v_eff_s = thermal.effective_wind_speed(ws_mean, 0.80)
    
    st.markdown(f"""
    - **Broadband Daylight Wind Speed (at 10m):** **{ws_mean:.2f} m/s**
    - **Logarithmic Effective Wind at Agri-PV center (H=2.85m):** **{v_eff_a:.2f} m/s**
    - **Logarithmic Effective Wind at Standard PV center (H=1.55m):** **{v_eff_s:.2f} m/s**
    - **Physical Delta:** Heightened mounting clears the ground boundary layer, exposing elevated modules to **+{((v_eff_a/v_eff_s)-1.0)*100.0:.1f}% higher wind speeds**. This increases convective cooling, lowering the mean cell temperature by **{metrics['delta_t']:.1f}°C** and preventing temperature-related power degradation.
    """)





# ==============================================================================
# TAB 6: REPORT CONFIGURATION & DOWNLOADS
# ==============================================================================
with tab_report:
    st.markdown("""
    <div class="header-din">
        <h2 style="margin:0; font-weight:800; color:white;">Report Configuration & Data Downloads</h2>
        <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem; color:white;">Configure Project Metadata, Generate 15-Page Technical PDF Validation Report, and Export Raw Simulation Data</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Report Configuration & Metadata")
    st.markdown("Customize project details and metadata embedded in the generated PDF technical validation report:")

    rc1, rc2 = st.columns(2)
    with rc1:
        rep_project_name = st.text_input("Project Name:", value=config.get("project_name", "Agri-PV Site Validation"), key="rep_proj_name")
        rep_prepared_by = st.text_input("Prepared By (Author / Company):", value="SunFarming Engineering", key="rep_prep_by")
    with rc2:
        rep_location = st.text_input("Project Location / Coordinates:", value=f"Lat: {config.get('latitude', 52.5):.3f}°, Lon: {config.get('longitude', 13.4):.3f}°", key="rep_loc_str")
        rep_notes = st.text_area("Custom Report Notes / Remarks:", value="Technical feasibility and crop suitability assessment for high-clearance Agri-PV installation.", height=68, key="rep_notes_text")

    st.markdown("---")

    st.subheader("Export & Download Center")

    export_cols = ['ghi', 'dni', 'dhi', 'temp_air', 't_avg', 't_cell', 'temp_factor', 'g_g', 'par']
    col_names   = ['GHI [W/m²]', 'DNI [W/m²]', 'DHI [W/m²]', 'T_ambient [°C]',
                   'Beam_transmission', 'T_cell [°C]', 'Temp_factor', 'G_ground [W/m²]', 'PAR [μmol/m²/s]']
    export_a = res_a[export_cols].copy(); export_a.columns = col_names

    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        st.markdown("#### Technical PDF Validation Report")
        st.markdown("Generates a comprehensive 15-page PDF document including executive summary, spatial heatmaps, electrical analysis, and crop compatibility profiles.")
        
        if st.button("Generate Technical Report (PDF)", use_container_width=True, type="primary", key="generate_pdf_btn"):
            with st.spinner("Generating 15-page PDF report with interactive charts..."):
                import importlib
                import report.report_styles
                import report.report_generator
                import report.report_charts
                importlib.reload(report.report_styles)
                importlib.reload(report.report_charts)
                importlib.reload(report.report_generator)
                from report.report_generator import ReportGenerator

                # Build DLI curve charts for top 3 crops
                from crop_suitability import CROP_REGISTRY as _CR
                _days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                _mnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                _m_agri = metrics['monthly_par_agri']
                _m_open = metrics['monthly_par_open']

                top_3_crop_data = []
                for cr in crop_results[:3]:
                    crop_profile = _CR.get(cr.crop_id)
                    if not crop_profile:
                        continue

                    dli_agri_m = [_m_agri[i] / _days[i] if _days[i] > 0 else 0 for i in range(12)]
                    dli_open_m = [_m_open[i] / _days[i] if _days[i] > 0 else 0 for i in range(12)]

                    fig_dli = go.Figure()
                    fig_dli.add_trace(go.Bar(x=_mnames, y=dli_agri_m, name="Agri-PV (mol/m²/d)", marker_color="#10b981"))
                    fig_dli.add_trace(go.Scatter(x=_mnames, y=dli_open_m, name="Open Field (mol/m²/d)", mode="lines+markers", marker_color="#475569"))
                    fig_dli.add_hline(y=crop_profile.DLI_target, line_dash="dash", line_color="#059669", annotation_text="DLI Target")
                    fig_dli.add_hline(y=crop_profile.DLI_min, line_dash="dot", line_color="#d97706", annotation_text="DLI Min")

                    for m in crop_profile.critical_months:
                        fig_dli.add_vrect(x0=m-1.4, x1=m-0.6, fillcolor="rgba(239,68,68,0.1)", line_width=0)

                    fig_dli.update_layout(
                        height=300, margin=dict(l=50, r=10, t=30, b=30),
                        yaxis_title="DLI (mol/m²/d)",
                        title=f"{crop_profile.display_name if hasattr(crop_profile, 'display_name') else cr.crop_id.capitalize()} — Monthly DLI",
                        title_font_size=13,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=9)
                    )

                    par_ref = metrics['par_open_field']
                    r_ann_pct = metrics['pa'] / par_ref * 100 if par_ref > 0 else 0
                    crit_months = crop_profile.critical_months
                    if crit_months:
                        s_agri = sum(_m_agri[m - 1] for m in crit_months)
                        s_ref = sum(_m_open[m - 1] for m in crit_months)
                        r_crit_pct = s_agri / s_ref * 100 if s_ref > 0 else r_ann_pct
                    else:
                        r_crit_pct = r_ann_pct

                    gs_months = crop_profile.growing_months or crop_profile.growing_season_months
                    if gs_months:
                        gs_days = sum(_days[m-1] for m in gs_months)
                        gs_par = sum(_m_agri[m-1] for m in gs_months)
                        mean_dli_val = gs_par / gs_days if gs_days > 0 else 0
                    else:
                        mean_dli_val = sum(dli_agri_m) / 12

                    top_3_crop_data.append({
                        'result': cr,
                        'profile': crop_profile,
                        'fig_dli': fig_dli,
                        'r_ann': r_ann_pct,
                        'r_crit': r_crit_pct,
                        'mean_dli': mean_dli_val,
                        'cv_par': metrics['cv_par'] * 100,
                    })

                # Evaluate meadow species for PDF
                from meadow_suitability import evaluate_all_meadow_species as _eval_meadow
                _meadow_pdf = _eval_meadow(
                    annual_PAR_agri=metrics['pa'],
                    annual_PAR_openfield=metrics['par_open_field'],
                    monthly_PAR_agri=metrics['monthly_par_agri'],
                    monthly_PAR_openfield=metrics['monthly_par_open'],
                    cv_PAR=metrics['cv_par'],
                )

                figures = {
                    'heat': fig_heat,
                    'crop': fig_arable_bar if 'fig_arable_bar' in locals() else None,
                    'elec': None,
                    'weather': fig_irr,
                    'layout': fig_sp,
                    'spatial_dict': fig_spatial_dict,
                    'radar': None,
                    'top_3_crops': top_3_crop_data,
                    'meadow_results': _meadow_pdf,
                }

                # Copy updated config
                report_config = dict(config)
                report_config["project_name"] = rep_project_name
                report_config["prepared_by"] = rep_prepared_by
                report_config["location"] = rep_location
                report_config["notes"] = rep_notes

                metrics['spatial_kpis'] = kpis
                generator = ReportGenerator(report_config, metrics, crop_results, figures)
                st.session_state['pdf_bytes'] = generator.generate().getvalue()

        if 'pdf_bytes' in st.session_state:
            st.download_button(
                "Download Ready: Technical Report (PDF)",
                st.session_state['pdf_bytes'],
                "Agri-PV_Technical_Validation_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="din_pdf_download"
            )

    with c_exp2:
        st.markdown("#### Raw Hourly Simulation Data (CSV)")
        st.markdown("Download full 8,760-hour simulation dataset containing hourly solar positions, irradiance components, cell temperatures, and ground PAR values.")
        st.download_button(
            "Download Hourly Simulation Data (CSV)",
            export_a.to_csv().encode('utf-8'),
            "agri_pv_hourly_calculations.csv",
            mime="text/csv",
            use_container_width=True,
            key="din_csv_download"
        )
