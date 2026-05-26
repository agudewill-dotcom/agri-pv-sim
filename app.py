import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import simulation
from crop_profiles import CROP_REGISTRY
from crop_scoring import evaluate_all_crops

st.set_page_config(
    page_title="Agri-PV Strategic Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Executive Styling ---
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
    
    /* Header Card styling */
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 35px;
        border-radius: 16px;
        margin-bottom: 30px;
        border-left: 8px solid #10b981;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 5px;
        color: #10b981;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 20px;
    }
    
    /* Premium Metric Card */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #f1f5f9;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
        border-top: 5px solid #0f172a;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a !important;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        color: #10b981;
    }
    
    /* Info Box */
    .info-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
    }
    
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 30px;
        color: #78350f;
        font-size: 0.9rem;
    }
    
    /* Button formatting */
    div.stButton > button {
        background-color: #0f172a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    
    div.stButton > button:hover {
        background-color: #1e293b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- RUN SIMULATION & CACHE ---
config, res_a, res_s, metrics = simulation.render_sidebar_and_run()

# Save all objects in session_state for access on sub-pages
st.session_state['config'] = config
st.session_state['res_a'] = res_a
st.session_state['res_s'] = res_s
st.session_state['metrics'] = metrics

# Compute top recommended crops for dashboard overview
par_ref = metrics['par_open_field']
par_ann = metrics['pa']
monthly_par = metrics['monthly_par_agri']
cv_par = metrics['cv_par']

crop_results = evaluate_all_crops(
    par_ann=par_ann,
    par_ref=par_ref,
    monthly_par=monthly_par,
    cv_par=cv_par,
    has_hourly=True
)
st.session_state['crop_results'] = crop_results

# --- EXECUTIVE OVERVIEW ---
st.markdown(f"""
<div class="header-box">
    <div class="header-title">🌾 Agri-PV Strategic Analytics Dashboard</div>
    <div class="header-subtitle">High-Fidelity Physical Modeling & Agronomic Decision Support System (v9.0)</div>
    <div style="font-size: 1.6rem; font-weight: 700; color: #ffffff;">
        +{(metrics['va']-metrics['vs']):.0f} kWh/m² Light Advantage | 
        +{(metrics['va']/metrics['vs']-1.0)*100.0:.1f}% Ground Irradiance vs. Std. PV
    </div>
</div>
""", unsafe_allow_html=True)

st.subheader("💡 Strategic Physical Delta (Agri-PV vs. Standard PV)")

# KPI Grid (Row 1)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Agricultural Light</div>
        <div class="metric-value">{(metrics['va']/metrics['vo'])*100.0:.1f}%</div>
        <div class="metric-delta">+{((metrics['va']/metrics['vs'])-1.0)*100.0:.1f}% vs Standard PV</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annual PAR Sum</div>
        <div class="metric-value">{metrics['pa']:.0f} mol</div>
        <div class="metric-delta">+{((metrics['pa']/metrics['ps'])-1.0)*100.0:.1f}% vs Standard PV</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Electrical Power Bonus</div>
        <div class="metric-value">+{metrics['temp_bonus_pct']:.2f}%</div>
        <div class="metric-delta">-{metrics['delta_t']:.1f}°C cooler module temp</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Specific Yield Bonus</div>
        <div class="metric-value">+{metrics['y_bonus']:.1f} kWh</div>
        <div class="metric-delta">Agri-PV: {metrics['ya_spec']:.0f} kWh/kWp</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Navigation Grid
st.subheader("🧭 Detailed System Analytics (Select in Sidebar or Click Below)")
n_col1, n_col2, n_col3 = st.columns(3)

with n_col1:
    st.markdown("""
    <div class="info-container" style="min-height: 250px; border-top: 4px solid #3b82f6;">
        <h4 style="margin-top:0; color:#1e3a8a;">🌾 High-Fidelity Light Results</h4>
        <p style="font-size:0.9rem; color:#475569; min-height:80px;">
            Examine spatial shadow profiles, 2D irradiance heatmaps, and hourly diffuse sky factors. 
            Visualize the Hottel periodic sky blockage and ground albedo reflection models.
        </p>
        <div style="font-size:0.9rem; font-weight:600; color:#2563eb;">→ Page: Light Results</div>
    </div>
    """, unsafe_allow_html=True)

with n_col2:
    # Get top 3 suitable crops
    top_3 = [r for r in crop_results if r.classification in {"sehr gut geeignet", "geeignet"}][:3]
    top_3_names = ", ".join([r.crop_name_de for r in top_3]) if top_3 else "Luzerne"
    
    st.markdown(f"""
    <div class="info-container" style="min-height: 250px; border-top: 4px solid #10b981;">
        <h4 style="margin-top:0; color:#064e3b;">🌱 Agronomic Suitability</h4>
        <p style="font-size:0.9rem; color:#475569; min-height:80px;">
            Rigorous multi-component suitability scoring for 11 Central European field crops. 
            Explore crop-specific growing calendars, confidence tiers, and limiting factors.
        </p>
        <div style="font-size:0.85rem; font-weight:700; color:#047857; margin-bottom:10px;">Top Crops: {top_3_names}</div>
        <div style="font-size:0.9rem; font-weight:600; color:#10b981;">→ Page: Kultureignung</div>
    </div>
    """, unsafe_allow_html=True)

with n_col3:
    st.markdown("""
    <div class="info-container" style="min-height: 250px; border-top: 4px solid #f59e0b;">
        <h4 style="margin-top:0; color:#78350f;">⚡ Electrical & Thermal Modeling</h4>
        <p style="font-size:0.9rem; color:#475569; min-height:80px;">
            Analyze wind-corrected cell temperatures using the Faiman (2008) model. 
            Observe temperature-dependent efficiency coefficients, monthly specific generation, and backsheet ventilation.
        </p>
        <div style="font-size:0.9rem; font-weight:600; color:#f59e0b;">→ Page: Electrical Results</div>
    </div>
    """, unsafe_allow_html=True)

# System Profile Details
st.subheader("📐 System Architecture Profile")
st.markdown(f"""
<div style="background-color: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
    <table style="width:100%; border-collapse:collapse; color:#1e293b;">
        <tr style="border-bottom: 2px solid #e2e8f0; font-weight: 700; text-transform: uppercase; font-size: 0.85rem; color: #64748b;">
            <th style="text-align: left; padding: 12px;">Component</th>
            <th style="text-align: left; padding: 12px;">Agri-PV System (Elevated)</th>
            <th style="text-align: left; padding: 12px;">Standard Ground-PV</th>
            <th style="text-align: left; padding: 12px;">Scientific Significance</th>
        </tr>
        <tr style="border-bottom: 1px solid #f1f5f9;">
            <td style="padding: 12px; font-weight: 600;">Mounting Clearance (H)</td>
            <td style="padding: 12px; color: #10b981; font-weight: 600;">2.10 m (lichthöhe)</td>
            <td style="padding: 12px; color: #64748b;">0.80 m</td>
            <td style="padding: 12px; font-size: 0.9rem; color: #475569;">Governs convective airflow under the modules and physical tractor clearance.</td>
        </tr>
        <tr style="border-bottom: 1px solid #f1f5f9; background-color: #f8fafc;">
            <td style="padding: 12px; font-weight: 600;">Row Pitch (P)</td>
            <td style="padding: 12px; font-weight: 600;">{config['pitch']:.2f} m</td>
            <td style="padding: 12px; font-weight: 600;">{config['pitch']:.2f} m</td>
            <td style="padding: 12px; font-size: 0.9rem; color: #475569;">Determines the ground row space and relative shade periodicity.</td>
        </tr>
        <tr style="border-bottom: 1px solid #f1f5f9;">
            <td style="padding: 12px; font-weight: 600;">Module Hardware</td>
            <td style="padding: 12px;">SUNfarming SF600-72N (Bifacial)</td>
            <td style="padding: 12px;">SUNfarming SF600-72N (Bifacial)</td>
            <td style="padding: 12px; font-size: 0.9rem; color: #475569;">Identical hardware to ensure comparative rigor in performance delta.</td>
        </tr>
        <tr style="border-bottom: 1px solid #f1f5f9; background-color: #f8fafc;">
            <td style="padding: 12px; font-weight: 600;">Transparency (τ)</td>
            <td style="padding: 12px; font-weight: 600;">{config['tau']*100:.0f}%</td>
            <td style="padding: 12px; font-weight: 600;">{config['tau']*100:.0f}%</td>
            <td style="padding: 12px; font-size: 0.9rem; color: #475569;">Governs beam and diffuse transparency through semi-transparent modules.</td>
        </tr>
        <tr style="border-bottom: 1px solid #f1f5f9;">
            <td style="padding: 12px; font-weight: 600;">Site Coordinates</td>
            <td style="padding: 12px;" colspan="2">Latitude: {config['lat']:.4f} | Longitude: {config['lon']:.4f}</td>
            <td style="padding: 12px; font-size: 0.9rem; color: #475569;">Uses local PVGIS SARAH-2 hourly solar series and NASA SRTM satellite topography.</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# Scientific Disclaimer
st.markdown(f"""
<div class="disclaimer-box">
    <strong>⚠️ Scientific Disclaimer:</strong><br/>
    This dashboard implements the <em>Agri-PV Simulation Physics Engine (v9.0)</em>. All outputs represent comparative 
    estimates based on physical first principles (Hottel crossed-strings, Faiman thermal log profiles, McCree PAR spectrum) 
    and are intended for early-stage strategic planning. Suitability recommendations are literature-backed agronomical guidelines 
    and do not replace local on-site agricultural soil tests or expert agronomy consultations.
</div>
""", unsafe_allow_html=True)
