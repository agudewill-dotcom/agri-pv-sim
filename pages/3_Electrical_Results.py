import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import thermal
import simulation

st.set_page_config(
    page_title="Agri-PV Electrical Results",
    page_icon="⚡",
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
        background: linear-gradient(135deg, #7c2d12 0%, #ea580c 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 6px solid #ffedd5;
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
    <h2 style="margin:0; font-weight:800;">⚡ Electrical & Thermal Modeling</h2>
    <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem;">Specific Yield Calculations, Faiman Convective Ventilation, and Cell Temperature Analysis</p>
</div>
""", unsafe_allow_html=True)

# SECTION 1: SPECIFIC YIELD BONUS
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

# SECTION 2: THERMAL KPI GRID
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

# SECTION 3: MONTHLY PERFORMANCE CHARTS
st.subheader("📊 Seasonal Performance Analysis")

# Enforced calendar order monthly dataframes
m_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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

# SECTION 4: THERMAL MODEL SUMMARY
st.subheader("🌡️ Logarithmic Wind Profile & Faiman Convective Cooling")
st.markdown("""
Mounted height doesn't just dictate shadow pathing; it directly affects the module's thermal balance.
Under the Faiman (2008) thermal model, module cell temperature is calculated as:
""")
st.latex(r"T_{cell} = T_{ambient} + \frac{G_{POA}}{u_0 + u_1 \cdot v_{eff}}")
st.latex(r"v_{eff} = v_{10m} \cdot \frac{\ln(h_{mod} / z_0)}{\ln(10 / z_0)}")

# Compute average wind speeds for display
ws_mean = res_a['wind_speed'][res_a['ghi'] > 50].mean()
v_eff_a = thermal.effective_wind_speed(ws_mean, 2.10)
v_eff_s = thermal.effective_wind_speed(ws_mean, 0.80)

st.markdown(f"""
- **Broadband Daylight Wind Speed (at 10m):** **{ws_mean:.2f} m/s**
- **Logarithmic Effective Wind at Agri-PV center (H=2.85m):** **{v_eff_a:.2f} m/s**
- **Logarithmic Effective Wind at Standard PV center (H=1.55m):** **{v_eff_s:.2f} m/s**
- **Physical Delta:** Heightened mounting clears the ground boundary layer, exposing elevated modules to **+{((v_eff_a/v_eff_s)-1.0)*100.0:.1f}% higher wind speeds**. This increases convective cooling, lowering the mean cell temperature by **{metrics['delta_t']:.1f}°C** and preventing temperature-related power degradation.
""")
