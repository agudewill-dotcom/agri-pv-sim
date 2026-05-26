import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import simulation
from crop_profiles import CROP_REGISTRY
from crop_scoring import evaluate_all_crops, evaluate_crop

st.set_page_config(
    page_title="Agri-PV Kultureignung",
    page_icon="🌱",
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
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 6px solid #a7f3d0;
    }
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

crop_results = st.session_state.get('crop_results')
if crop_results is None:
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

st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-weight:800;">🌱 Agronomic Suitability Engine</h2>
    <p style="margin:5px 0 0 0; opacity:0.9; font-size:1.05rem;">Literature-Backed Crop Compatibility Modeling and spatial micro-climate scoring</p>
</div>
""", unsafe_allow_html=True)

# SECTION 1: DETAILED RANKING TABLE
st.subheader("📊 Farm-Average Crop Suitability Ranking")
st.markdown("All 11 arable crops sorted by composite suitability score based on farm-average light levels:")

df_ranking = pd.DataFrame([
    {
        "Kultur": r.crop_name_de,
        "Score": f"{r.score*100:.1f}%",
        "Eignungsklasse": r.classification.upper(),
        "Konfidenz": f"{r.confidence.upper()} ({r.confidence_value*100:.0f}%)",
        "Evidenzstärke": f"Tier {r.evidence_tier}",
        "Hauptengpass": r.limiting_factor.replace("_", " ").upper(),
        "Jahres-PAR (min/target)": f"{r.par_min_abs:.0f} / {r.par_target_abs:.0f} mol"
    }
    for r in crop_results
])

# Styling classification color
def color_class(val):
    if "SEHR GUT" in val or "GEEIGNET" in val and "GRENZ" not in val:
        return 'color: #065f46; font-weight: 600;'
    elif "GRENZ" in val:
        return 'color: #92400e; font-weight: 600;'
    else:
        return 'color: #991b1b; font-weight: 600;'

st.dataframe(
    df_ranking.style.applymap(color_class, subset=['Eignungsklasse']),
    use_container_width=True,
    hide_index=True
)

st.divider()

# SECTION 2: PREMIUM CELL-LEVEL SPATIAL CROP Explorer
st.subheader("🌾 Cell-Level Spatial Suitability Explorer (Micro-Climate)")
st.markdown("""
Since the module rows shade some parts of the ground more than others, crop suitability changes across the pitch period.
This Rigorous Spatial simulation models suitability at 11 separate cells across the row pitch period (from row-to-row spacing).
""")

# Run 1D spatial simulation
x_points, spatial_par_annual, spatial_par_monthly = simulation.compute_spatial_annual_par(
    config['lat'], config['lon'], config['g_slope'], config['g_aspect'], 
    config['tau'], config['albedo'], config['pitch'], n_points=11
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
for crop_id in ["luzerne", "winterweizen", "hafer", "mais"]:
    crop = CROP_REGISTRY[crop_id]
    fig_spatial.add_trace(go.Scatter(
        x=x_points, 
        y=spatial_scores[crop_id],
        mode='lines+markers',
        name=crop.name_de,
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
💡 **Agronomic Insights from Spatial Profile:** 
Cereals like Wheat (Weizen) show high suitability in the row gap center (left & right) but drop significantly directly under the modules (shaded zone).
Lucerne (Luzerne) remains highly robust and suited across the entire row pitch. Maize (Mais) is fully unsuited regardless of location.
""")

st.divider()

# SECTION 3: TOP RECOMMENDED CROP DETAIL CARDS
st.subheader("🌱 Detailed Crop Recommendation Cards")
st.markdown("Detailed breakdown of the top recommended crops:")

# Group crops by classification
rec_crops = [r for r in crop_results if r.classification in {"sehr gut geeignet", "geeignet"}]
if not rec_crops:
    rec_crops = crop_results[:3]  # fallback to top 3 if none suitable

cols = st.columns(len(rec_crops[:3]))

for idx, r in enumerate(rec_crops[:3]):
    crop = CROP_REGISTRY[r.crop_id]
    card_class = "crop-card-geeignet" if r.classification == "geeignet" else "crop-card-grenzwertig"
    badge_class = "badge-geeignet" if r.classification == "geeignet" else "badge-grenzwertig"
    
    with cols[idx]:
        st.markdown(f"""
        <div class="crop-card {card_class}">
            <h3 style="margin:0; color:#0f172a;">{r.crop_name_de}</h3>
            <div style="font-size:0.9rem; color:#64748b; font-style:italic; margin-bottom:8px;">{crop.name_en}</div>
            <div class="badge {badge_class}">{r.classification}</div>
            <div style="margin-top:12px; font-size:1.6rem; font-weight:800; color:#0f172a;">Score: {r.score*100:.1f}%</div>
            <p style="font-size:0.9rem; color:#334155; margin-top:8px; line-height:1.4;">{r.notes_de}</p>
            <div class="limiting-box">
                <strong>Limiting Factor:</strong> {r.limiting_factor.replace("_", " ").upper()}<br/>
                <strong>Evidence Tier:</strong> {r.evidence_tier} ({r.confidence.upper()} CONFIDENCE)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expander details
        with st.expander(f"📖 View Agronomic Details for {r.crop_name_de}"):
            st.markdown(f"**Evidence Literature Sources:**")
            for src in r.sources:
                st.markdown(f"- *{src}*")
            
            st.markdown(f"**Component Scoring Breakdown:**")
            c_scores = r.component_scores
            st.write(f"- 📆 Annual PAR Adequacy (A): **{c_scores['A']*100:.1f}%**")
            st.write(f"- 📈 Seasonal PAR Sum (S): **{c_scores['S']*100:.1f}%**")
            st.write(f"- 🗓️ Critical Phase DLI (C): **{c_scores['C']*100:.1f}%**")
            st.write(f"- 🔲 Spatial Homogeneity (H): **{c_scores['H']*100:.1f}%**")
            
            # Show growing calendar months
            st.markdown(f"**Growing Season Calendar:**")
            calendar_str = " | ".join([f"**{m}**" if m in crop.growing_months else f"{m}" for m in range(1, 13)])
            st.markdown(f"Months (active in bold): `{calendar_str}`")
            
            st.markdown(f"**Critical Light Sensitivity Window:**")
            crit_str = " | ".join([f"**{m}**" if m in crop.critical_months else f"{m}" for m in range(1, 13)])
            st.markdown(f"Months (critical in bold): `{crit_str}`")

# Bottom strategic message
st.markdown("""
<div style="background-color:#eff6ff; border-left:6px solid #3b82f6; padding:18px 24px; border-radius:8px; margin-top:20px;">
    <strong style="color:#1e3a8a;">💡 Strategic Recommendation:</strong><br/>
    For dynamic nachgeführte systems (Category II under DIN SPEC 91434) with row pitches ≥ 8m, <strong>Luzerne (Lucerne)</strong> 
    and robust C3 cereals (such as <strong>Hafer</strong> and <strong>Dinkel</strong>) represent the most reliable agricultural choice. 
    They maintain robust yields under partial shading and show high spatial homogeneity across the layout.
</div>
""", unsafe_allow_html=True)
