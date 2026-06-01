import sys

def inject_spatial_tab():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    injection_code = """
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
            margin=dict(l=50, r=20, t=50, b=50)
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
        fig_spr = plot_heatmap(layers['par_spring'], "Spring PAR (mol/m²)", "YlGnBu")
        st.plotly_chart(fig_spr, use_container_width=True)
        fig_spatial_dict['spring'] = fig_spr
        
        fig_aut = plot_heatmap(layers['par_autumn'], "Autumn PAR (mol/m²)", "YlGnBu")
        st.plotly_chart(fig_aut, use_container_width=True)
        fig_spatial_dict['autumn'] = fig_aut
    with sc2:
        fig_sum = plot_heatmap(layers['par_summer'], "Summer PAR (mol/m²)", "YlGnBu")
        st.plotly_chart(fig_sum, use_container_width=True)
        fig_spatial_dict['summer'] = fig_sum
        
        fig_win = plot_heatmap(layers['par_winter'], "Winter PAR (mol/m²)", "YlGnBu")
        st.plotly_chart(fig_win, use_container_width=True)
        fig_spatial_dict['winter'] = fig_win

"""
    
    # Injection before TAB 3
    target = "# TAB 3: AGRONOMIC SUITABILITY (CROP COMPATIBILITY)"
    if target in content and "TAB 2.5: SPATIAL HEATMAPS" not in content:
        content = content.replace(target, injection_code + "\n\n" + target)
        
    # We also need to add fig_spatial_dict and kpis to figures and metrics
    # Let's find: 'elec': None,
    target_figures = "'layout': fig_sp,"
    if target_figures in content and "'spatial_dict': fig_spatial_dict" not in content:
        content = content.replace(target_figures, target_figures + "\n        'spatial_dict': fig_spatial_dict,")
        
    # Wait, kpis is a local variable in tab_spatial block. We can just add it to metrics dictionary before generating report.
    # Look for: generator = ReportGenerator(config, metrics, crop_results, figures)
    target_generator = "generator = ReportGenerator(config, metrics, crop_results, figures)"
    if target_generator in content:
        content = content.replace(target_generator, "metrics['spatial_kpis'] = kpis\n                " + target_generator)
        
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
if __name__ == '__main__':
    inject_spatial_tab()
