import sys

def inject_report():
    with open(r'report\report_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    call_target = "self.create_page_7_light_results()"
    call_injection = "        self.create_page_7_b_spatial_heatmaps()"
    if call_target in content and "create_page_7_b_spatial_heatmaps" not in content:
        content = content.replace(call_target, call_target + "\n" + call_injection)
        
    def_target = "def create_page_8_method_electrical(self):"
    
    def_injection = """
    def create_page_7_b_spatial_heatmaps(self):
        if 'spatial_dict' not in self.figures or not self.figures['spatial_dict']:
            return
            
        sp_figs = self.figures['spatial_dict']
        sp_kpis = self.metrics.get('spatial_kpis', {})
        
        # PAGE A: Annual PAR and KPIs
        self.story.append(Paragraph("Spatial Light and PAR Heatmaps", self.styles['Heading1']))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("A high-resolution 2D spatio-temporal grid was simulated across 8760 hours to compute the exact shadow patterns from the PV array. The PV module rows are indicated by overlaid rectangles. The X-axis runs parallel to the rows, while the Y-axis runs cross-row.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        if 'rem_par' in sp_figs:
            img = export_plotly_to_image(sp_figs['rem_par'], width=600, height=350)
            if img: self.story.append(img)
            
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph("Spatial Key Performance Indicators", self.styles['Heading3']))
        
        kpi_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Mean Remaining PAR", f"{sp_kpis.get('mean_rem', 0):.1f} %", "Shadow Frequency", f"{sp_kpis.get('mean_shadow_freq', 0):.1f} %"],
            ["Median Remaining PAR", f"{sp_kpis.get('median_rem', 0):.1f} %", "Heterogeneity (CV)", f"{sp_kpis.get('cv_rem', 0)*100:.1f} %"],
            ["Area < 50% PAR", f"{sp_kpis.get('below_50_pct', 0):.1f} %", "Area < 60% PAR", f"{sp_kpis.get('below_60_pct', 0):.1f} %"]
        ]
        t = Table(kpi_data, colWidths=[130, 80, 130, 80])
        t.setStyle(get_table_style_kpi())
        self.story.append(t)
        
        # Automatic Interpretation
        self.story.append(Spacer(1, 10))
        interp = "<b>Interpretation:</b> "
        if sp_kpis.get('mean_rem', 100) < 50:
            interp += "The site is strongly shaded (Mean PAR < 50%). "
        else:
            interp += "The site has moderate to high light availability. "
            
        if sp_kpis.get('cv_rem', 0) > 0.20:
            interp += "Light distribution is highly heterogeneous, typical of fixed-tilt Agri-PV. "
        else:
            interp += "Light distribution is relatively uniform. "
            
        if sp_kpis.get('below_50_pct', 0) > 30:
            interp += "Warning: More than 30% of the area receives less than half of the open-field PAR, posing a compatibility risk for light-hungry crops."
            
        self.story.append(Paragraph(interp, self.styles['Disclaimer']))
        self.story.append(PageBreak())
        
        # PAGE B: PAR Loss and Shadow Frequency
        self.story.append(Paragraph("Shading Impact and Shadow Frequency", self.styles['Heading2']))
        if 'par_loss' in sp_figs:
            img = export_plotly_to_image(sp_figs['par_loss'], width=600, height=350)
            if img: self.story.append(img)
            
        self.story.append(Spacer(1, 10))
        if 'shadow_freq' in sp_figs:
            img = export_plotly_to_image(sp_figs['shadow_freq'], width=600, height=350)
            if img: self.story.append(img)
            
        self.story.append(PageBreak())
        
        # PAGE C: Seasonal Heatmaps
        self.story.append(Paragraph("Seasonal PAR Availability", self.styles['Heading2']))
        
        t_data = []
        row1 = []
        if 'spring' in sp_figs:
            img = export_plotly_to_image(sp_figs['spring'], width=300, height=250)
            row1.append(img if img else "")
        if 'summer' in sp_figs:
            img = export_plotly_to_image(sp_figs['summer'], width=300, height=250)
            row1.append(img if img else "")
            
        row2 = []
        if 'autumn' in sp_figs:
            img = export_plotly_to_image(sp_figs['autumn'], width=300, height=250)
            row2.append(img if img else "")
        if 'winter' in sp_figs:
            img = export_plotly_to_image(sp_figs['winter'], width=300, height=250)
            row2.append(img if img else "")
            
        if row1: t_data.append(row1)
        if row2: t_data.append(row2)
        
        if t_data:
            t = Table(t_data, colWidths=[250, 250])
            self.story.append(t)
            
        self.story.append(PageBreak())

"""
    if def_target in content and "def create_page_7_b_spatial_heatmaps" not in content:
        content = content.replace(def_target, def_injection + "\n    " + def_target)
        
    with open(r'report\report_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
if __name__ == '__main__':
    inject_report()
