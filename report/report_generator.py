import os
import io
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import geometry
from geometry import TableGeometry, GEOMETRY_PRESETS
import irradiance
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak, Image
from reportlab.lib.units import mm

from .report_styles import get_report_styles, get_table_style_standard, get_table_style_kpi
from .report_charts import export_plotly_to_image, render_latex_to_image, create_dli_chart_img, create_horizontal_bar_chart_img

# Attempt to load a logo, or we will use a text placeholder
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logo.png")

class ReportGenerator:
    def __init__(self, config, metrics, crop_results, figures):
        self.config = config
        self.metrics = metrics
        self.crop_results = crop_results
        self.figures = figures
        self.styles = get_report_styles()
        self.story = []
        
    def _header_footer(self, canvas, doc):
        """Draws the header and footer on each page."""
        canvas.saveState()
        
        # Footer
        footer_text = f"Project: {self.config.get('project_name', 'Agri-PV Site')} | {datetime.now().strftime('%d %B %Y')}"
        p_footer = Paragraph(footer_text, self.styles['Footer'])
        w, h = p_footer.wrap(doc.width, doc.bottomMargin)
        p_footer.drawOn(canvas, doc.leftMargin, 15 * mm)
        
        page_num = Paragraph(f"Page {doc.page}", self.styles['FooterRight'])
        w2, h2 = page_num.wrap(doc.width, doc.bottomMargin)
        page_num.drawOn(canvas, doc.leftMargin, 15 * mm)
        
        canvas.setStrokeColorRGB(0.8, 0.8, 0.8)
        canvas.line(doc.leftMargin, 20 * mm, doc.leftMargin + doc.width, 20 * mm)
        
        canvas.restoreState()

    def generate(self):
        """Assembles the 15 pages and returns a BytesIO object containing the PDF."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=25*mm
        )
        
        self.create_page_1_executive_summary()
        self.create_page_2_configuration()
        self.create_page_3_site_weather()
        self.create_page_4_method_solar()
        self.create_page_5_method_irradiance()
        self.create_page_6_method_shading()
        self.create_page_7_light_results()
        self.create_page_7_b_spatial_heatmaps()
        self.create_page_8_method_electrical()
        self.create_page_9_electrical_results()
        self.create_page_10_method_crop()
        self.create_page_11_crop_results()
        self.create_page_meadow_species()
        self.create_page_14_assumptions()
        self.create_page_15_appendix()
        
        doc.build(self.story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        buffer.seek(0)
        return buffer

    def create_page_1_executive_summary(self):
        # Logo handling
        if os.path.exists(LOGO_PATH):
            img = Image(LOGO_PATH, width=50*mm, height=20*mm, kind='proportional')
            img.hAlign = 'RIGHT'
            self.story.append(img)
            self.story.append(Spacer(1, 10))
            
        self.story.append(Paragraph("Agri-PV Technical Simulation Report", self.styles['Title']))
        
        # Meta table
        meta_data = [
            ["Project Name", self.config.get('project_name', 'Default Site')],
            ["Scenario", "Baseline High-Clearance System"],
            ["Coordinates", f"Lat: {self.config.get('lat', 48.0):.4f}, Lon: {self.config.get('lon', 11.0):.4f}"],
            ["Simulation Date", datetime.now().strftime('%Y-%m-%d %H:%M')],
            ["Engine Version", "v9.0 (Hottel crossed-strings, Faiman thermal)"]
        ]
        t = Table(meta_data, colWidths=[120, 330])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Executive Summary", self.styles['Heading1']))
        self.story.append(Paragraph("This document provides a rigorous technical validation and performance estimation of the proposed Agrivoltaic (Agri-PV) installation. It evaluates the physical irradiance distribution (spatial shading), electrical generation potential, and the agronomic compatibility of the selected arable and specialty crops.", self.styles['Normal']))
        self.story.append(Spacer(1, 15))
        
        # KPIs
        self.story.append(Paragraph("Key Performance Indicators (KPIs)", self.styles['Heading2']))
        
        rem_par = (self.metrics['pa'] / self.metrics['par_open_field']) * 100
        kpi_data = [
            ["Remaining PAR", f"{rem_par:.1f} %", "Specific Energy Yield", f"{self.metrics['ya_spec']:.0f} kWh/kWp"],
            ["PAR Reduction", f"{100-rem_par:.1f} %", "Yield Bonus (Thermal)", f"+{self.metrics['temp_bonus_pct']:.2f} %"],
            ["Light Deviation (cv)", f"{self.metrics['cv_par']*100:.1f} %", "Mean Cell Temp (Agri)", f"{self.metrics['ta_cell']:.1f} °C"],
        ]
        t_kpi = Table(kpi_data, colWidths=[120, 100, 140, 90])
        t_kpi.setStyle(get_table_style_kpi())
        self.story.append(t_kpi)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Interpretation", self.styles['Heading3']))
        interp = f"The system allows {rem_par:.1f}% of natural light to reach the crop level. The spatial light distribution shows a deviation of {self.metrics['cv_par']*100:.1f}%, indicating a {'homogeneous' if self.metrics['cv_par'] < 0.2 else 'heterogeneous'} environment. The electrical specific yield reaches {self.metrics['ya_spec']:.0f} kWh/kWp, profiting from a temperature-induced module efficiency bonus."
        self.story.append(Paragraph(interp, self.styles['Normal']))
        self.story.append(PageBreak())

    def create_page_2_configuration(self):
        self.story.append(Paragraph("PV System Configuration (PV-Tischgeometrie nach Schnitt)", self.styles['Heading1']))
        self.story.append(Paragraph("Physical layout, cross-section dimensions, and geometry parameters of the Agri-PV system.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        geo_dict = self.config.get('geometry', {})
        geo = TableGeometry.from_dict(geo_dict)
        
        data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Geometry Mode", f"{geo.geometry_mode}", "Clearance Height (LH)", f"{geo.clear_height_m:.2f} m"],
            ["Source / Reference", f"{geo.source_label}", "Top Edge Height (H_high)", f"{geo.h_high_m:.2f} m"],
            ["Tilt Angle", f"{geo.tilt_deg:.1f}°", "Row Pitch", f"{geo.row_pitch_m:.2f} m"],
            ["Surface Azimuth", f"{geo.surface_azimuth_deg:.1f}°", "Free Ground Gap", f"{geo.ground_gap_m:.2f} m"],
            ["Table Inclined Length", f"{geo.table_length_m:.2f} m", "Ground Coverage (GCR)", f"{geo.ground_coverage_ratio*100:.1f} %"],
            ["Projected Width", f"{geo.table_projected_width_m:.2f} m", "Structural Light Loss", f"{geo.structural_loss_percent:.1f} %"],
            ["Table Vertical Rise", f"{geo.table_vertical_rise_m:.2f} m", "Module Transparency (τ)", f"{self.config.get('tau', 0.20)*100:.1f} %"],
        ]
        t = Table(data, colWidths=[120, 110, 110, 110])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 12))
        
        if geo.geometry_mode == "Predefined Table 12°":
            self.story.append(Paragraph("<b>Reference Note (Drawing 12°):</b> Geometrie gemäß SUNfarming Schnitt Agri-PV 12°: LH 2,70 m, Tischlänge ca. 5,75 m, horizontale Projektion ca. 5,62 m, Pitch ca. 8,28 m, freier Gap ca. 2,63 m.", self.styles['NormalGray']))
            self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("Spatial Light Distribution (Cross-Section)", self.styles['Heading2']))
        if 'layout' in self.figures and self.figures['layout']:
            img = export_plotly_to_image(self.figures['layout'], width=800, height=350)
            if img: self.story.append(img)
        else:
            self.story.append(Paragraph("A cross-section representation of the selected row-to-row pitch and module orientation is depicted above.", self.styles['NormalGray']))
        
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph(f"Interpretation: The highest shading intensity occurs beneath the modules, while the inter-row areas receive peaks closer to open-field irradiance. The standard deviation of {self.metrics['cv_par']*100:.1f}% maps this variance across the ground.", self.styles['Normal']))
        
        self.story.append(PageBreak())

    def create_page_3_site_weather(self):
        self.story.append(Paragraph("Site and Weather Data Basis", self.styles['Heading1']))
        self.story.append(Paragraph("The meteorological data driving the solar physics engine.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        data = [
            ["Source", "PVGIS TMY (Typical Meteorological Year) / ERA5"],
            ["Resolution", "Hourly (8760 steps)"],
            ["Global Horizontal Irradiance", f"{self.metrics.get('vo', 1200):.0f} kWh/m²/a (Open Field)"],
            ["Albedo", "0.22 (Standard agricultural soil/grass)"],
            ["Data Handling", "Missing values interpolated linearly"]
        ]
        t = Table(data, colWidths=[180, 270])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Monthly Meteorological Averages", self.styles['Heading2']))
        if 'weather' in self.figures and self.figures['weather']:
            img = export_plotly_to_image(self.figures['weather'], width=800, height=350)
            if img: self.story.append(img)
        else:
            self.story.append(Paragraph("Weather component breakdown omitted from this view.", self.styles['NormalGray']))
            
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph("Note: All weather and irradiation values used in the report are based on the actual app data source via the API.", self.styles['Disclaimer']))
        self.story.append(PageBreak())

    def create_page_4_method_solar(self):
        self.story.append(Paragraph("Calculation Methodology: Solar Geometry", self.styles['Heading1']))
        self.story.append(Paragraph("The precise physical path of the sun is modelled for the specified latitude and longitude across the entire year.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("Solar Declination (δ)", self.styles['Heading2']))
        self.story.append(render_latex_to_image(r'\delta = 23.45^\circ \cdot \sin\left(360^\circ \cdot \frac{284 + n}{365}\right)'))
        self.story.append(Spacer(1, 15))
        
        self.story.append(Paragraph("Hour Angle (ω)", self.styles['Heading2']))
        self.story.append(render_latex_to_image(r'\omega = 15^\circ \cdot (t_{solar} - 12)'))
        self.story.append(Spacer(1, 15))
        
        self.story.append(Paragraph("Solar Elevation (α) and Zenith (θz)", self.styles['Heading2']))
        self.story.append(render_latex_to_image(r'\sin(\alpha) = \sin(\phi)\sin(\delta) + \cos(\phi)\cos(\delta)\cos(\omega)'))
        self.story.append(Spacer(1, 5))
        self.story.append(render_latex_to_image(r'\theta_z = 90^\circ - \alpha'))
        self.story.append(Spacer(1, 15))
        
        self.story.append(Paragraph("Incidence Angle on PV Plane (θi)", self.styles['Heading2']))
        self.story.append(render_latex_to_image(r'\cos(\theta_i) = \sin(\delta)\sin(\phi)\cos(\beta) - \sin(\delta)\cos(\phi)\sin(\beta)\cos(\gamma) + \dots'))
        self.story.append(Spacer(1, 20))
        self.story.append(PageBreak())

    def create_page_5_method_irradiance(self):
        # Calculate dynamic values
        geo_a_calc = geometry.calculate_derived_geometry(self.config.get('tilt', 15), length=self.config.get('module_length', 5.63), clearance=self.config.get('height', 2.10))
        geo_s_calc = geometry.calculate_derived_geometry(self.config.get('tilt', 15), length=self.config.get('module_length', 5.63), clearance=0.80)
        pw_a = geo_a_calc['projected_width']
        pw_s = geo_s_calc['projected_width']
        h_top_a = geo_a_calc['top_edge_height']
        h_top_s = geo_s_calc['top_edge_height']
        block_val = 0.81
        tau_val = self.config.get('tau', 0.05)
        pitch = self.config.get('pitch', 8.0)
        
        tau_eff_a = max(0, (pw_a - block_val) / pw_a) * tau_val
        tau_eff_s = max(0, (pw_s - block_val) / pw_s) * tau_val
        
        svf_a = irradiance.sky_view_factor_periodic(h_top_a, pw_a, pitch, tau_eff_a, h_clearance=self.config.get('height', 2.10))
        svf_s = irradiance.sky_view_factor_periodic(h_top_s, pw_s, pitch, tau_eff_s, h_clearance=0.80)

        self.story.append(Paragraph("2. Beam Transmission (Direct Light Interception)", self.styles['Heading1']))
        self.story.append(render_latex_to_image(r'\tau_{eff} = \left( \frac{w_{proj} - w_{block}}{w_{proj}} \right) \cdot \tau'))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph(f"• <b>Agri-PV ({self.config.get('height', 2.10):.2f}m):</b> ({pw_a:.3f} - {block_val}) / {pw_a:.3f} * {tau_val:.2f} = <b>{tau_eff_a:.4f}</b> ({tau_eff_a*100:.1f}%)", self.styles['Normal']))
        self.story.append(Paragraph(f"• <b>Standard PV (0.80m):</b> ({pw_s:.3f} - {block_val}) / {pw_s:.3f} * {tau_val:.2f} = <b>{tau_eff_s:.4f}</b> ({tau_eff_s*100:.1f}%)", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        self.story.append(render_latex_to_image(r'T_{beam} = 1 - \min\left(1, \frac{L \cdot \max(0, \cos(AOI_{mod}))}{P \cdot \cos(AOI_{ground})}\right) \cdot (1 - \tau_{eff})'))
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("3. Diffuse Light — Analytical Periodic Sky View Factor", self.styles['Heading1']))
        self.story.append(render_latex_to_image(r'SVF = \frac{1}{P} \int_{0}^{P} \left[ 1 - \frac{\arctan(H/x) + \arctan(H/(P - x))}{\pi} \cdot (1 - \tau_{eff}) \right] dx'))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph(f"• <b>Agri-PV (H={h_top_a:.2f}m):</b> SVF = <b>{svf_a:.4f}</b> ({svf_a*100:.1f}%)", self.styles['Normal']))
        self.story.append(Paragraph(f"• <b>Standard PV (H={h_top_s:.2f}m):</b> SVF = <b>{svf_s:.4f}</b> ({svf_s*100:.1f}%)", self.styles['Normal']))
        self.story.append(Paragraph("• <b>Scientific Significance:</b> High clearance height allows stray diffuse light to enter from adjacent row gaps. No arbitrary constants — height acts directly from Hottel integration.", self.styles['Normal']))
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("4. Total Ground Irradiance Formulation", self.styles['Heading1']))
        self.story.append(render_latex_to_image(r'G_{ground} = G_{beam} + G_{diffuse} + G_{reflected}'))
        self.story.append(Spacer(1, 5))
        self.story.append(render_latex_to_image(r'G_{beam} = DNI \cdot \cos(AOI_{ground}) \cdot T_{beam}'))
        self.story.append(Spacer(1, 5))
        self.story.append(render_latex_to_image(r'G_{diffuse} = DHI \cdot SVF \cdot \frac{1 + \cos(\beta)}{2}'))
        self.story.append(PageBreak())

    def create_page_6_method_shading(self):
        self.story.append(Paragraph("Calculation Methodology: Shading & PAR", self.styles['Heading1']))
        self.story.append(Paragraph("Calculating the Photosynthetically Active Radiation (PAR) reaching the crop level involves integrating geometric view factors and direct beam blockage.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("Cell-Level Remaining PAR", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'PAR_{rem} = PAR_{ref} \cdot (f_{shade}) \cdot \tau \cdot L_{struct}'))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("Spatial Homogeneity", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'PAR_{dev\%} = \sigma(PAR_{PV} / PAR_{ref}) \cdot 100'))
        self.story.append(Spacer(1, 20))

        self.story.append(Paragraph("Calculation Methodology: Electrical & Thermal", self.styles['Heading1']))
        self.story.append(Paragraph("DC Power Conversion", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'P_{DC,raw} = POA_{total} \cdot A_{mod} \cdot \eta_{mod}'))
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph("Thermal Derating (Faiman Model)", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'T_{cell} = T_{amb} + \frac{POA}{U_0 + U_1 \cdot v_{wind}}'))
        self.story.append(Spacer(1, 5))
        self.story.append(render_latex_to_image(r'P_{DC} = P_{DC,raw} \cdot [1 + \gamma \cdot (T_{cell} - 25)]'))
        self.story.append(Spacer(1, 10))
        self.story.append(PageBreak())

    def create_page_7_light_results(self):


        self.story.append(Paragraph("Light Simulation Results", self.styles['Heading1']))
        self.story.append(Paragraph("Spatial distribution of PAR under the Agri-PV structure.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        rem_par = (self.metrics['pa'] / self.metrics['par_open_field']) * 100
        data = [
            ["Metric", "Value"],
            ["Average Remaining PAR", f"{rem_par:.1f} %"],
            ["PAR Reduction", f"{100 - rem_par:.1f} %"],
            ["PAR Deviation (Spatial)", f"± {self.metrics['cv_par']*100:.1f} %"],
            ["Structural Blockage Loss", "Estimated 2-5% (hardware)"],
        ]
        t = Table(data, colWidths=[200, 150])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Diurnal & Seasonal Irradiance Heatmap", self.styles['Heading2']))
        if 'heat' in self.figures and self.figures['heat']:
            img = export_plotly_to_image(self.figures['heat'], width=800, height=450)
            if img: self.story.append(img)
        else:
            self.story.append(Paragraph("Spatial heatmap view omitted.", self.styles['NormalGray']))
            
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph("Interpretation: This temporal heatmap displays light availability beneath the Agri-PV array over the year. Peak irradiance occurs during summer middays, while mornings, evenings, and winter months experience systematically lower light levels, which fundamentally drives the seasonal crop suitability.", self.styles['Normal']))
        self.story.append(PageBreak())

    
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
        t = Table(kpi_data, colWidths=[140, 70, 140, 70])
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
            img = export_plotly_to_image(sp_figs['spring'], width=600, height=500, pdf_width=250, pdf_height=208)
            row1.append(img if img else "")
        if 'summer' in sp_figs:
            img = export_plotly_to_image(sp_figs['summer'], width=600, height=500, pdf_width=250, pdf_height=208)
            row1.append(img if img else "")
            
        row2 = []
        if 'autumn' in sp_figs:
            img = export_plotly_to_image(sp_figs['autumn'], width=600, height=500, pdf_width=250, pdf_height=208)
            row2.append(img if img else "")
        if 'winter' in sp_figs:
            img = export_plotly_to_image(sp_figs['winter'], width=600, height=500, pdf_width=250, pdf_height=208)
            row2.append(img if img else "")
            
        if row1: t_data.append(row1)
        if row2: t_data.append(row2)
        
        if t_data:
            t = Table(t_data, colWidths=[250, 250])
            self.story.append(t)
            
        self.story.append(PageBreak())


    def create_page_8_method_electrical(self):
        self.story.append(Paragraph("Calculation Methodology: Electrical", self.styles['Heading1']))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("DC Power Conversion", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'P_{DC,raw} = POA_{total} \cdot A_{mod} \cdot \eta_{mod}'))
        
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph("Thermal Derating (Faiman Model)", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'T_{cell} = T_{amb} + \frac{POA}{U_0 + U_1 \cdot v_{wind}}'))
        self.story.append(Spacer(1, 5))
        self.story.append(render_latex_to_image(r'P_{DC} = P_{DC,raw} \cdot [1 + \gamma \cdot (T_{cell} - 25)]'))
        
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph("System Losses and AC Clipping", self.styles['Heading3']))
        self.story.append(render_latex_to_image(r'P_{AC} = \min(P_{DC} \cdot (1 - losses_{DC}) \cdot \eta_{inv}, P_{AC,rated})'))
        
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph("The thermal model explicitly accounts for the enhanced convective wind cooling available to elevated Agri-PV structures compared to standard 0.8m ground-mount PV. Bifacial gains are dynamically evaluated based on rear-side irradiance calculations.", self.styles['Normal']))
        self.story.append(PageBreak())

    def create_page_9_electrical_results(self):
        self.story.append(Paragraph("Electrical Simulation Results", self.styles['Heading1']))
        self.story.append(Spacer(1, 10))
        
        data = [
            ["Output Parameter", "Agri-PV Value", "Standard Ground-PV"],
            ["Specific Energy Yield", f"{self.metrics['ya_spec']:.1f} kWh/kWp", f"{self.metrics['ys_spec']:.1f} kWh/kWp"],
            ["Yield Bonus (Thermal)", f"+{self.metrics['temp_bonus_pct']:.2f} %", "Baseline"],
            ["Mean Cell Temperature", f"{self.metrics['ta_cell']:.1f} °C", f"{self.metrics['ts_cell']:.1f} °C"],
        ]
        t = Table(data, colWidths=[150, 150, 150])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        if 'elec' in self.figures:
            img = export_plotly_to_image(self.figures['elec'], width=800, height=350)
            if img: self.story.append(img)
            
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph(f"Interpretation: The Agri-PV system produces a bonus of {self.metrics['y_bonus']:.1f} kWh/kWp annually compared to standard layouts, primarily due to the {self.metrics['ts_cell'] - self.metrics['ta_cell']:.1f}°C temperature reduction.", self.styles['Normal']))
        self.story.append(PageBreak())

    def create_page_10_method_crop(self):
        self.story.append(Paragraph("Calculation Methodology: Crop Response", self.styles['Heading1']))
        self.story.append(Paragraph("Evaluating agronomic suitability via physiological light models.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        self.story.append(Paragraph("Evaluation Frameworks", self.styles['Heading3']))
        self.story.append(Paragraph("1. Reproductive Yield (Korn/Fruchtbildung): Focuses on annual PAR thresholds and critical phase Daily Light Integral (DLI) minimums.<br/>2. Vegetative Biomass: Focuses on growing season PAR, P10 shadow floor thresholds, and total absorbed light efficiency.", self.styles['Normal']))
        
        self.story.append(Spacer(1, 15))
        self.story.append(Paragraph("Crops are evaluated against a registry of empirical thresholds mapped from Agri-PV field trials and ecophysiological literature (e.g. Laub et al., Fraunhofer ISE).", self.styles['Normal']))
        self.story.append(Paragraph("The crop yield result is an agronomic approximation based on light availability. It is not a substitute for a site-specific agronomic assessment.", self.styles['Disclaimer']))
        self.story.append(PageBreak())

    def create_page_11_crop_results(self):
        """Arable Crop Results: Balkendiagramm overview + individual DLI profile pages."""
        self.story.append(Paragraph("Arable & Agricultural Crops — Suitability Results", self.styles['Heading1']))
        self.story.append(Paragraph(
            "Suitability scores for all selected arable and agricultural crops under this specific Agri-PV layout. "
            "Scores are derived from a multi-component evaluation of annual PAR, seasonal PAR, critical-phase DLI and spatial homogeneity.",
            self.styles['Normal']
        ))
        self.story.append(Spacer(1, 10))

        selected_arable = self.figures.get('selected_arable', [])
        _days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        _mnames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        _m_agri = self.metrics['monthly_par_agri']
        _m_open = self.metrics['monthly_par_open']
        dli_agri_base = [_m_agri[i] / _days[i] if _days[i] > 0 else 0 for i in range(12)]
        dli_open_base = [_m_open[i] / _days[i] if _days[i] > 0 else 0 for i in range(12)]

        # --- Overview Balkendiagramm ---
        if selected_arable:
            self.story.append(Paragraph("Suitability Score Overview (Balkendiagramm)", self.styles['Heading2']))
            labels = [f"{cd['profile'].name_en} ({cd['profile'].name_de})" for cd in selected_arable]
            scores = [cd['result'].score * 100.0 for cd in selected_arable]
            img_bar = create_horizontal_bar_chart_img(labels, scores, title=f"Arable Crops Suitability Scores ({len(labels)} selected)")
            if img_bar:
                self.story.append(img_bar)
            self.story.append(Spacer(1, 10))

        # --- Summary Table ---
        self.story.append(Paragraph("Summary Table", self.styles['Heading2']))
        data = [["Crop (EN / DE)", "Class", "Score", "Limiting Factor", "Evidence"]]
        for cd in selected_arable:
            cr = cd['result']
            cp = cd['profile']
            data.append([
                Paragraph(f"{cp.name_en}<br/><i>{cp.name_de}</i>", self.styles['Normal']),
                Paragraph(cr.classification, self.styles['Normal']),
                f"{cr.score*100:.1f}%",
                cr.limiting_factor.replace('_', ' ').title(),
                f"Tier {cr.evidence_tier}"
            ])
        t = Table(data, colWidths=[120, 145, 50, 100, 45])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(PageBreak())

        # --- Individual DLI Profile Page per Selected Arable Crop ---
        for i, crop_data in enumerate(selected_arable):
            cr = crop_data['result']
            cp = crop_data['profile']

            self.story.append(Paragraph(f"{cp.name_en} ({cp.name_de})", self.styles['Heading1']))
            if cp.botanical_name:
                self.story.append(Paragraph(f"<i>{cp.botanical_name}</i>", self.styles['NormalGray']))
            self.story.append(Spacer(1, 8))

            crit_months_str = ", ".join(str(m) for m in cp.critical_months) if cp.critical_months else "—"
            profile_data = [
                ["Parameter", "Value", "Parameter", "Value"],
                ["Classification", Paragraph(cr.classification, self.styles['Normal']), "Evidence Tier", f"Tier {cr.evidence_tier}"],
                ["Crop Group", cp.crop_group.replace("_", " ").title(), "Confidence", cr.confidence],
                ["Score", f"{cr.score*100:.1f} %", "Critical Months", crit_months_str],
                ["Rel. PAR (Annual)", f"{crop_data['r_ann']:.1f} %", "Rel. PAR (Crit.)", f"{crop_data['r_crit']:.1f} %"],
                ["Mean DLI (GS)", f"{crop_data['mean_dli']:.1f} mol/m\u00b2/d", "DLI Target", f"{cp.DLI_target:.0f} mol/m\u00b2/d"],
                ["DLI Minimum", f"{cp.DLI_min:.0f} mol/m\u00b2/d", "Light Homogeneity (CV)", f"{crop_data['cv_par']:.1f} %"],
            ]
            t = Table(profile_data, colWidths=[120, 110, 120, 100])
            t.setStyle(get_table_style_standard())
            self.story.append(t)
            self.story.append(Spacer(1, 8))

            if cr.notes_de:
                self.story.append(Paragraph("<b>Evaluation:</b>", self.styles['Heading3']))
                self.story.append(Paragraph(cr.notes_de, self.styles['Normal']))
            self.story.append(Spacer(1, 8))

            self.story.append(Paragraph("Monthly DLI — Agri-PV vs. Open Field", self.styles['Heading3']))
            img = create_dli_chart_img(
                _mnames, dli_agri_base, dli_open_base,
                target_dli=cp.DLI_target, min_dli=cp.DLI_min,
                title=f"{cp.name_en} ({cp.name_de}) — Monthly DLI",
                crit_months=cp.critical_months
            )
            if img:
                self.story.append(img)

            self.story.append(PageBreak())

        # --- Medicinal & Special Crops ---
        selected_med = self.figures.get('selected_med', [])
        if selected_med:
            self.story.append(Paragraph("Medicinal & Special Crops — Suitability Results", self.styles['Heading1']))
            self.story.append(Paragraph(
                "Light availability assessment for all selected medicinal and specialty crops. "
                "Annual and critical-phase relative PAR values are compared against species-specific light thresholds.",
                self.styles['Normal']
            ))
            self.story.append(Spacer(1, 10))

            labels_med = [mr['result'].crop_name for mr in selected_med]
            scores_med = [mr['result'].r_ann * 100.0 for mr in selected_med]
            self.story.append(Paragraph("PAR Availability Overview (Balkendiagramm)", self.styles['Heading2']))
            img_med_ov = create_horizontal_bar_chart_img(labels_med, scores_med, title=f"Medicinal Crops Annual rPAR ({len(labels_med)} selected)")
            if img_med_ov:
                self.story.append(img_med_ov)
            self.story.append(Spacer(1, 10))

            data_med = [["Crop", "Botanical Name", "Annual rPAR", "Crit. rPAR", "Class", "Limiting Factor"]]
            for md in selected_med:
                mr = md['result']
                mp = md['profile']
                data_med.append([
                    Paragraph(mr.crop_name, self.styles['Normal']),
                    Paragraph(f"<i>{mp.botanical_name}</i>", self.styles['Normal']),
                    f"{mr.r_ann*100:.1f}%",
                    f"{mr.r_crit*100:.1f}%",
                    Paragraph(mr.suitability_class, self.styles['Normal']),
                    mr.limiting_factor
                ])
            t_med = Table(data_med, colWidths=[90, 120, 55, 55, 80, 60])
            t_med.setStyle(get_table_style_standard())
            self.story.append(t_med)
            self.story.append(PageBreak())

            for md in selected_med:
                mr = md['result']
                mp = md['profile']
                self.story.append(Paragraph(f"{mr.crop_name}", self.styles['Heading1']))
                self.story.append(Paragraph(f"<i>{mp.botanical_name}</i> — {mp.use_type.title()}", self.styles['NormalGray']))
                self.story.append(Spacer(1, 6))

                med_profile_data = [
                    ["Parameter", "Value", "Parameter", "Value"],
                    ["Suitability Class", Paragraph(mr.suitability_class, self.styles['Normal']), "Homogeneity", mr.homogeneity_class.title()],
                    ["Annual rPAR", f"{mr.r_ann*100:.1f}%", "Critical rPAR", f"{mr.r_crit*100:.1f}%"],
                    ["DLI Min Threshold", f"{mp.DLI_min:.1f} mol/m\u00b2/d", "Limiting Factor", mr.limiting_factor],
                ]
                t_mp = Table(med_profile_data, colWidths=[120, 110, 120, 100])
                t_mp.setStyle(get_table_style_standard())
                self.story.append(t_mp)
                self.story.append(Spacer(1, 8))

                self.story.append(Paragraph("Monthly DLI — Agri-PV vs. Open Field", self.styles['Heading3']))
                img_md = create_dli_chart_img(
                    _mnames, dli_agri_base, dli_open_base,
                    min_dli=mp.DLI_min,
                    title=f"{mp.display_name} — Monthly DLI"
                )
                if img_md:
                    self.story.append(img_md)
                self.story.append(PageBreak())

    def create_page_meadow_species(self):
        """Page: Wet Meadow & Floodplain Species suitability with DLI charts."""
        selected_meadow = self.figures.get('selected_meadow', [])
        meadow_results  = self.figures.get('meadow_results', [])
        if not meadow_results and not selected_meadow:
            return  # No meadow data, skip page

        self.story.append(Paragraph("Wet Meadow & Floodplain Species — Suitability Results", self.styles['Heading1']))
        self.story.append(Paragraph(
            "Suitability assessment for selected wet meadow, floodplain and grassland species under this Agri-PV layout. "
            "Scores are derived from Ellenberg / Landolt light indicator values (L), hydrology (F), and measured relative PAR.",
            self.styles['Normal']
        ))
        self.story.append(Spacer(1, 10))

        # Summary table
        data = [["Species", "Score", "Light Class", "Hydro Class", "Zone", "L/F", "rPAR"]]
        for r in meadow_results:
            name_para = Paragraph(f"<b>{r.display_name}</b><br/><i>{r.botanical_name}</i>", self.styles['Normal'])
            data.append([
                name_para,
                f"{r.score:.0f}",
                Paragraph(r.light_class, self.styles['Normal']),
                Paragraph(r.hydro_class, self.styles['Normal']),
                Paragraph(r.zone_hint, self.styles['Normal']),
                f"L{r.ellenberg_L}/F{r.ellenberg_F}",
                f"{r.rPAR_actual*100:.0f}%",
            ])
        t = Table(data, colWidths=[110, 35, 90, 80, 80, 35, 30])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 8))
        self.story.append(Paragraph(
            "<b>Legend:</b> L = Ellenberg Light Value, F = Ellenberg Moisture Value. "
            "rPAR = Relative PAR (Agri-PV / Open Field). Score = weighted combination of light (70%), hydrology (20%) and homogeneity (10%).",
            self.styles['Normal']
        ))
        self.story.append(PageBreak())

        # Individual DLI pages for selected meadow species
        for md in selected_meadow:
            mwr = md['result']
            mwp = md['profile']
            self.story.append(Paragraph(f"{mwp.display_name}", self.styles['Heading1']))
            self.story.append(Paragraph(f"<i>{mwp.botanical_name}</i> — {mwp.species_group.title() if hasattr(mwp, 'species_group') else ''}", self.styles['NormalGray']))
            self.story.append(Spacer(1, 6))

            mw_profile_data = [
                ["Parameter", "Value", "Parameter", "Value"],
                ["Suitability Score", f"{mwr.score:.1f} / 100", "Light Class", mwr.light_class],
                ["Hydrology Class", mwr.hydro_class, "Recommended Zone", mwr.zone_hint],
                ["Ellenberg L", str(mwr.ellenberg_L), "Ellenberg F", str(mwr.ellenberg_F)],
                ["Actual rPAR", f"{mwr.rPAR_actual*100:.1f}%", "Limiting Factor", mwr.limiting_factor],
                ["DLI Min", f"{mwp.DLI_min:.1f} mol/m\u00b2/d", "DLI Target", f"{mwp.DLI_target:.1f} mol/m\u00b2/d"],
            ]
            t_mw = Table(mw_profile_data, colWidths=[120, 110, 120, 100])
            t_mw.setStyle(get_table_style_standard())
            self.story.append(t_mw)
            self.story.append(Spacer(1, 8))

            self.story.append(Paragraph("Monthly DLI — Agri-PV vs. Open Field", self.styles['Heading3']))
            img_mw = create_dli_chart_img(
                _mnames, dli_agri_base, dli_open_base,
                target_dli=mwp.DLI_target, min_dli=mwp.DLI_min,
                title=f"{mwp.display_name} — Monthly DLI"
            )
            if img_mw:
                self.story.append(img_mw)
            self.story.append(PageBreak())


    def create_page_12_combined_evaluation(self):
        self.story.append(Paragraph("Combined Agri-PV Scenario Evaluation", self.styles['Heading1']))
        self.story.append(Spacer(1, 10))
            
        self.story.append(Paragraph("Scenario Classification:", self.styles['Heading3']))
        rem_par = (self.metrics['pa'] / self.metrics['par_open_field']) * 100
        classification = "Electricity-Optimized" if rem_par < 40 else "Balanced Agri-PV" if rem_par < 70 else "Crop-Optimized"
        self.story.append(Paragraph(f"Based on the resulting {rem_par:.1f}% light availability, this design is classified as <b>{classification}</b>.", self.styles['Normal']))
        self.story.append(PageBreak())

    def create_page_13_recommendations(self):
        self.story.append(Paragraph("Design Recommendations & Sensitivities", self.styles['Heading1']))
        self.story.append(Spacer(1, 10))
        
        rem_par = (self.metrics['pa'] / self.metrics['par_open_field']) * 100
        recs = []
        if rem_par < 50:
            recs.append("Increase row-to-row pitch to allow more light penetration if cultivating high-light demanding crops (e.g., maize, wheat).")
        if self.metrics['cv_par'] > 0.2:
            recs.append("Light distribution is highly heterogeneous. Consider raising the clearance height to diffuse shadows, or select highly shade-resilient crops for the under-panel zones.")
        recs.append("Bifacial module selection is strongly recommended due to the elevated albedo tracking of the crop canopy.")
            
        for rec in recs:
            self.story.append(Paragraph(f"• {rec}", self.styles['Bullet']))
            
        self.story.append(PageBreak())

    def create_page_14_assumptions(self):
        self.story.append(Paragraph("Assumptions, Limitations, and Boundaries", self.styles['Heading1']))
        self.story.append(Spacer(1, 10))
        
        data = [
            ["Domain", "Assumption / Limitation"],
            ["Weather", Paragraph("TMY dataset. Inter-annual variability and extreme events are not captured.", self.styles['Normal'])],
            ["Albedo", Paragraph("Fixed at 0.22. Real albedo varies dynamically with crop phenology.", self.styles['Normal'])],
            ["Shading", Paragraph("Support structures (poles) are estimated as a fixed 3% transmission loss. Row-to-row shadows are modelled in 2D cross-section.", self.styles['Normal'])],
            ["Electrical", Paragraph("Inverter clipping and detailed DC cable losses are estimated coarsely.", self.styles['Normal'])],
            ["Agronomic", Paragraph("Microclimate effects (humidity, wind shielding, soil moisture retention) are NOT simulated. Crop scores are light-based only.", self.styles['Normal'])]
        ]
        t = Table(data, colWidths=[120, 330])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Disclaimer", self.styles['Heading3']))
        self.story.append(Paragraph("The simulation results are planning-level estimates. They are intended for scenario comparison and preliminary technical evaluation. They do not replace bankable yield assessments, detailed engineering, agronomic field trials, structural design, glare assessment, environmental assessment, permitting documents or expert validation.", self.styles['Disclaimer']))
        self.story.append(PageBreak())

    def create_page_15_appendix(self):
        self.story.append(Paragraph("References and Sources", self.styles['Heading1']))
        self.story.append(Paragraph("The following scientific literature, standards, and data sources were used in the methodology and crop evaluation of this report.", self.styles['Normal']))
        self.story.append(Spacer(1, 15))
        
        refs = [
            ["ID", "Reference"],
            ["[1]", Paragraph("Laub, M. et al. (2022). <i>Contrasting yield responses at varying levels of shade suggest different suitability of crops for dual land-use systems: a meta-analysis.</i> Agronomy for Sustainable Development. DOI: 10.1007/s13593-022-00783-7", self.styles['Normal'])],
            ["[2]", Paragraph("Weselek, A. et al. (2021). <i>Agrivoltaic system impacts on microclimate and yield of different crops within an organic crop rotation in a temperate climate.</i> Agronomy for Sustainable Development. DOI: 10.1007/s13593-021-00714-y", self.styles['Normal'])],
            ["[3]", Paragraph("Edouard, S. et al. (2023). <i>Increasing land productivity with agriphotovoltaics: Application to an alfalfa field.</i> Applied Energy. DOI: 10.1016/j.apenergy.2022.120207", self.styles['Normal'])],
            ["[4]", Paragraph("Arenas-Corraliza, M. G. et al. (2019). <i>Wheat and barley can increase grain yield in shade through acclimation of physiological and morphological traits in Mediterranean conditions.</i> Scientific Reports. DOI: 10.1038/s41598-019-46027-9", self.styles['Normal'])],
            ["[5]", Paragraph("DIN SPEC 91434 (2021). <i>Agri-Photovoltaik-Anlagen — Anforderungen an die landwirtschaftliche Hauptnutzung.</i> Deutsches Institut für Normung.", self.styles['Normal'])],
            ["[6]", Paragraph("Fraunhofer ISE. <i>Agri-Photovoltaik: Chance für Landwirtschaft und Energiewende — Ein Leitfaden für Deutschland.</i> Fraunhofer Institute for Solar Energy Systems.", self.styles['Normal'])],
            ["[7]", Paragraph("Faiman, D. (2008). <i>Assessing the outdoor operating temperature of photovoltaic modules.</i> Progress in Photovoltaics. DOI: 10.1002/pip.813", self.styles['Normal'])],
            ["[8]", Paragraph("Hottel, H.C. (1954). <i>Radiant-heat transmission.</i> In: McAdams, W.H. (Ed.), Heat Transmission (3rd ed.). McGraw-Hill. (Crossed-strings method for view factors.)", self.styles['Normal'])],
            ["[9]", Paragraph("PVGIS / ERA5 (Copernicus). <i>Typical Meteorological Year (TMY) hourly datasets.</i> European Commission Joint Research Centre.", self.styles['Normal'])],
            ["[10]", Paragraph("Liu, B.Y.H.; Jordan, R.C. (1963). <i>The long-term average performance of flat-plate solar-energy collectors.</i> Solar Energy. (Isotropic sky diffuse model.)", self.styles['Normal'])],
        ]
        t = Table(refs, colWidths=[30, 420])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(PageBreak())
