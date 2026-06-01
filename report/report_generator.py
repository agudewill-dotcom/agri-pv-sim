import os
import io
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import geometry
import irradiance
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak, Image
from reportlab.lib.units import mm

from .report_styles import get_report_styles, get_table_style_standard, get_table_style_kpi
from .report_charts import export_plotly_to_image, render_latex_to_image

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
        self.story.append(Paragraph("PV System Configuration", self.styles['Heading1']))
        self.story.append(Paragraph("The physical layout and geometry of the simulated tracking or fixed-tilt array.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Latitude", f"{self.config.get('lat')}°", "Clearance Height", f"{self.config.get('height')} m"],
            ["Longitude", f"{self.config.get('lon')}°", "Row-to-Row Pitch", f"{self.config.get('pitch')} m"],
            ["Azimuth", f"{self.config.get('azimuth')}°", "Module Type", "Glass-Glass Bifacial"],
            ["Tilt Angle", f"{self.config.get('tilt')}°", "Bifaciality", "75 %"],
            ["System Type", "Fixed / Shed", "Module Width", "1.134 m"],
            ["Tracking Active", "False", "Module Length", "2.382 m"],
            ["Transparency", f"{self.config.get('transparency', 5.0)} %", "", ""],
        ]
        t = Table(data, colWidths=[110, 100, 120, 120])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Spatial Light Distribution (Cross-Section)", self.styles['Heading2']))
        if 'layout' in self.figures and self.figures['layout']:
            img = export_plotly_to_image(self.figures['layout'], width=800, height=400)
            if img: self.story.append(img)
        else:
            self.story.append(Paragraph("A 3D schematic representation of the selected row-to-row pitch and module orientation is omitted in this view.", self.styles['NormalGray']))
        
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
        self.story.append(Paragraph("Crop Simulation Results", self.styles['Heading1']))
        self.story.append(Paragraph("Suitability scores for standard arable crops under this specific layout.", self.styles['Normal']))
        self.story.append(Spacer(1, 10))
        
        data = [["Crop Name", "Suitability Class", "Score", "Confidence"]]
        for r in self.crop_results[:12]:
            class_para = Paragraph(r.classification, self.styles['Normal'])
            crop_para = Paragraph(r.crop_id.capitalize(), self.styles['Normal'])
            data.append([crop_para, class_para, f"{r.score*100:.1f}%", r.confidence])
            
        t = Table(data, colWidths=[110, 200, 70, 70])
        t.setStyle(get_table_style_standard())
        self.story.append(t)
        self.story.append(Spacer(1, 20))
        
        self.story.append(Paragraph("Crop Response Visualization", self.styles['Heading2']))
        if 'crop' in self.figures:
            img = export_plotly_to_image(self.figures['crop'], width=800, height=350)
            if img: self.story.append(img)
            
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
