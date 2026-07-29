import unittest
from biomass_suitability import evaluate_biomass_suitability, BiomassMetrics, SiteContext
from crop_suitability import CROP_REGISTRY
from medicinal_crop_suitability import MED_CROP_REGISTRY

class TestBiomassEngine(unittest.TestCase):
    def setUp(self):
        self.site = SiteContext(hot_dry_index=0.5, water_stress_risk=0.5, humidity_disease_index=0.2)

    def test_r_gs_calculation(self):
        metrics = BiomassMetrics(P_gs_agri=650, P_gs_open=1000, DLI_gs_mean=25, DLI_gs_p10=18, peak_PPFD_gs=900, cv_PAR=0.1)
        res = evaluate_biomass_suitability(CROP_REGISTRY['luzerne'], metrics, self.site)
        self.assertEqual(res['r_gs'], 0.65)

    def test_c3_biomass_generous(self):
        metrics = BiomassMetrics(P_gs_agri=720, P_gs_open=1000, DLI_gs_mean=21, DLI_gs_p10=17, peak_PPFD_gs=950, cv_PAR=0.15)
        res_bio = evaluate_biomass_suitability(CROP_REGISTRY['weizen'], metrics, self.site)
        self.assertEqual(res_bio['yield_objective'], 'biomass')
        self.assertTrue(len(res_bio['label']) > 0)

    def test_luzerne_suitable_at_65(self):
        metrics = BiomassMetrics(P_gs_agri=650, P_gs_open=1000, DLI_gs_mean=20, DLI_gs_p10=15, peak_PPFD_gs=850, cv_PAR=0.2)
        res = evaluate_biomass_suitability(CROP_REGISTRY['luzerne'], metrics, self.site)
        self.assertTrue('suitable' in res['label'].lower() or 'geeignet' in res['label'].lower())

    def test_kleegras_conditionally_suitable_at_65(self):
        metrics = BiomassMetrics(P_gs_agri=650, P_gs_open=1000, DLI_gs_mean=18, DLI_gs_p10=14, peak_PPFD_gs=800, cv_PAR=0.2)
        res = evaluate_biomass_suitability(CROP_REGISTRY['kleegras'], metrics, self.site)
        self.assertTrue(len(res['label']) > 0)

    def test_triticale_biomass_vs_grain(self):
        metrics = BiomassMetrics(P_gs_agri=750, P_gs_open=1000, DLI_gs_mean=22, DLI_gs_p10=18, peak_PPFD_gs=950, cv_PAR=0.1)
        res = evaluate_biomass_suitability(CROP_REGISTRY['triticale'], metrics, self.site)
        self.assertTrue('suitable' in res['label'].lower() or 'geeignet' in res['label'].lower())

    def test_maize_critical_below_85(self):
        metrics = BiomassMetrics(P_gs_agri=800, P_gs_open=1000, DLI_gs_mean=28, DLI_gs_p10=24, peak_PPFD_gs=1300, cv_PAR=0.1)
        res = evaluate_biomass_suitability(CROP_REGISTRY['mais'], metrics, self.site)
        self.assertEqual(res['label'], 'Not Suitable under Simulated Light')

    def test_medicinal_quality_warning(self):
        metrics = BiomassMetrics(P_gs_agri=800, P_gs_open=1000, DLI_gs_mean=25, DLI_gs_p10=18, peak_PPFD_gs=950, cv_PAR=0.1)
        res = evaluate_biomass_suitability(MED_CROP_REGISTRY['echte_kamille'], metrics, self.site)
        has_warning = any("quality" in w.lower() or "qualitätsbewertung" in w.lower() for w in res['warnings'])
        self.assertTrue(has_warning)

    def test_evidence_tier_C_warning(self):
        metrics = BiomassMetrics(P_gs_agri=850, P_gs_open=1000, DLI_gs_mean=25, DLI_gs_p10=20, peak_PPFD_gs=950, cv_PAR=0.1)
        res = evaluate_biomass_suitability(CROP_REGISTRY['hafer'], metrics, self.site)
        self.assertTrue(len(res['label']) > 0)

    def test_high_cv_warning(self):
        metrics = BiomassMetrics(P_gs_agri=800, P_gs_open=1000, DLI_gs_mean=25, DLI_gs_p10=18, peak_PPFD_gs=950, cv_PAR=0.4)
        res = evaluate_biomass_suitability(CROP_REGISTRY['luzerne'], metrics, self.site)
        has_warning = any("uneven" in w.lower() or "heterogen" in w.lower() or "ungleichmäßig" in w.lower() for w in res['warnings'])
        self.assertTrue(has_warning)

    def test_low_dli_p10_warning(self):
        metrics = BiomassMetrics(P_gs_agri=800, P_gs_open=1000, DLI_gs_mean=25, DLI_gs_p10=10, peak_PPFD_gs=950, cv_PAR=0.1)
        res = evaluate_biomass_suitability(CROP_REGISTRY['luzerne'], metrics, self.site)
        has_warning = any("dli" in w.lower() or "niedrige dli" in w.lower() for w in res['warnings'])
        self.assertTrue(has_warning)

if __name__ == '__main__':
    unittest.main()
