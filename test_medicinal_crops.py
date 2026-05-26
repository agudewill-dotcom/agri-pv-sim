import traceback
from medicinal_crop_suitability import (
    MED_CROP_REGISTRY,
    evaluate_medicinal_crop,
    evaluate_all_medicinal_crops,
    parse_months
)

def test_month_parsing():
    assert parse_months(["May", "Jun", "Jul"]) == [5, 6, 7]

def test_salbei_not_suitable_at_low_r_ann():
    salbei = MED_CROP_REGISTRY["salbei"]
    res = evaluate_medicinal_crop(
        crop=salbei,
        annual_PAR_agri=60,
        annual_PAR_openfield=100, # r_ann = 0.60
        monthly_PAR_agri=[5]*12,
        monthly_PAR_openfield=[10]*12, # r_crit = 0.50
        DLI_crit=15.0,
        peak_PPFD_crit=500.0,
        cv_PAR=0.10
    )
    assert res.r_ann == 0.60
    assert "ungeeignet" in res.suitability_class

def test_echinacea_marginal_at_065():
    echinacea = MED_CROP_REGISTRY["echinacea_purpurea"]
    res = evaluate_medicinal_crop(
        crop=echinacea,
        annual_PAR_agri=65,
        annual_PAR_openfield=100, # r_ann = 0.65
        monthly_PAR_agri=[6.5]*12,
        monthly_PAR_openfield=[10]*12, # r_crit = 0.65
        DLI_crit=20.0,
        peak_PPFD_crit=600.0,
        cv_PAR=0.10
    )
    assert "ungeeignet" in res.suitability_class  # Since r_ann < 0.75

def test_kamille_suitable_if_dli_sufficient():
    kamille = MED_CROP_REGISTRY["echte_kamille"]
    res = evaluate_medicinal_crop(
        crop=kamille,
        annual_PAR_agri=85,
        annual_PAR_openfield=100,
        monthly_PAR_agri=[8.5]*12,
        monthly_PAR_openfield=[10]*12,
        DLI_crit=28.0,
        peak_PPFD_crit=700.0,
        cv_PAR=0.10
    )
    assert res.r_ann == 0.85
    assert res.suitability_class == "geeignet als Sonder-Hauptackerfrucht mit agronomischer Prüfung"
    assert "C" in kamille.evidence_tier

def test_evidence_tier_c_never_sicher_geeignet():
    for cid, crop in MED_CROP_REGISTRY.items():
        if "C" in crop.evidence_tier:
            res = evaluate_medicinal_crop(
                crop=crop,
                annual_PAR_agri=100,
                annual_PAR_openfield=100,
                monthly_PAR_agri=[10]*12,
                monthly_PAR_openfield=[10]*12,
                DLI_crit=50.0,
                peak_PPFD_crit=1000.0,
                cv_PAR=0.05
            )
            assert "sicher geeignet" not in res.suitability_class
            if cid == "kapuzinerkresse":
                assert "Feldversuch" in res.suitability_class or "Prüfung" in res.suitability_class

def test_kapuzinerkresse_warning():
    kap = MED_CROP_REGISTRY["kapuzinerkresse"]
    res = evaluate_medicinal_crop(
        crop=kap,
        annual_PAR_agri=85,
        annual_PAR_openfield=100,
        monthly_PAR_agri=[8.5]*12,
        monthly_PAR_openfield=[10]*12,
        DLI_crit=28.0,
        peak_PPFD_crit=700.0,
        cv_PAR=0.10
    )
    assert "Feldversuch" in res.suitability_class
    assert "Nicht als klassische deutsche Hauptackerfrucht validiert" in res.warning_text

def test_homogeneity_penalty():
    kamille = MED_CROP_REGISTRY["echte_kamille"]
    res = evaluate_medicinal_crop(
        crop=kamille,
        annual_PAR_agri=85,
        annual_PAR_openfield=100,
        monthly_PAR_agri=[8.5]*12,
        monthly_PAR_openfield=[10]*12,
        DLI_crit=28.0,
        peak_PPFD_crit=700.0,
        cv_PAR=0.30  # > 0.25
    )
    # Original would be geeignet -> downgrade to grenzwertig
    assert "grenzwertig" in res.suitability_class
    assert res.limiting_factor == "zu heterogene Lichtverteilung"
    assert res.homogeneity_class == "kritisch"

def test_peak_ppfd_penalty():
    kamille = MED_CROP_REGISTRY["echte_kamille"]
    res = evaluate_medicinal_crop(
        crop=kamille,
        annual_PAR_agri=85,
        annual_PAR_openfield=100,
        monthly_PAR_agri=[8.5]*12,
        monthly_PAR_openfield=[10]*12,
        DLI_crit=28.0,
        peak_PPFD_crit=500.0, # min is 600
        cv_PAR=0.10
    )
    assert "grenzwertig" in res.suitability_class
    assert res.limiting_factor == "zu niedrige Spitzen-PAR / PPFD in der kritischen Phase"

if __name__ == "__main__":
    tests = [
        test_month_parsing,
        test_salbei_not_suitable_at_low_r_ann,
        test_echinacea_marginal_at_065,
        test_kamille_suitable_if_dli_sufficient,
        test_evidence_tier_c_never_sicher_geeignet,
        test_kapuzinerkresse_warning,
        test_homogeneity_penalty,
        test_peak_ppfd_penalty
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
        except Exception as e:
            print(f"ERROR: {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed != len(tests):
        exit(1)
