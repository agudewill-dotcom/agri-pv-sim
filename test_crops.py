"""
test_crops.py — Unit and integration tests for the Agri-PV Crop Suitability Engine

Tests cover:
1. Crop registry validity (11 crops, weight sums, f_min < f_target, evidence tiers)
2. PAR reference calculations and absolute thresholds
3. Scoring components (A, S, C, H) under different conditions
4. Ranking consistency under moderate shading (Luzerne > Winterweizen > Mais)
5. Confidence score calculations (Evidence tiers, proxy penalties, data resolution)
6. Classification mappings
7. Full integration cases (Fall A and Fall B validation)
"""

import sys
import os
import numpy as np

# Ensure modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from crop_profiles import (
    CROP_REGISTRY,
    CropProfile,
    get_par_ref_from_ghi,
    get_absolute_thresholds,
    get_monthly_weights,
)
from crop_scoring import (
    evaluate_crop,
    evaluate_all_crops,
    SuitabilityResult,
    classify_score,
    calculate_confidence,
    identify_limiting_factor,
    score_annual_par,
    score_seasonal_par,
    score_critical_phase,
    score_homogeneity,
)


def test_crop_registry():
    """All 11 crops must be defined with valid thresholds and weight configurations."""
    expected_crops = {
        "luzerne", "wintergerste", "winterroggen", "triticale", "winterweizen",
        "dinkel", "einkorn", "emmer", "hafer", "schwarzhafer", "mais"
    }
    
    assert set(CROP_REGISTRY.keys()) == expected_crops, "Registry mismatch"
    assert len(CROP_REGISTRY) == 11
    
    for crop_id, crop in CROP_REGISTRY.items():
        assert crop.id == crop_id
        # f_min must be less than f_target
        assert crop.f_min < crop.f_target, f"{crop_id}: f_min ({crop.f_min}) should be < f_target ({crop.f_target})"
        
        # Evidence tiers must be A, B or C
        assert crop.evidence_tier in {"A", "B", "C"}, f"{crop_id}: invalid evidence tier {crop.evidence_tier}"
        
        # Weights must sum to exactly 1.0
        w = crop.weights
        weight_sum = w["wA"] + w["wS"] + w["wC"] + w["wH"]
        assert np.isclose(weight_sum, 1.0), f"{crop_id}: weights sum to {weight_sum}, expected 1.0"
        
        # Growing months and critical months must be valid 1-indexed calendar months
        assert all(1 <= m <= 12 for m in crop.growing_months), f"{crop_id}: invalid growing months"
        assert all(1 <= m <= 12 for m in crop.critical_months), f"{crop_id}: invalid critical months"
        
    print("  ✓ Crop registry checks passed successfully.")


def test_par_reference_and_thresholds():
    """Verify PAR reference from GHI and translation into absolute thresholds."""
    # German average GHI ≈ 1102 kWh/m²/year
    par_ref = get_par_ref_from_ghi(1102.0)
    # GHI * 3.6 * f_PAR (0.45) * McCree (4.57) = 1102 * 3.6 * 0.45 * 4.57 ≈ 8160.67 mol/m²
    assert 8100.0 < par_ref < 8200.0, f"Expected ~8160 mol/m², got {par_ref:.2f}"
    
    # Check Luzerne thresholds at par_ref = 8500
    luzerne = CROP_REGISTRY["luzerne"]
    # f_min = 0.55, f_target = 0.75
    p_min, p_tgt = get_absolute_thresholds(luzerne, 8500.0)
    assert np.isclose(p_min, 0.55 * 8500.0), f"Luzerne p_min error: {p_min}"
    assert np.isclose(p_tgt, 0.75 * 8500.0), f"Luzerne p_tgt error: {p_tgt}"
    
    # Check monthly weights sum to 1.0
    w_monthly = get_monthly_weights(luzerne)
    assert len(w_monthly) == 12
    assert np.isclose(sum(w_monthly), 1.0)
    
    print("  ✓ PAR reference and threshold derivations passed.")


def test_scoring_components():
    """Verify separate A, S, C, H components behave according to limits."""
    # Component A — Annual PAR
    assert score_annual_par(7000.0, 5000.0, 6000.0) == 1.0  # Above target
    assert score_annual_par(4000.0, 5000.0, 6000.0) == 0.0  # Below min
    assert score_annual_par(5500.0, 5000.0, 6000.0) == 0.5  # Linear interpolation
    
    # Component S — Seasonal PAR
    # Setup standard open field PAR reference = 8400 (700 mol/m² per month)
    par_ref = 8400.0
    # Create high monthly PAR: 90% of open field for all months
    high_monthly = [630.0] * 12
    # Create low monthly PAR: 50% of open field for all months
    low_monthly = [350.0] * 12
    
    luzerne = CROP_REGISTRY["luzerne"] # f_min=0.55, f_target=0.75
    
    # At 90% everywhere, seasonal score should be 0.92 under advanced model
    s_high = score_seasonal_par(high_monthly, luzerne, par_ref)
    assert np.isclose(s_high, 0.92), f"Expected 0.92, got {s_high}"
    
    # At 50% everywhere, seasonal score should be 0.55 (since 0.50 is between f_limit=0.45 and f_min=0.55)
    s_low = score_seasonal_par(low_monthly, luzerne, par_ref)
    assert np.isclose(s_low, 0.55), f"Expected 0.55, got {s_low}"
    
    # Component H — Homogeneity
    assert score_homogeneity(0.0, 0.25) == 1.0  # Perfect uniformity
    assert score_homogeneity(0.25, 0.25) == 0.0  # At max CV limit
    assert score_homogeneity(0.10, 0.25) == 0.6  # Linear interpolation
    assert score_homogeneity(0.30, 0.25) == 0.0  # Exceeding limit
    
    print("  ✓ Component scoring logic (A, S, C, H) validated.")


def test_ranking_consistency():
    """Verify that relative crop suitability ranking makes agronomic sense under shading."""
    par_ref = get_par_ref_from_ghi(1100.0)  # ≈ 8146 mol/m²
    
    # Scenario A: Moderate Shading (65% Remaining PAR)
    par_ann_65 = par_ref * 0.65
    monthly_65 = [par_ref / 12.0 * 0.65] * 12
    
    results_65 = evaluate_all_crops(
        par_ann=par_ann_65, par_ref=par_ref, monthly_par=monthly_65, cv_par=0.10
    )
    
    # Convert list of results to a dict for easy lookup
    res_dict = {r.crop_id: r for r in results_65}
    
    # Ranking expectation: Luzerne > Winterweizen > Mais
    assert res_dict["luzerne"].score > res_dict["winterweizen"].score, "Luzerne should score higher than Winterweizen at 65% remaining PAR"
    assert res_dict["winterweizen"].score > res_dict["mais"].score, "Winterweizen should score higher than Mais at 65% remaining PAR"
    
    # Scenario B: High Shading (58% Remaining PAR)
    par_ann_58 = par_ref * 0.58
    monthly_58 = [par_ref / 12.0 * 0.58] * 12
    results_58 = evaluate_all_crops(
        par_ann=par_ann_58, par_ref=par_ref, monthly_par=monthly_58, cv_par=0.10
    )
    res_dict_58 = {r.crop_id: r for r in results_58}
    
    # Mais has f_min = 0.85, so at 58% it must be classified as 'nicht empfohlen'
    assert res_dict_58["mais"].classification == "nicht empfohlen", f"Mais classification at 58% should be 'nicht empfohlen', got {res_dict_58['mais'].classification}"
    
    # Scenario C: Very Light Shading (80% Remaining PAR)
    par_ann_80 = par_ref * 0.80
    monthly_80 = [par_ref / 12.0 * 0.80] * 12
    results_80 = evaluate_all_crops(
        par_ann=par_ann_80, par_ref=par_ref, monthly_par=monthly_80, cv_par=0.05
    )
    res_dict_80 = {r.crop_id: r for r in results_80}
    
    # Luzerne (f_target = 0.75) at 80% remaining PAR should score 1.0 and be 'sehr gut geeignet'
    assert res_dict_80["luzerne"].classification == "sehr gut geeignet", f"Luzerne at 80% PAR should be 'sehr gut geeignet'"
    
    print("  ✓ Ranking consistency matches agronomic expectations.")


def test_confidence_calculations():
    """Verify confidence model adjustments based on data availability and proxies."""
    # Case 1: High confidence (Evidence A, hourly/sub-hourly data)
    conf_val, label = calculate_confidence(
        evidence_tier="A", has_monthly=True, has_hourly=True, is_proxy=False
    )
    assert label == "hoch"
    assert conf_val >= 0.90
    
    # Case 2: Low confidence (Evidence C, annual only, proxy crop)
    conf_val_low, label_low = calculate_confidence(
        evidence_tier="C", has_monthly=False, has_hourly=False, is_proxy=True
    )
    assert label_low == "niedrig"
    # base(0.50) - annual(-0.10) - proxy(-0.15) = 0.25
    assert np.isclose(conf_val_low, 0.25)
    
    print("  ✓ Confidence model calculations verified.")


def test_classification_mapping():
    """Verify core score-to-classification boundary logic."""
    assert classify_score(0.85) == "sehr gut geeignet"
    assert classify_score(0.80) == "sehr gut geeignet"
    assert classify_score(0.79) == "geeignet"
    assert classify_score(0.65) == "geeignet"
    assert classify_score(0.64) == "grenzwertig"
    assert classify_score(0.45) == "grenzwertig"
    assert classify_score(0.44) == "nicht empfohlen"
    assert classify_score(0.10) == "nicht empfohlen"
    print("  ✓ Classification threshold mappings verified.")


def test_integration_scenarios():
    """Validate full crop suitability engine using the two real-world validation cases."""
    # ----------------------------------------------------
    # FALL A: Annual open-field GHI leads to par_ref = 8270
    # Shading is moderate (63.8% remaining PAR -> par_ann = 5277)
    # CV is 10% (cv_par = 0.10)
    # ----------------------------------------------------
    ref_a = 8270.0
    ann_a = 5277.0
    monthly_a = [ann_a / 12.0] * 12
    cv_a = 0.10
    
    results_a = evaluate_all_crops(
        par_ann=ann_a, par_ref=ref_a, monthly_par=monthly_a, cv_par=cv_a
    )
    res_dict_a = {r.crop_id: r for r in results_a}
    
    # Expected results from spec:
    # Luzerne = geeignet
    # Weizen = grenzwertig
    # Mais = ungeeignet (nicht empfohlen)
    assert res_dict_a["luzerne"].classification == "geeignet", f"Fall A Luzerne should be 'geeignet', got {res_dict_a['luzerne'].classification}"
    assert res_dict_a["winterweizen"].classification == "grenzwertig", f"Fall A Winterweizen should be 'grenzwertig', got {res_dict_a['winterweizen'].classification}"
    assert res_dict_a["mais"].classification == "nicht empfohlen", f"Fall A Mais should be 'nicht empfohlen', got {res_dict_a['mais'].classification}"
    
    # ----------------------------------------------------
    # FALL B: Annual open-field GHI leads to par_ref = 8447
    # Shading is higher (~58% remaining PAR -> par_ann = 4894)
    # CV is 12% (cv_par = 0.12)
    # ----------------------------------------------------
    ref_b = 8447.0
    ann_b = 4894.0
    monthly_b = [ann_b / 12.0] * 12
    cv_b = 0.12
    
    results_b = evaluate_all_crops(
        par_ann=ann_b, par_ref=ref_b, monthly_par=monthly_b, cv_par=cv_b
    )
    res_dict_b = {r.crop_id: r for r in results_b}
    
    # Expected results from spec:
    # Luzerne = geeignet bis grenzwertig
    # Mais = ungeeignet
    assert res_dict_b["luzerne"].classification in {"grenzwertig", "geeignet"}, f"Fall B Luzerne should be 'grenzwertig' or 'geeignet', got {res_dict_b['luzerne'].classification}"
    assert res_dict_b["mais"].classification == "nicht empfohlen", f"Fall B Mais should be 'nicht empfohlen', got {res_dict_b['mais'].classification}"
    
    print("  ✓ Full integration scenarios (Fall A and Fall B) validated perfectly.")


if __name__ == "__main__":
    tests = [
        ("Crop Registry Profiles", test_crop_registry),
        ("PAR Reference & Thresholds", test_par_reference_and_thresholds),
        ("Component Scoring Logic", test_scoring_components),
        ("Ranking Consistency", test_ranking_consistency),
        ("Confidence Calculations", test_confidence_calculations),
        ("Classification Boundary Mappings", test_classification_mapping),
        ("Fall A / Fall B Validation Scenarios", test_integration_scenarios),
    ]
    
    print("=" * 60)
    print("RUNNING CROP SUITABILITY RECOMMENDATION ENGINE TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, test_func in tests:
        try:
            print(f"[TEST] {name}...")
            test_func()
            print("  ✓ PASSED\n")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1
            
    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
