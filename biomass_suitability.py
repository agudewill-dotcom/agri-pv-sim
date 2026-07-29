import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

@dataclass
class SiteContext:
    hot_dry_index: float
    water_stress_risk: float
    humidity_disease_index: float

@dataclass
class BiomassMetrics:
    P_gs_agri: float
    P_gs_open: float
    DLI_gs_mean: float
    DLI_gs_p10: float
    peak_PPFD_gs: float
    cv_PAR: float

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def evaluate_biomass_suitability(crop_profile, metrics: BiomassMetrics, site_context: SiteContext) -> Dict:
    """
    Evaluates a crop's suitability based purely on vegetative biomass potential under Agri-PV.
    """
    if 'biomass' not in crop_profile.supported_objectives:
        return {
            "crop": getattr(crop_profile, 'display_name', getattr(crop_profile, 'name_de', 'Unknown')),
            "yield_objective": "biomass",
            "score": 0.0,
            "label": "Biomass mode not supported for this crop",
            "warnings": ["No biomass logic available."]
        }
        
    t = crop_profile.thresholds.get('biomass', {})
    
    r_gs = metrics.P_gs_agri / metrics.P_gs_open if metrics.P_gs_open > 0 else 0
    dli_mean = metrics.DLI_gs_mean
    dli_p10 = metrics.DLI_gs_p10
    peak_ppfd = metrics.peak_PPFD_gs
    cv = metrics.cv_PAR

    # Thresholds
    r_gs_min = t.get('r_gs_min', 0.8)
    r_gs_target = t.get('r_gs_target', 0.95)
    dli_min = t.get('DLI_min', 20.0)
    dli_target = t.get('DLI_target', 28.0)
    dli_p10_min = t.get('DLI_p10_min', 16.0)
    peak_ppfd_min = t.get('peak_PPFD_min', 800.0)
    cv_warn = t.get('cv_PAR_warning', 0.25)
    cv_crit = t.get('cv_PAR_critical', 0.35)

    # Scoring components
    if r_gs >= r_gs_min:
        s_r = 0.4 + 0.6 * clamp((r_gs - r_gs_min) / (r_gs_target - r_gs_min), 0, 1) if r_gs_target > r_gs_min else 1.0
    else:
        s_r = clamp(r_gs / r_gs_min, 0, 1) * 0.4
        
    if dli_mean >= dli_min:
        s_dli = 0.4 + 0.6 * clamp((dli_mean - dli_min) / (dli_target - dli_min), 0, 1) if dli_target > dli_min else 1.0
    else:
        s_dli = clamp(dli_mean / dli_min, 0, 1) * 0.4
        
    if dli_p10 >= dli_p10_min:
        s_dli_floor = 0.4 + 0.6 * clamp((dli_p10 - dli_p10_min) / (dli_min - dli_p10_min), 0, 1) if dli_min > dli_p10_min else 1.0
    else:
        s_dli_floor = clamp(dli_p10 / dli_p10_min, 0, 1) * 0.4
        
    s_peak = clamp(peak_ppfd / peak_ppfd_min, 0, 1) if peak_ppfd_min > 0 else 1.0

    if cv <= cv_warn:
        s_hom = 1.0
    elif cv <= cv_crit:
        s_hom = 0.65
    else:
        s_hom = 0.35

    # Site correction
    site_bonus = 0.0
    if crop_profile.crop_group in ["forage_legume_biomass", "forage_biomass", "robust_C3_cereal_biomass"]:
        site_bonus += 0.05 * site_context.hot_dry_index
        site_bonus += 0.03 * site_context.water_stress_risk

    site_malus = 0.0
    if site_context.humidity_disease_index > 0.6:
        site_malus += 0.05
    if crop_profile.quality_warning:
        site_malus += 0.03 * site_context.humidity_disease_index

    # Raw score
    raw_score = (
        0.40 * s_r +
        0.25 * s_dli +
        0.15 * s_dli_floor +
        0.08 * s_peak +
        0.12 * s_hom
    ) + site_bonus - site_malus

    evidence_factor = {
        "A": 1.00,
        "A-B": 0.96,
        "B": 0.92,
        "B-C": 0.86,
        "C": 0.80,
        "C-D": 0.72,
        "D": 0.65
    }.get(crop_profile.evidence_tier, 0.80)

    score = 100 * clamp(raw_score, 0, 1) * evidence_factor

    # Classification
    if score >= 80:
        label = "Suitable for Biomass Target"
    elif score >= 60:
        label = "Conditionally Suitable for Biomass Target"
    elif score >= 40:
        label = "Critical / Field Verification Required"
    else:
        label = "Not Suitable under Simulated Light"

    # Hard rules
    if crop_profile.crop_group in ["high_light_C4_biomass", "high_light_crop"] and r_gs < r_gs_min:
        label = "Not Suitable under Simulated Light"

    if crop_profile.evidence_tier in ["C", "C-D", "D"] and label in ["Suitable for Biomass Target", "Conditionally Suitable for Biomass Target"]:
        label = "Special Crop (Contract Required)"

    # Warnings
    warnings = []
    if crop_profile.evidence_tier in ["C", "C-D", "D"]:
        warnings.append("Proxy derivation: no robust species-specific Agri-PV biomass curve.")
    if getattr(crop_profile, 'quality_warning', False):
        warnings.append("Biomass suitability does not replace active ingredient or essential oil quality evaluation.")
    if cv > cv_warn:
        warnings.append("Light distribution is heterogeneous; non-uniform stand development possible.")
    if dli_p10 < dli_p10_min:
        warnings.append("Low DLI in shaded areas may restrict local growth.")
    if site_context.humidity_disease_index > 0.6:
        warnings.append("Humidity-related disease or quality risk requires site verification.")

    return {
        "crop": getattr(crop_profile, 'display_name', getattr(crop_profile, 'name_de', 'Unknown')),
        "crop_group": crop_profile.crop_group,
        "biomass_target_type": crop_profile.biomass_target_type,
        "yield_objective": "biomass",
        "score": score,
        "label": label,
        "r_gs": r_gs,
        "DLI_gs_mean": dli_mean,
        "DLI_gs_p10": dli_p10,
        "peak_PPFD_gs": peak_ppfd,
        "cv_PAR": cv,
        "evidence_tier": crop_profile.evidence_tier,
        "confidence": crop_profile.confidence_default,
        "warnings": warnings,
        "limiting_factor": "none" if score >= 80 else "Sub-optimal light for robust biomass"
    }

def get_biomass_metrics(crop_profile, metrics_dict, res_a: pd.DataFrame) -> BiomassMetrics:
    gs = crop_profile.growing_season_months
    
    # 1. P_gs_agri / P_gs_open
    p_gs_agri = sum(metrics_dict['monthly_par_agri'][m - 1] for m in gs)
    p_gs_open = sum(metrics_dict['par_open_field']/12.0 for m in gs) # simplistic since we don't have monthly_par_open in metrics sometimes, or we can use metrics_dict.get('monthly_par_open', ... )
    if 'monthly_par_open' in metrics_dict:
        p_gs_open = sum(metrics_dict['monthly_par_open'][m - 1] for m in gs)
        
    # 2. DLI stats
    res_a_gs = res_a[res_a.index.month.isin(gs)]
    if 'par' not in res_a_gs.columns:
        res_a_gs = res_a_gs.copy()
        res_a_gs['par'] = res_a_gs['g_g'] * 2.0565
        
    daily_dli = res_a_gs.groupby(res_a_gs.index.date)['par'].sum()
    dli_mean = float(daily_dli.mean()) if not daily_dli.empty else 0.0
    dli_p10 = float(daily_dli.quantile(0.10)) if not daily_dli.empty else 0.0
    
    # 3. Peak PPFD
    peak_ppfd = float(res_a_gs['par'].max() * (1000000 / 3600)) if not res_a_gs.empty else 0.0
    
    return BiomassMetrics(
        P_gs_agri=p_gs_agri,
        P_gs_open=p_gs_open,
        DLI_gs_mean=dli_mean,
        DLI_gs_p10=dli_p10,
        peak_PPFD_gs=peak_ppfd,
        cv_PAR=metrics_dict['cv_par']
    )

def evaluate_all_biomass(metrics_dict, res_a, site_context: SiteContext) -> list[Dict]:
    from crop_suitability import CROP_REGISTRY
    results = []
    for crop in CROP_REGISTRY.values():
        if 'biomass' not in crop.supported_objectives:
            continue
        c_metrics = get_biomass_metrics(crop, metrics_dict, res_a)
        res = evaluate_biomass_suitability(crop, c_metrics, site_context)
        if res['score'] > 0:
            results.append(res)
    results.sort(key=lambda r: r['score'], reverse=True)
    return results
