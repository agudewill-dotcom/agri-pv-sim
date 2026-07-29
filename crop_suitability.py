"""
crop_suitability.py – Agri-PV Crop Suitability Engine
======================================================
Loads agronomic databases and performs multi-component suitability evaluations
using relative PAR, DLI, peak PPFD, and spatial homogeneity metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Load Databases
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crop_database.yaml")
SOURCES_PATH = os.path.join(BASE_DIR, "sources.yaml")

with open(DB_PATH, "r", encoding="utf-8") as f:
    CROP_DB = yaml.safe_load(f)

with open(SOURCES_PATH, "r", encoding="utf-8") as f:
    SOURCES_REGISTRY = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Result Class
# ---------------------------------------------------------------------------
@dataclass
class SuitabilityResult:
    crop_id: str
    crop_name_de: str
    score: float
    classification: str
    confidence: str
    confidence_value: float
    evidence_tier: str
    limiting_factor: str
    component_scores: Dict[str, float]
    par_min_abs: float
    par_target_abs: float
    is_proxy: bool
    sources: List[str]
    notes_de: str
    warning: str
    source_references: List[str]

    @property
    def annual_PAR_score(self) -> float:
        return self.component_scores.get("A", 0.0)

    @property
    def growing_season_PAR_score(self) -> float:
        return self.component_scores.get("S", 0.0)

    @property
    def DLI_score(self) -> float:
        return self.component_scores.get("C", 0.0)

    @property
    def homogeneity_score(self) -> float:
        return self.component_scores.get("H", 0.0)

    @property
    def confidence_level(self) -> str:
        return self.confidence


# ---------------------------------------------------------------------------
# Dataclass for Crop Schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CropProfile:
    id: str
    name_de: str
    name_en: str
    f_min: float
    f_target: float
    evidence_tier: str
    evidence_sources: List[str]
    crop_group: str
    critical_months: List[int]
    growing_months: List[int]
    peak_ppfd_min: float
    cv_max: float
    weights: Dict[str, float]
    notes_de: str
    is_proxy: bool
    # Additional specs
    botanical_name: str
    proxy_group: str
    use_type: str
    r_ann_min: float
    r_ann_target: float
    r_crit_min: float
    r_crit_target: float
    DLI_min: float
    DLI_target: float
    peak_PPFD_min: float
    warning_text: str
    source_group: str
    source_references: List[str]
    
    # Phase 2 dual-objective fields
    supported_objectives: List[str]
    biomass_target_type: str
    growing_season_months: List[int]
    biomass_critical_months: List[int]
    reproductive_critical_months: List[int]
    thresholds: Dict[str, Dict[str, float]]
    quality_warning: bool
    confidence_default: str


# Build backward-compatible profile registry
CROP_REGISTRY: Dict[str, CropProfile] = {}
for cid, entry in CROP_DB.items():
    t_rep = entry.get("thresholds", {}).get("reproductive", {})
    CROP_REGISTRY[cid] = CropProfile(
        id=cid,
        name_de=entry.get("display_name", ""),
        name_en=cid.replace("_", " ").title(),
        f_min=t_rep.get("r_ann_min", 0.8),
        f_target=t_rep.get("r_ann_target", 0.95),
        evidence_tier=entry.get("evidence_tier", "C"),
        evidence_sources=[
            f"{s}: {SOURCES_REGISTRY.get(s, {}).get('title', 'Ref')}"
            for s in entry.get("source_references", [])
        ],
        crop_group=entry.get("crop_group", ""),
        critical_months=entry.get("reproductive_critical_months", []),
        growing_months=entry.get("growing_season_months", []),
        peak_ppfd_min=t_rep.get("peak_PPFD_min", 800.0),
        cv_max=entry.get("cv_max", 0.3),
        weights=entry.get("weights", {"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10}),
        notes_de=entry.get("notes_de", ""),
        is_proxy=entry.get("is_proxy", False),
        botanical_name=entry.get("botanical_name", ""),
        proxy_group=entry.get("proxy_group", ""),
        use_type=entry.get("use_type", ""),
        r_ann_min=t_rep.get("r_ann_min", 0.8),
        r_ann_target=t_rep.get("r_ann_target", 0.95),
        r_crit_min=t_rep.get("r_crit_min", 0.8),
        r_crit_target=t_rep.get("r_crit_target", 0.95),
        DLI_min=t_rep.get("DLI_min", 20.0),
        DLI_target=t_rep.get("DLI_target", 28.0),
        peak_PPFD_min=t_rep.get("peak_PPFD_min", 800.0),
        warning_text=entry.get("warning_text", ""),
        source_group=entry.get("source_group", ""),
        source_references=entry.get("source_references", []),
        supported_objectives=entry.get("supported_objectives", ["reproductive"]),
        biomass_target_type=entry.get("biomass_target_type", ""),
        growing_season_months=entry.get("growing_season_months", []),
        biomass_critical_months=entry.get("biomass_critical_months", []),
        reproductive_critical_months=entry.get("reproductive_critical_months", []),
        thresholds=entry.get("thresholds", {}),
        quality_warning=entry.get("quality_warning", False),
        confidence_default=entry.get("confidence_default", "low")
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_par_ref_from_ghi(ghi_annual_kwh: float, f_par: float = 0.45) -> float:
    return ghi_annual_kwh * 3.6 * f_par * 4.57


def get_absolute_thresholds(crop: CropProfile, par_ref: float) -> tuple[float, float]:
    return crop.r_ann_min * par_ref, crop.r_ann_target * par_ref


def get_monthly_weights(crop: CropProfile) -> list[float]:
    n_active = len(crop.growing_months)
    if n_active == 0:
        return [0.0] * 12
    weight_per_month = 1.0 / n_active
    return [
        weight_per_month if (month_idx + 1) in crop.growing_months else 0.0
        for month_idx in range(12)
    ]


# ---------------------------------------------------------------------------
# Piecewise Rescaling
# ---------------------------------------------------------------------------
def rescale_fraction(val: float, f_min: float, f_target: float) -> float:
    f_limit = f_min - 0.10
    if val >= f_target:
        if f_target >= 1.0:
            return 1.0
        score = 0.80 + (val - f_target) / (1.0 - f_target) * 0.20
    elif val >= f_min:
        if f_target <= f_min:
            return 0.80
        score = 0.65 + (val - f_min) / (f_target - f_min) * 0.15
    elif val >= f_limit:
        if f_min <= f_limit:
            return 0.65
        score = 0.45 + (val - f_limit) / (f_min - f_limit) * 0.20
    else:
        if f_limit <= 0.0:
            return 0.0
        score = val / f_limit * 0.45
    return float(np.clip(score, 0.0, 1.0))


def score_annual_par(
    par_ann: float,
    par_min: float,
    par_target: float,
    par_ref: Optional[float] = None,
) -> float:
    if par_ref is not None and par_ref > 0:
        val = par_ann / par_ref
        f_min = par_min / par_ref
        f_target = par_target / par_ref
        return rescale_fraction(val, f_min, f_target)
    if par_target <= par_min:
        return 1.0 if par_ann >= par_target else 0.0
    score = (par_ann - par_min) / (par_target - par_min)
    return float(np.clip(score, 0.0, 1.0))


def score_seasonal_par(
    monthly_par: list[float],
    crop: CropProfile,
    par_ref: float,
) -> float:
    if len(monthly_par) != 12:
        raise ValueError("monthly_par must have exactly 12 elements")
    weights = get_monthly_weights(crop)
    monthly_ref = par_ref / 12.0
    weighted_fraction = sum(
        w * (mp / monthly_ref) for w, mp in zip(weights, monthly_par) if monthly_ref > 0
    )
    return rescale_fraction(weighted_fraction, crop.r_ann_min, crop.r_ann_target)


def score_critical_phase(
    monthly_par: list[float],
    crop: CropProfile,
    par_ref: float,
) -> float:
    if len(monthly_par) != 12:
        raise ValueError("monthly_par must have exactly 12 elements")
    monthly_ref = par_ref / 12.0
    critical_fractions = [
        monthly_par[m - 1] / monthly_ref
        for m in crop.critical_months
        if monthly_ref > 0
    ]
    if not critical_fractions:
        return 1.0
    min_fraction = min(critical_fractions)
    return rescale_fraction(min_fraction, crop.r_crit_min, crop.r_crit_target)


def score_homogeneity(cv_par: float, cv_max: float) -> float:
    if cv_max <= 0:
        return 0.0 if cv_par > 0 else 1.0
    score = 1.0 - (cv_par / cv_max)
    return float(np.clip(score, 0.0, 1.0))


def calculate_confidence(
    evidence_tier: str,
    has_monthly: bool,
    has_hourly: bool,
    is_proxy: bool,
) -> tuple[float, str]:
    tier_scores = {"A": 0.90, "B": 0.70, "C": 0.50}
    base = tier_scores.get(evidence_tier.upper(), 0.50)
    if has_hourly:
        base += 0.05
    elif has_monthly:
        base += 0.00
    else:
        base -= 0.10
    if is_proxy:
        base -= 0.15
    value = float(np.clip(base, 0.0, 1.0))
    if value >= 0.75:
        label = "hoch"
    elif value >= 0.50:
        label = "mittel"
    else:
        label = "niedrig"
    return value, label


def identify_limiting_factor(components: dict[str, float]) -> str:
    weakest_key = min(components, key=components.get)
    mapping = {
        "A": "annual_par",
        "S": "seasonal_par",
        "C": "critical_phase",
        "H": "homogeneity",
    }
    return mapping[weakest_key]


def classify_score(score: float) -> str:
    if score >= 0.65:
        return "geeignet als Hauptkultur"
    if score >= 0.45:
        return "grenzwertig"
    return "nicht empfohlen"


# ---------------------------------------------------------------------------
# English Explanation Builder
# ---------------------------------------------------------------------------
def generate_english_explanation(
    crop: CropProfile,
    r_ann: float,
    r_crit: float,
    classification: str,
    limiting: str,
    comp_scores: dict[str, float],
) -> str:
    parts = []
    # Intro
    parts.append(
        f"{crop.name_en} is classified as <b>{classification}</b>. "
        f"The relative annual PAR is <b>{r_ann*100:.1f}%</b> (Threshold: {crop.r_ann_min*100:.0f}% min / {crop.r_ann_target*100:.0f}% target) "
        f"and the relative radiation during the critical period ({', '.join(map(str, crop.critical_months))}) is <b>{r_crit*100:.1f}%</b>."
    )
    # Shading and Limiting Factor
    if limiting == "annual_par":
        parts.append(
            "The main limiting factor is the insufficient annual light sum (A), "
            "which falls below the physiological needs of the crop."
        )
    elif limiting == "seasonal_par":
        parts.append(
            "The seasonal light availability (S) during the active growing phase is insufficient."
        )
    elif limiting == "critical_phase":
        parts.append(
            "A critical lack of light (C) during sensitive developmental stages "
            "represents the main limitation and will likely impair grain or fruit set."
        )
    elif limiting == "homogeneity":
        parts.append(
            "The high spatial light heterogeneity (H) beneath the modules carries the risk of uneven ripening."
        )
    else:
        parts.append(
            "The light parameters are within the agronomically safe range."
        )

    # Literature / Evidence
    if crop.evidence_tier in ["A", "B"]:
        parts.append(
            f"The classification is based on the agronomic evidence group <b>{crop.source_group}</b> "
            f"with verified field trials (Tier {crop.evidence_tier})."
        )
    else:
        parts.append(
            f"Since there is no robust species-specific Agri-PV PAR curve for this crop, "
            f"it is conservatively evaluated as a proxy using the group <b>{crop.source_group}</b>."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_crop(
    crop: CropProfile,
    par_ann: float,
    par_ref: float,
    monthly_par: list[float] | None = None,
    cv_par: float | None = None,
    has_hourly: bool = False,
    peak_ppfd_crit: float | None = None,
) -> SuitabilityResult:
    par_min, par_target = get_absolute_thresholds(crop, par_ref)
    comp_a = score_annual_par(par_ann, par_min, par_target, par_ref)

    has_monthly = monthly_par is not None
    monthly_ref = par_ref / 12.0
    r_ann = par_ann / par_ref if par_ref > 0 else 0.0

    if has_monthly and monthly_par is not None:
        comp_s = score_seasonal_par(monthly_par, crop, par_ref)
        comp_c = score_critical_phase(monthly_par, crop, par_ref)
        
        # Calculate r_crit
        sum_agri = sum(monthly_par[m - 1] for m in crop.critical_months)
        sum_ref = monthly_ref * len(crop.critical_months)
        r_crit = sum_agri / sum_ref if sum_ref > 0 else 0.0
    else:
        comp_s = comp_a
        comp_c = comp_a
        r_crit = r_ann

    if cv_par is not None:
        comp_h = score_homogeneity(cv_par, crop.cv_max)
    else:
        comp_h = 1.0
        cv_par = 0.0

    # Optional Peak PPFD Penalty
    if peak_ppfd_crit is not None and peak_ppfd_crit < crop.peak_PPFD_min:
        comp_c = min(comp_c, 0.40)  # penalize critical component

    components = {"A": comp_a, "S": comp_s, "C": comp_c, "H": comp_h}
    w = crop.weights
    final_score = w["wA"] * comp_a + w["wS"] * comp_s + w["wC"] * comp_c + w["wH"] * comp_h
    final_score = float(np.clip(final_score, 0.0, 1.0))

    # Base Suitability
    if final_score >= 0.80:
        suitability = "Highly Suitable"
    elif final_score >= 0.65:
        suitability = "Suitable"
    elif final_score >= 0.45:
        suitability = "Marginal"
    else:
        suitability = "Not Recommended"

    # Peak PPFD penalty on base suitability
    if peak_ppfd_crit is not None and peak_ppfd_crit < crop.peak_PPFD_min:
        suitability = "Marginal" if suitability == "Suitable" else "Not Recommended"

    # Homogeneity Penalty on base suitability
    if cv_par > 0.25 and crop.crop_group in ["robust_cereal", "ancient_grain", "high_light_crop"]:
        suitability = "Marginal" if suitability == "Suitable" else "Not Recommended"

    # Standard DIN classification and warnings
    warning_text = crop.warning_text
    if crop.evidence_tier in ["A", "B"] and crop.crop_group in ["forage", "robust_cereal", "ancient_grain", "niche_crop"]:
        if suitability in ["Suitable", "Highly Suitable"]:
            classification = "Suitable as Primary Crop"
        elif suitability == "Marginal":
            classification = "Marginal / Requires Verification"
        else:
            classification = "Not Recommended"
    else:
        # Special crop floor check
        passes_floor = (comp_a >= 0.65) and (comp_c >= 0.65) and (comp_h >= 0.50)
        if suitability in ["Suitable", "Highly Suitable"] and passes_floor:
            if crop.crop_group == "special_crop":
                classification = "Suitable as Special Crop / Flower Strip"
            else:
                classification = "Special Crop Only (Contract Required)"
        elif suitability == "Not Recommended" or final_score < 0.45:
            classification = "Not Recommended"
        else:
            classification = "Requires Agronomic Verification"
            if not warning_text:
                warning_text = (
                    "No robust species-specific Agri-PV PAR curve available for this crop. "
                    "Evaluation performed as proxy based on light preference, crop group, and site PAR."
                )

    # Specific C3 Cereal, Forage and Maize shade constraints
    if crop.id == "mais" and r_ann < 0.80:
        classification = "Not Recommended"

    confidence_value, confidence_label = calculate_confidence(
        crop.evidence_tier, has_monthly, has_hourly=has_hourly, is_proxy=(crop.evidence_tier == "C")
    )
    limiting = identify_limiting_factor(components)
    explanation = generate_english_explanation(crop, r_ann, r_crit, classification, limiting, components)

    return SuitabilityResult(
        crop_id=crop.id,
        crop_name_de=crop.name_de,
        score=final_score,
        classification=classification,
        confidence=confidence_label,
        confidence_value=confidence_value,
        evidence_tier=crop.evidence_tier,
        limiting_factor=limiting,
        component_scores=components,
        par_min_abs=par_min,
        par_target_abs=par_target,
        is_proxy=(crop.evidence_tier == "C"),
        sources=list(crop.evidence_sources),
        notes_de=explanation,
        warning=warning_text,
        source_references=crop.source_references,
    )


def evaluate_all_crops(
    par_ann: float,
    par_ref: float,
    monthly_par: list[float] | None = None,
    cv_par: float | None = None,
    has_hourly: bool = False,
    peak_ppfd_crit: float | None = None,
) -> list[SuitabilityResult]:
    results = [
        evaluate_crop(crop, par_ann, par_ref, monthly_par, cv_par, has_hourly, peak_ppfd_crit)
        for crop in CROP_REGISTRY.values()
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results
