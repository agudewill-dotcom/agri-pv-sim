"""
crop_scoring.py – Agri-PV Crop Suitability Engine: Scoring Module
=================================================================

Implements the **multi-component suitability score** used to evaluate
whether a given Agri-PV light environment is compatible with each crop.

Four scoring components are computed:

* **A – Annual PAR**: overall annual light availability vs. crop threshold.
* **S – Seasonal PAR**: weighted monthly PAR compared to the reference.
* **C – Critical Phase**: light during the crop's most sensitive months.
* **H – Homogeneity**: spatial uniformity of sub-panel PAR (via CV).

Each component returns a 0–1 value.  The final weighted score is
classified into one of four German-language categories:

+----------+---------------------+
| Score    | Classification      |
+==========+=====================+
| ≥ 0.80  | sehr gut geeignet   |
+----------+---------------------+
| 0.65–0.79| geeignet            |
+----------+---------------------+
| 0.45–0.64| grenzwertig         |
+----------+---------------------+
| < 0.45  | nicht empfohlen     |
+----------+---------------------+

All public functions carry full type hints and docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from crop_profiles import (
    CROP_REGISTRY,
    CropProfile,
    get_absolute_thresholds,
    get_monthly_weights,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SuitabilityResult:
    """Container for a single crop's suitability evaluation.

    Attributes
    ----------
    crop_id : str
        Machine-readable crop identifier.
    crop_name_de : str
        German crop name.
    score : float
        Overall weighted suitability score, 0–1.
    classification : str
        German-language classification string derived from *score*.
    confidence : str
        Confidence label: ``'hoch'``, ``'mittel'``, or ``'niedrig'``.
    confidence_value : float
        Numeric confidence value, 0–1.
    evidence_tier : str
        Evidence quality tier (``'A'``, ``'B'``, or ``'C'``).
    limiting_factor : str
        Key of the weakest scoring component
        (``'annual_par'``, ``'seasonal_par'``, ``'critical_phase'``,
        ``'homogeneity'``).
    component_scores : dict
        Individual component scores with keys ``'A'``, ``'S'``,
        ``'C'``, ``'H'``.
    par_min_abs : float
        Absolute minimum PAR threshold in mol m⁻² a⁻¹.
    par_target_abs : float
        Absolute target PAR threshold in mol m⁻² a⁻¹.
    is_proxy : bool
        Whether the crop profile is proxy-based.
    sources : list[str]
        Evidence source references.
    notes_de : str
        German-language notes for UI display.
    """

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


# ---------------------------------------------------------------------------
# Component scoring functions
# ---------------------------------------------------------------------------

def rescale_fraction(val: float, f_min: float, f_target: float) -> float:
    """Helper to map a relative PAR fraction (val) to a 0-1 suitability score
    aligned with guidelines:
    - val >= f_target -> score in [0.80, 1.0] (sehr gut geeignet)
    - f_min <= val < f_target -> score in [0.65, 0.80] (geeignet)
    - f_limit <= val < f_min -> score in [0.45, 0.65] (grenzwertig)
    - val < f_limit -> score in [0.0, 0.45] (nicht empfohlen)
    
    where f_limit is f_min - 0.10.
    """
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
    """Score **Component A** – annual PAR availability.

    Uses a piecewise linear model when ``par_ref`` is provided to align
    directly with classification boundaries:
    - ``par_ann / par_ref >= f_target`` → score ≥ 0.80 (sehr gut geeignet)
    - ``par_ann / par_ref >= f_min`` → score ≥ 0.65 (geeignet)
    - ``par_ann / par_ref >= f_min - 0.10`` → score ≥ 0.45 (grenzwertig)
    - Else → score < 0.45 (nicht empfohlen)

    If ``par_ref`` is not provided, defaults to simple linear interpolation.
    """
    if par_ref is not None and par_ref > 0:
        val = par_ann / par_ref
        f_min = par_min / par_ref
        f_target = par_target / par_ref
        return rescale_fraction(val, f_min, f_target)

    # Fallback to simple linear interpolation
    if par_target <= par_min:
        return 1.0 if par_ann >= par_target else 0.0
    score = (par_ann - par_min) / (par_target - par_min)
    return float(np.clip(score, 0.0, 1.0))


def score_seasonal_par(
    monthly_par: list[float],
    crop: CropProfile,
    par_ref: float,
) -> float:
    """Score **Component S** – seasonally-weighted PAR.

    Computes the weighted average of monthly PAR fractions
    using the crop's growing-month weights, then rescales via piecewise mapping.
    """
    if len(monthly_par) != 12:
        raise ValueError(
            f"monthly_par must have exactly 12 elements, got {len(monthly_par)}"
        )

    weights = get_monthly_weights(crop)
    monthly_ref = par_ref / 12.0  # simple equal-split reference per month

    # Weighted average of monthly fraction (actual / reference)
    weighted_fraction = sum(
        w * (mp / monthly_ref) for w, mp in zip(weights, monthly_par) if monthly_ref > 0
    )

    return rescale_fraction(weighted_fraction, crop.f_min, crop.f_target)


def score_critical_phase(
    monthly_par: list[float],
    crop: CropProfile,
    par_ref: float,
) -> float:
    """Score **Component C** – critical-phase light availability.

    Uses the *minimum* monthly PAR fraction across the crop's
    ``critical_months``, then rescales via piecewise mapping.
    """
    if len(monthly_par) != 12:
        raise ValueError(
            f"monthly_par must have exactly 12 elements, got {len(monthly_par)}"
        )

    monthly_ref = par_ref / 12.0

    # Gather fractions for critical months only
    critical_fractions = [
        monthly_par[m - 1] / monthly_ref
        for m in crop.critical_months
        if monthly_ref > 0
    ]

    if not critical_fractions:
        return 1.0  # no critical months defined → no penalty

    min_fraction = min(critical_fractions)

    return rescale_fraction(min_fraction, crop.f_min, crop.f_target)


def score_homogeneity(cv_par: float, cv_max: float) -> float:
    """Score **Component H** – spatial PAR homogeneity.

    Returns 1.0 when ``cv_par == 0`` (perfectly uniform), 0.0 when
    ``cv_par ≥ cv_max``, and linearly interpolates between.

    Parameters
    ----------
    cv_par : float
        Measured spatial coefficient of variation of PAR under panels.
    cv_max : float
        Maximum tolerable CV for the crop.

    Returns
    -------
    float
        Score in [0, 1].

    Examples
    --------
    >>> score_homogeneity(0.10, 0.25)
    0.6
    >>> score_homogeneity(0.0, 0.25)
    1.0
    >>> score_homogeneity(0.30, 0.25)
    0.0
    """
    if cv_max <= 0:
        return 0.0 if cv_par > 0 else 1.0
    score = 1.0 - (cv_par / cv_max)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Classification, confidence & limiting-factor helpers
# ---------------------------------------------------------------------------

def classify_score(score: float) -> str:
    """Map a 0–1 suitability score to a German classification string.

    Parameters
    ----------
    score : float
        Weighted suitability score.

    Returns
    -------
    str
        One of ``'sehr gut geeignet'``, ``'geeignet'``,
        ``'grenzwertig'``, or ``'nicht empfohlen'``.

    Examples
    --------
    >>> classify_score(0.85)
    'sehr gut geeignet'
    >>> classify_score(0.70)
    'geeignet'
    >>> classify_score(0.50)
    'grenzwertig'
    >>> classify_score(0.30)
    'nicht empfohlen'
    """
    if score >= 0.80:
        return "sehr gut geeignet"
    if score >= 0.65:
        return "geeignet"
    if score >= 0.45:
        return "grenzwertig"
    return "nicht empfohlen"


def calculate_confidence(
    evidence_tier: str,
    has_monthly: bool,
    has_hourly: bool,
    is_proxy: bool,
) -> tuple[float, str]:
    """Compute a confidence value and label for the evaluation.

    The confidence reflects both the quality of the underlying crop
    evidence and the resolution of the input data provided by the user.

    Parameters
    ----------
    evidence_tier : str
        ``'A'``, ``'B'``, or ``'C'``.
    has_monthly : bool
        Whether monthly PAR data was supplied.
    has_hourly : bool
        Whether hourly / sub-hourly PAR data was available.
    is_proxy : bool
        Whether the crop profile is proxy-based.

    Returns
    -------
    tuple[float, str]
        ``(confidence_value, confidence_label)`` where the label is
        ``'hoch'``, ``'mittel'``, or ``'niedrig'``.
    """
    # Base confidence from evidence tier
    tier_scores = {"A": 0.90, "B": 0.70, "C": 0.50}
    base = tier_scores.get(evidence_tier.upper(), 0.50)

    # Bonus for higher-resolution input data
    if has_hourly:
        base += 0.05
    elif has_monthly:
        base += 0.00  # monthly is the expected baseline
    else:
        base -= 0.10  # annual-only → less reliable

    # Penalty for proxy-based profiles
    if is_proxy:
        base -= 0.15

    value = float(np.clip(base, 0.0, 1.0))

    # Map to label
    if value >= 0.75:
        label = "hoch"
    elif value >= 0.50:
        label = "mittel"
    else:
        label = "niedrig"

    return value, label


_COMPONENT_NAME_MAP: Dict[str, str] = {
    "A": "annual_par",
    "S": "seasonal_par",
    "C": "critical_phase",
    "H": "homogeneity",
}


def identify_limiting_factor(components: dict[str, float]) -> str:
    """Return the human-readable name of the weakest scoring component.

    Parameters
    ----------
    components : dict[str, float]
        Component scores with keys ``'A'``, ``'S'``, ``'C'``, ``'H'``.

    Returns
    -------
    str
        One of ``'annual_par'``, ``'seasonal_par'``,
        ``'critical_phase'``, ``'homogeneity'``.

    Examples
    --------
    >>> identify_limiting_factor({'A': 0.9, 'S': 0.7, 'C': 0.3, 'H': 0.8})
    'critical_phase'
    """
    weakest_key = min(components, key=components.get)  # type: ignore[arg-type]
    return _COMPONENT_NAME_MAP[weakest_key]


# ---------------------------------------------------------------------------
# Main evaluation functions
# ---------------------------------------------------------------------------

def evaluate_crop(
    crop: CropProfile,
    par_ann: float,
    par_ref: float,
    monthly_par: list[float] | None = None,
    cv_par: float | None = None,
    has_hourly: bool = False,
) -> SuitabilityResult:
    """Evaluate the suitability of a single crop for the given light regime.

    This is the main entry point for single-crop scoring.  It computes
    all four components (A, S, C, H), applies crop-specific weights,
    and packages the result into a :class:`SuitabilityResult`.

    When ``monthly_par`` is not provided, components **S** and **C**
    default to the same score as component **A** (annual-only fallback).
    When ``cv_par`` is not provided, component **H** defaults to 1.0
    (optimistic assumption).

    Parameters
    ----------
    crop : CropProfile
        Crop profile to evaluate.
    par_ann : float
        Annual PAR under panels in mol m⁻² a⁻¹.
    par_ref : float
        Annual open-field PAR reference in mol m⁻² a⁻¹.
    monthly_par : list[float] | None
        Optional length-12 list of monthly PAR values in mol m⁻².
    cv_par : float | None
        Optional spatial CV of PAR under panels.
    has_hourly : bool
        Whether hourly data was used (affects confidence).

    Returns
    -------
    SuitabilityResult
        Full evaluation result.
    """
    par_min, par_target = get_absolute_thresholds(crop, par_ref)

    # Component A – always available
    comp_a = score_annual_par(par_ann, par_min, par_target, par_ref)

    # Component S & C – need monthly data
    has_monthly = monthly_par is not None
    if has_monthly:
        assert monthly_par is not None  # help type checker
        comp_s = score_seasonal_par(monthly_par, crop, par_ref)
        comp_c = score_critical_phase(monthly_par, crop, par_ref)
    else:
        # Fallback: propagate annual score
        comp_s = comp_a
        comp_c = comp_a

    # Component H – needs CV data
    if cv_par is not None:
        comp_h = score_homogeneity(cv_par, crop.cv_max)
    else:
        # Optimistic default: assume perfectly uniform
        comp_h = 1.0

    components: Dict[str, float] = {
        "A": comp_a,
        "S": comp_s,
        "C": comp_c,
        "H": comp_h,
    }

    # Weighted final score
    w = crop.weights
    final_score = (
        w["wA"] * comp_a
        + w["wS"] * comp_s
        + w["wC"] * comp_c
        + w["wH"] * comp_h
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    classification = classify_score(final_score)
    confidence_value, confidence_label = calculate_confidence(
        crop.evidence_tier, has_monthly, has_hourly, crop.is_proxy
    )
    limiting = identify_limiting_factor(components)

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
        is_proxy=crop.is_proxy,
        sources=list(crop.evidence_sources),
        notes_de=crop.notes_de,
    )


def evaluate_all_crops(
    par_ann: float,
    par_ref: float,
    monthly_par: list[float] | None = None,
    cv_par: float | None = None,
    has_hourly: bool = False,
) -> list[SuitabilityResult]:
    """Evaluate all 11 crops in the registry and return sorted results.

    Parameters
    ----------
    par_ann : float
        Annual PAR under panels in mol m⁻² a⁻¹.
    par_ref : float
        Annual open-field PAR reference in mol m⁻² a⁻¹.
    monthly_par : list[float] | None
        Optional length-12 list of monthly PAR values in mol m⁻².
    cv_par : float | None
        Optional spatial CV of PAR under panels.
    has_hourly : bool
        Whether hourly data was used (affects confidence).

    Returns
    -------
    list[SuitabilityResult]
        All 11 crop results, sorted by ``score`` descending.

    Examples
    --------
    >>> from crop_profiles import get_par_ref_from_ghi
    >>> par_ref = get_par_ref_from_ghi(1100.0)
    >>> results = evaluate_all_crops(par_ann=6000.0, par_ref=par_ref)
    >>> len(results)
    11
    >>> results[0].score >= results[-1].score
    True
    """
    results = [
        evaluate_crop(crop, par_ann, par_ref, monthly_par, cv_par, has_hourly)
        for crop in CROP_REGISTRY.values()
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results
