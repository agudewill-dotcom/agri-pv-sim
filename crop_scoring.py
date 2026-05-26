"""
crop_scoring.py – Agri-PV Crop Suitability Engine: Scoring Module (Bridge)
==========================================================================
Bridges suitability scoring components and main crop evaluation logic directly
from the unified `crop_suitability` engine. Achieves zero duplication.
"""

from __future__ import annotations

from crop_suitability import (
    SuitabilityResult,
    calculate_confidence,
    classify_score,
    evaluate_all_crops,
    evaluate_crop,
    identify_limiting_factor,
    rescale_fraction,
    score_annual_par,
    score_critical_phase,
    score_homogeneity,
    score_seasonal_par,
)
