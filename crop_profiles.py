"""
crop_profiles.py – Agri-PV Crop Suitability Engine: Crop Profiles (Bridge)
========================================================================
Bridges crop profile dataclass and crop registry directly from the unified
YAML-based database module `crop_suitability`. Achieves zero duplication of data.
"""

from __future__ import annotations

from crop_suitability import (
    CROP_REGISTRY,
    CropProfile,
    get_absolute_thresholds,
    get_monthly_weights,
    get_par_ref_from_ghi,
)
