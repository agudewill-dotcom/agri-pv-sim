"""
agri-pv-simulator – Agri-PV Crop Suitability Engine
====================================================

Top-level package.  Re-exports the main types and functions.
"""

from crop_profiles import CropProfile, CROP_REGISTRY, get_par_ref_from_ghi
from crop_scoring import SuitabilityResult, evaluate_crop, evaluate_all_crops

__all__ = [
    "CropProfile",
    "CROP_REGISTRY",
    "get_par_ref_from_ghi",
    "SuitabilityResult",
    "evaluate_crop",
    "evaluate_all_crops",
]
