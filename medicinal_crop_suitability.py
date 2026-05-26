"""
medicinal_crop_suitability.py – Suitability engine for medicinal and special crops
"""

import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

# Load databases
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "crop_database_medicinal.yaml")
SOURCES_PATH = os.path.join(BASE_DIR, "sources_medicinal.yaml")

with open(DB_PATH, "r", encoding="utf-8") as f:
    MED_CROP_DB = yaml.safe_load(f)

with open(SOURCES_PATH, "r", encoding="utf-8") as f:
    MED_SOURCES_REGISTRY = yaml.safe_load(f)

@dataclass
class MedicinalCropProfile:
    id: str
    display_name: str
    botanical_name: str
    crop_group: str
    use_type: str
    critical_months: List[int]
    r_ann_min: float
    r_ann_target: float
    r_crit_min: float
    r_crit_target: float
    DLI_min: float
    DLI_target: float
    peak_PPFD_min: float
    homogeneity_sensitivity: str
    evidence_tier: str
    confidence_default: str
    source_group: str
    source_references: List[str]
    warning_text: str
    notes_de: str

@dataclass
class MedicinalSuitabilityResult:
    crop_id: str
    crop_name: str
    botanical_name: str
    crop_group: str
    use_type: str
    r_ann: float
    critical_months: List[int]
    r_crit: float
    DLI_crit: float
    DLI_min: float
    DLI_target: float
    peak_PPFD_crit: float
    peak_PPFD_min: float
    cv_PAR: float
    homogeneity_class: str
    suitability_class: str
    confidence_level: str
    evidence_tier: str
    limiting_factor: str
    explanation_en: str
    warning_text: str
    source_references: List[str]

# Month mapping
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def parse_months(month_list: List[str]) -> List[int]:
    return [MONTH_MAP[m] for m in month_list if m in MONTH_MAP]

# Build registry
MED_CROP_REGISTRY: Dict[str, MedicinalCropProfile] = {}
for cid, entry in MED_CROP_DB.items():
    MED_CROP_REGISTRY[cid] = MedicinalCropProfile(
        id=cid,
        display_name=entry["display_name"],
        botanical_name=entry.get("botanical_name", ""),
        crop_group=entry["crop_group"],
        use_type=entry.get("use_type", ""),
        critical_months=parse_months(entry["critical_months"]),
        r_ann_min=entry["r_ann_min"],
        r_ann_target=entry["r_ann_target"],
        r_crit_min=entry["r_crit_min"],
        r_crit_target=entry["r_crit_target"],
        DLI_min=entry["DLI_min"],
        DLI_target=entry["DLI_target"],
        peak_PPFD_min=entry["peak_PPFD_min"],
        homogeneity_sensitivity=entry["homogeneity_sensitivity"],
        evidence_tier=entry["evidence_tier"],
        confidence_default=entry["confidence_default"],
        source_group=entry["source_group"],
        source_references=entry.get("source_references", []),
        warning_text=entry.get("warning_text", ""),
        notes_de=entry.get("notes_de", ""),
    )

def calculate_dli_and_peak(hourly_par: pd.Series, critical_months: List[int]) -> tuple[float, float]:
    mask = hourly_par.index.month.isin(critical_months)
    crit_data = hourly_par[mask]
    
    if crit_data.empty:
        return 0.0, 0.0
        
    # DLI: sum of PAR in mol per day.
    # daily integral = sum(µmol/m2/s * 3600) / 1000000 = mol/m2/d
    # Then take mean over all days in the critical months
    daily_dli = (crit_data * 3600 / 1e6).groupby(crit_data.index.date).sum()
    mean_dli = daily_dli.mean()
    
    # Peak PPFD: 90th percentile of hourly PAR between 11:00 and 14:00 to represent clear sky peaks
    peak_mask = crit_data.index.hour.isin([11, 12, 13, 14])
    peak_data = crit_data[peak_mask]
    mean_peak_ppfd = peak_data.quantile(0.90) if not peak_data.empty else 0.0
    
    return float(mean_dli), float(mean_peak_ppfd)

def evaluate_medicinal_crop(
    crop: MedicinalCropProfile,
    annual_PAR_agri: float,
    annual_PAR_openfield: float,
    monthly_PAR_agri: List[float],
    monthly_PAR_openfield: List[float],
    cv_PAR: float,
    hourly_par: Optional[pd.Series] = None,
    DLI_crit: Optional[float] = None,
    peak_PPFD_crit: Optional[float] = None
) -> MedicinalSuitabilityResult:
    
    # 1. Calculate r_ann
    r_ann = annual_PAR_agri / annual_PAR_openfield if annual_PAR_openfield > 0 else 0.0
    
    # 2. Calculate r_crit
    sum_agri_crit = sum(monthly_PAR_agri[m - 1] for m in crop.critical_months)
    sum_open_crit = sum(monthly_PAR_openfield[m - 1] for m in crop.critical_months)
    r_crit = sum_agri_crit / sum_open_crit if sum_open_crit > 0 else 0.0

    # Calculate DLI and Peak if hourly data is provided
    if hourly_par is not None:
        c_dli, c_peak = calculate_dli_and_peak(hourly_par, crop.critical_months)
        if DLI_crit is None: DLI_crit = c_dli
        if peak_PPFD_crit is None: peak_PPFD_crit = c_peak
    else:
        if DLI_crit is None: DLI_crit = 0.0
        if peak_PPFD_crit is None: peak_PPFD_crit = 0.0

    # 3. Base Classification
    # As requested, the minimal tolerated PAR in critical months (and DLI) is the reference for Suitable
    if r_crit >= crop.r_crit_min and DLI_crit >= crop.DLI_min:
        base_class = "suitable"
    else:
        base_class = "unsuitable"

    limiting_factor = ""
    
    if base_class == "unsuitable":
        if r_crit < crop.r_crit_min: limiting_factor = "Sub-optimal PAR in critical phase"
        elif DLI_crit < crop.DLI_min: limiting_factor = "Sub-optimal DLI"

    warning_text = crop.warning_text

    # 4. Homogeneity Class
    if cv_PAR <= 0.15:
        homogeneity_class = "good"
    elif cv_PAR <= 0.25:
        homogeneity_class = "moderate"
        if "moderate" not in warning_text.lower():
            warning_text += " Moderate light heterogeneity."
    else:
        homogeneity_class = "critical"

    # 5. Peak PPFD Downgrade
    if peak_PPFD_crit < crop.peak_PPFD_min:
        if base_class == "suitable":
            base_class = "marginal"
        elif base_class == "marginal":
            base_class = "unsuitable"
        limiting_factor = "Peak PAR/PPFD too low in critical phase"
    
    # 6. Homogeneity Downgrade
    if cv_PAR > 0.25:
        if base_class == "suitable":
            base_class = "marginal"
        elif base_class == "marginal":
            base_class = "unsuitable"
        if not limiting_factor:
            limiting_factor = "Light distribution too heterogeneous"


    # 8. Evidence Rule Formatting
    final_class = base_class
    if "C" in crop.evidence_tier:
        if final_class == "suitable":
            final_class = "suitable as special crop with agronomic trial"
        elif final_class == "marginal":
            final_class = "marginal / agronomic trial required"
    if "D" in crop.evidence_tier:
        if final_class != "unsuitable":
            final_class = "only with field trial / market proof"

    # 9. Explanation Generation
    explanation = (
        f"{crop.display_name} is classified as {final_class}. "
        f"The relative annual PAR is {r_ann*100:.1f}%, "
        f"PAR in critical period {crop.critical_months} is {r_crit*100:.1f}%, "
        f"and mean DLI in critical phase is {DLI_crit:.1f} mol/m²/d. "
        f"Classification based on {crop.source_group}. Since no robust species-specific Agri-PV PAR curve is available, "
        f"this evaluation is provided with confidence level '{crop.confidence_default}'."
    )

    if not limiting_factor and final_class != "unsuitable":
        limiting_factor = "none (values within agronomically plausible range)"
    elif not limiting_factor:
        limiting_factor = "insufficient PAR / DLI for this crop group"

    sources = [f"{s}: {MED_SOURCES_REGISTRY.get(s, {}).get('title', 'Ref')}" for s in crop.source_references]

    return MedicinalSuitabilityResult(
        crop_id=crop.id,
        crop_name=crop.display_name,
        botanical_name=crop.botanical_name,
        crop_group=crop.crop_group,
        use_type=crop.use_type,
        r_ann=r_ann,
        critical_months=crop.critical_months,
        r_crit=r_crit,
        DLI_crit=DLI_crit,
        DLI_min=crop.DLI_min,
        DLI_target=crop.DLI_target,
        peak_PPFD_crit=peak_PPFD_crit,
        peak_PPFD_min=crop.peak_PPFD_min,
        cv_PAR=cv_PAR,
        homogeneity_class=homogeneity_class,
        suitability_class=final_class,
        confidence_level=crop.confidence_default,
        evidence_tier=crop.evidence_tier,
        limiting_factor=limiting_factor,
        explanation_en=explanation,
        warning_text=warning_text.strip(),
        source_references=sources
    )

def evaluate_all_medicinal_crops(
    annual_PAR_agri: float,
    annual_PAR_openfield: float,
    monthly_PAR_agri: List[float],
    monthly_PAR_openfield: List[float],
    cv_PAR: float,
    hourly_par: Optional[pd.Series] = None
) -> List[MedicinalSuitabilityResult]:
    
    results = []
    for cid, crop in MED_CROP_REGISTRY.items():
        res = evaluate_medicinal_crop(
            crop=crop,
            annual_PAR_agri=annual_PAR_agri,
            annual_PAR_openfield=annual_PAR_openfield,
            monthly_PAR_agri=monthly_PAR_agri,
            monthly_PAR_openfield=monthly_PAR_openfield,
            cv_PAR=cv_PAR,
            hourly_par=hourly_par
        )
        results.append(res)
    
    # Sort: put best suited first, then by r_ann
    def sort_key(r):
        score = 0
        if "suitable" in r.suitability_class: score = 3
        elif "marginal" in r.suitability_class: score = 2
        elif "trial" in r.suitability_class: score = 1
        return (score, r.r_ann)

    results.sort(key=sort_key, reverse=True)
    return results
