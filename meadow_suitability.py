"""
meadow_suitability.py – Suitability engine for wet meadow & floodplain species
=============================================================================
Uses Ellenberg/Landolt-derived rPAR thresholds for light assessment and
Ellenberg F/N values for hydrology/nutrient scoring.

IMPORTANT: These are derived screening thresholds from ecological indicator
values, NOT experimental PAR norms.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Load database
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "meadow_database.yaml")

with open(DB_PATH, "r", encoding="utf-8") as f:
    MEADOW_DB = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MeadowSpeciesProfile:
    id: str
    display_name: str
    botanical_name: str
    species_group: str  # grass, herb, floodplain
    use_type: str
    evidence_basis: str
    ellenberg_L: int
    ellenberg_F: int
    ellenberg_N: int
    rPAR_min: float
    rPAR_target: float
    DLI_min: float
    DLI_target: float
    peak_PPFD_min: float
    growing_season_months: List[int]
    critical_months: List[int]
    mowing_tolerance: str  # hoch, mittel, gering
    evidence_tier: str
    confidence_default: str
    source_references: List[str]
    notes_de: str
    warning_text: str


@dataclass
class MeadowSuitabilityResult:
    species_id: str
    display_name: str
    botanical_name: str
    species_group: str
    use_type: str
    # Scores
    score: float  # 0–100
    light_score: float
    hydro_score: float
    # Classifications
    light_class: str  # "lichtseitig geeignet" / "grenzwertig" / "nur in hellen Reihenabständen" / "nicht empfohlen"
    hydro_class: str  # "hydrologisch geeignet" / "hydrologisch prüfpflichtig" / "nur in feuchten Senken"
    zone_hint: str    # "gesamte Fläche" / "nur in hellen Reihenabständen / Gaps" / "nicht empfohlen"
    # Metrics
    rPAR_actual: float
    rPAR_min: float
    rPAR_target: float
    r_crit: float
    DLI_gs: float
    DLI_min: float
    DLI_target: float
    cv_PAR: float
    ellenberg_L: int
    ellenberg_F: int
    ellenberg_N: int
    mowing_tolerance: str
    # Evidence
    evidence_tier: str
    confidence: str
    evidence_basis: str
    limiting_factor: str
    explanation_de: str
    warning_text: str
    source_references: List[str]


# ---------------------------------------------------------------------------
# Build registry
# ---------------------------------------------------------------------------
MEADOW_REGISTRY: Dict[str, MeadowSpeciesProfile] = {}
for sid, entry in MEADOW_DB.items():
    MEADOW_REGISTRY[sid] = MeadowSpeciesProfile(
        id=sid,
        display_name=entry.get("display_name", sid),
        botanical_name=entry.get("botanical_name", ""),
        species_group=entry.get("species_group", "herb"),
        use_type=entry.get("use_type", ""),
        evidence_basis=entry.get("evidence_basis", ""),
        ellenberg_L=entry.get("ellenberg_L", 7),
        ellenberg_F=entry.get("ellenberg_F", 5),
        ellenberg_N=entry.get("ellenberg_N", 5),
        rPAR_min=entry.get("rPAR_min", 0.55),
        rPAR_target=entry.get("rPAR_target", 0.75),
        DLI_min=entry.get("DLI_min", 18),
        DLI_target=entry.get("DLI_target", 25),
        peak_PPFD_min=entry.get("peak_PPFD_min", 700),
        growing_season_months=entry.get("growing_season_months", [4,5,6,7,8,9]),
        critical_months=entry.get("critical_months", [5,6,7]),
        mowing_tolerance=entry.get("mowing_tolerance", "mittel"),
        evidence_tier=entry.get("evidence_tier", "C"),
        confidence_default=entry.get("confidence_default", "niedrig"),
        source_references=entry.get("source_references", []),
        notes_de=entry.get("notes_de", "").strip(),
        warning_text=entry.get("warning_text", ""),
    )


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def _score_light(rPAR_actual: float, rPAR_min: float, rPAR_target: float) -> float:
    """Score 0–100 based on annual relative PAR vs species thresholds."""
    if rPAR_actual >= rPAR_target:
        return 100.0
    elif rPAR_actual >= rPAR_min:
        # Linear interpolation between min (50) and target (100)
        frac = (rPAR_actual - rPAR_min) / (rPAR_target - rPAR_min)
        return 50.0 + frac * 50.0
    elif rPAR_actual >= rPAR_min * 0.8:
        # Below min but within 80% — marginal
        frac = (rPAR_actual - rPAR_min * 0.8) / (rPAR_min * 0.2)
        return 20.0 + frac * 30.0
    else:
        # Far below — scale 0–20
        return max(0.0, rPAR_actual / (rPAR_min * 0.8) * 20.0)


def _score_dli(DLI_actual: float, DLI_min: float, DLI_target: float) -> float:
    """Score 0–100 for daily light integral."""
    if DLI_actual >= DLI_target:
        return 100.0
    elif DLI_actual >= DLI_min:
        frac = (DLI_actual - DLI_min) / (DLI_target - DLI_min)
        return 50.0 + frac * 50.0
    elif DLI_actual >= DLI_min * 0.8:
        frac = (DLI_actual - DLI_min * 0.8) / (DLI_min * 0.2)
        return 20.0 + frac * 30.0
    else:
        return max(0.0, DLI_actual / (DLI_min * 0.8) * 20.0)


def _score_hydro(ellenberg_F: int) -> float:
    """
    Hydrology score: penalize high F values under Agri-PV since module
    foundations may alter drainage/water table.
    F <= 5: fully compatible (100)
    F == 6: slight concern (85)
    F == 7: moderate concern (65)
    F >= 8: strong concern (40)
    """
    if ellenberg_F <= 5:
        return 100.0
    elif ellenberg_F == 6:
        return 85.0
    elif ellenberg_F == 7:
        return 65.0
    else:  # F >= 8
        return 40.0


def _classify_light(score: float, rPAR_actual: float, rPAR_min: float) -> str:
    """Return English light classification."""
    if score >= 70:
        return "Light Suitable"
    elif score >= 50:
        return "Marginal Light"
    elif score >= 25:
        return "Inter-row Gaps Only"
    else:
        return "Not Suitable for Heavy Shade"


def _classify_hydro(ellenberg_F: int) -> str:
    """Return English hydrology classification."""
    if ellenberg_F <= 5:
        return "Hydrologically Suitable"
    elif ellenberg_F <= 7:
        return "Hydrology Verification Needed"
    else:
        return "Wet Depressions Only"


def _classify_zone(light_class: str, hydro_class: str, rPAR_actual: float, rPAR_min: float) -> str:
    """Combined zone recommendation."""
    if light_class == "Not Suitable for Heavy Shade":
        return "Not Recommended"
    elif light_class == "Inter-row Gaps Only":
        return "Inter-row Gaps Only"
    elif hydro_class == "Wet Depressions Only":
        return "Wet Depressions Only"
    elif light_class == "Marginal Light" or hydro_class == "Hydrology Verification Needed":
        return "Zone Dependent"
    else:
        return "Entire Field"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_meadow_species(
    species: MeadowSpeciesProfile,
    annual_PAR_agri: float,
    annual_PAR_openfield: float,
    monthly_PAR_agri: List[float],
    monthly_PAR_openfield: List[float],
    cv_PAR: float,
) -> MeadowSuitabilityResult:
    """Evaluate a single meadow species for Agri-PV suitability."""

    DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 1) Annual relative PAR
    rPAR_actual = annual_PAR_agri / annual_PAR_openfield if annual_PAR_openfield > 0 else 0.0

    # 2) Critical phase relative PAR
    if species.critical_months:
        crit_agri = sum(monthly_PAR_agri[m - 1] for m in species.critical_months)
        crit_open = sum(monthly_PAR_openfield[m - 1] for m in species.critical_months)
        r_crit = crit_agri / crit_open if crit_open > 0 else rPAR_actual
    else:
        r_crit = rPAR_actual

    # 3) Growing season DLI
    gs_m = species.growing_season_months
    if gs_m:
        gs_days = sum(DAYS_PER_MONTH[m - 1] for m in gs_m)
        gs_par = sum(monthly_PAR_agri[m - 1] for m in gs_m)
        DLI_gs = gs_par / gs_days if gs_days > 0 else 0.0
    else:
        DLI_gs = annual_PAR_agri / 365.0

    # 4) Light score (weighted: 60% annual rPAR, 40% DLI)
    s_rPAR = _score_light(rPAR_actual, species.rPAR_min, species.rPAR_target)
    s_DLI = _score_dli(DLI_gs, species.DLI_min, species.DLI_target)
    light_score = s_rPAR * 0.6 + s_DLI * 0.4

    # 5) Hydrology score
    hydro_score = _score_hydro(species.ellenberg_F)

    # 6) Homogeneity penalty
    cv_penalty = 0.0
    if cv_PAR > 0.30:
        cv_penalty = min(15.0, (cv_PAR - 0.30) * 100)

    # 7) Combined score (70% light, 20% hydro, 10% homogeneity)
    homo_score = max(0.0, 100.0 - cv_penalty * 10)
    score = light_score * 0.70 + hydro_score * 0.20 + homo_score * 0.10

    # 8) Classifications
    light_class = _classify_light(light_score, rPAR_actual, species.rPAR_min)
    hydro_class = _classify_hydro(species.ellenberg_F)
    zone_hint = _classify_zone(light_class, hydro_class, rPAR_actual, species.rPAR_min)

    # 9) Limiting factor
    if light_score < 50 and hydro_score < 60:
        limiting = "Light + Hydrology"
    elif light_score < 50:
        limiting = "Light (rPAR too low)"
    elif hydro_score < 60:
        limiting = "Hydrology (F>=8)"
    elif s_DLI < 50:
        limiting = "DLI (too low)"
    else:
        limiting = "None"

    # 10) Explanation
    explanation_parts = []
    if rPAR_actual >= species.rPAR_target:
        explanation_parts.append(f"Relative PAR {rPAR_actual*100:.0f}% >= Target {species.rPAR_target*100:.0f}%.")
    elif rPAR_actual >= species.rPAR_min:
        explanation_parts.append(f"Relative PAR {rPAR_actual*100:.0f}% between Minimum ({species.rPAR_min*100:.0f}%) and Target ({species.rPAR_target*100:.0f}%).")
    else:
        explanation_parts.append(f"Relative PAR {rPAR_actual*100:.0f}% below Minimum ({species.rPAR_min*100:.0f}%).")

    if species.ellenberg_F >= 7:
        explanation_parts.append(f"Moisture indicator F{species.ellenberg_F} — hydrological site verification required.")

    explanation_de = " ".join(explanation_parts)

    return MeadowSuitabilityResult(
        species_id=species.id,
        display_name=species.display_name,
        botanical_name=species.botanical_name,
        species_group=species.species_group,
        use_type=species.use_type,
        score=round(score, 1),
        light_score=round(light_score, 1),
        hydro_score=round(hydro_score, 1),
        light_class=light_class,
        hydro_class=hydro_class,
        zone_hint=zone_hint,
        rPAR_actual=round(rPAR_actual, 4),
        rPAR_min=species.rPAR_min,
        rPAR_target=species.rPAR_target,
        r_crit=round(r_crit, 4),
        DLI_gs=round(DLI_gs, 2),
        DLI_min=species.DLI_min,
        DLI_target=species.DLI_target,
        cv_PAR=round(cv_PAR, 4),
        ellenberg_L=species.ellenberg_L,
        ellenberg_F=species.ellenberg_F,
        ellenberg_N=species.ellenberg_N,
        mowing_tolerance=species.mowing_tolerance,
        evidence_tier=species.evidence_tier,
        confidence=species.confidence_default,
        evidence_basis=species.evidence_basis,
        limiting_factor=limiting,
        explanation_de=explanation_de,
        warning_text=species.warning_text,
        source_references=species.source_references,
    )


def evaluate_all_meadow_species(
    annual_PAR_agri: float,
    annual_PAR_openfield: float,
    monthly_PAR_agri: List[float],
    monthly_PAR_openfield: List[float],
    cv_PAR: float,
) -> List[MeadowSuitabilityResult]:
    """Evaluate all meadow species, return sorted by score descending."""
    results = []
    for sid, sp in MEADOW_REGISTRY.items():
        r = evaluate_meadow_species(
            sp,
            annual_PAR_agri,
            annual_PAR_openfield,
            monthly_PAR_agri,
            monthly_PAR_openfield,
            cv_PAR,
        )
        results.append(r)

    results.sort(key=lambda x: x.score, reverse=True)
    return results
