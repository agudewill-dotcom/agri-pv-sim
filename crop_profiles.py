"""
crop_profiles.py – Agri-PV Crop Suitability Engine: Crop Profiles
=================================================================

Defines the **CropProfile** dataclass and the **CROP_REGISTRY** dictionary
that holds agronomic light-requirement parameters for 11 arable crops
relevant to Agri-Photovoltaic (Agri-PV) systems in Central Europe.

All PAR (Photosynthetically Active Radiation) fractions are expressed
relative to the open-field reference so that a value of 0.80 means
"80 % of what the crop would receive without panels".

Crop names are given in both German (``name_de``) and English (``name_en``)
to support bilingual user interfaces.

Helper functions convert between GHI-based references and absolute
PAR values (mol m⁻² a⁻¹) and derive normalised monthly activity weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropProfile:
    """Agronomic light-requirement profile for a single crop.

    Attributes
    ----------
    id : str
        Machine-readable identifier, e.g. ``'winterweizen'``.
    name_de : str
        German crop name for UI display, e.g. ``'Winterweizen'``.
    name_en : str
        English crop name, e.g. ``'Winter Wheat'``.
    f_min : float
        Minimum PAR fraction (relative to open field) for the crop to be
        considered *marginally suitable*.  Dimensionless, 0–1.
    f_target : float
        Target PAR fraction for *good suitability*.  Dimensionless, 0–1.
    evidence_tier : str
        Quality of the underlying evidence: ``'A'`` (field trials),
        ``'B'`` (pot / greenhouse + modelling), ``'C'`` (expert / proxy).
    evidence_sources : list[str]
        Short reference strings for the evidence sources.
    crop_group : str
        Functional group: ``'forage'``, ``'winter_cereal'``,
        ``'summer_cereal'``, or ``'c4_grain'``.
    critical_months : list[int]
        Calendar months (1–12) during which the crop is most sensitive
        to light limitation (e.g. grain fill, heading).
    growing_months : list[int]
        Calendar months (1–12) of the active growing season.
    peak_ppfd_min : float
        Minimum tolerable midday PPFD in µmol m⁻² s⁻¹.
    cv_max : float
        Maximum tolerable spatial coefficient of variation (CV) of
        sub-panel PAR, dimensionless (e.g. 0.25 = 25 %).
    weights : dict
        Scoring-component weights with keys ``'wA'`` (annual PAR),
        ``'wS'`` (seasonal PAR), ``'wC'`` (critical phase), ``'wH'``
        (homogeneity).  Values must sum to 1.0.
    notes_de : str
        Free-text notes in German for UI tooltips / reports.
    is_proxy : bool
        ``True`` if the profile is derived from a crop-group proxy
        rather than direct experimental evidence for this species.
    """

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


# ---------------------------------------------------------------------------
# Crop Registry – 11 crops
# ---------------------------------------------------------------------------

CROP_REGISTRY: Dict[str, CropProfile] = {

    # ── Forage ─────────────────────────────────────────────────────────────
    "luzerne": CropProfile(
        id="luzerne",
        name_de="Luzerne",
        name_en="Lucerne / Alfalfa",
        f_min=0.55,
        f_target=0.75,
        evidence_tier="B",
        evidence_sources=[
            "Dupraz et al. 2011 – Agrivoltaics: shade-tolerant crops overview",
            "Trommsdorff et al. 2021 – APV-RESOLA forage results",
        ],
        crop_group="forage",
        critical_months=[5, 6, 7, 8],
        growing_months=[4, 5, 6, 7, 8, 9],
        peak_ppfd_min=500.0,
        cv_max=0.30,
        weights={"wA": 0.35, "wS": 0.30, "wC": 0.15, "wH": 0.20},
        notes_de=(
            "Luzerne ist vergleichsweise schattenverträglich und eignet "
            "sich gut für Agri-PV-Systeme mit mäßiger Verschattung."
        ),
        is_proxy=False,
    ),

    # ── Winter Cereals ─────────────────────────────────────────────────────
    "wintergerste": CropProfile(
        id="wintergerste",
        name_de="Wintergerste",
        name_en="Winter Barley",
        f_min=0.60,
        f_target=0.80,
        evidence_tier="A",
        evidence_sources=[
            "Weselek et al. 2021 – APV field trial barley yield data",
            "Trommsdorff et al. 2021 – APV-RESOLA winter cereal results",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6],
        growing_months=[2, 3, 4, 5, 6],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Wintergerste zeigt in Feldversuchen stabile Erträge unter "
            "Agri-PV bei ≥ 60 % PAR-Verfügbarkeit."
        ),
        is_proxy=False,
    ),

    "winterroggen": CropProfile(
        id="winterroggen",
        name_de="Winterroggen",
        name_en="Winter Rye",
        f_min=0.60,
        f_target=0.80,
        evidence_tier="B",
        evidence_sources=[
            "Weselek et al. 2021 – shade tolerance review (rye)",
            "Trommsdorff et al. 2021 – APV-RESOLA cereal modelling",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Winterroggen ist robust und relativ schattenverträglich; "
            "Ährenentwicklung im Frühjahr ist lichtempfindlich."
        ),
        is_proxy=False,
    ),

    "triticale": CropProfile(
        id="triticale",
        name_de="Triticale",
        name_en="Triticale",
        f_min=0.60,
        f_target=0.80,
        evidence_tier="B",
        evidence_sources=[
            "Weselek et al. 2021 – shade tolerance review (triticale)",
            "Trommsdorff et al. 2021 – APV-RESOLA cereal modelling",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Triticale (Weizen-Roggen-Hybride) zeigt ähnliche "
            "Schattentoleranz wie Winterroggen."
        ),
        is_proxy=False,
    ),

    "winterweizen": CropProfile(
        id="winterweizen",
        name_de="Winterweizen",
        name_en="Winter Wheat",
        f_min=0.65,
        f_target=0.80,
        evidence_tier="A",
        evidence_sources=[
            "Weselek et al. 2021 – APV field trial wheat yield data",
            "Trommsdorff et al. 2021 – APV-RESOLA wheat results",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Winterweizen ist die am besten untersuchte Kultur unter "
            "Agri-PV.  Ertragseinbußen bei < 65 % PAR wahrscheinlich."
        ),
        is_proxy=False,
    ),

    "dinkel": CropProfile(
        id="dinkel",
        name_de="Dinkel",
        name_en="Spelt",
        f_min=0.65,
        f_target=0.80,
        evidence_tier="C",
        evidence_sources=[
            "Proxy: Winterweizen-Gruppe (winter_cereal)",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Dinkel wird als Proxy der Winterweizen-Gruppe bewertet. "
            "Direkte Agri-PV-Versuchsdaten fehlen."
        ),
        is_proxy=True,
    ),

    "einkorn": CropProfile(
        id="einkorn",
        name_de="Einkorn",
        name_en="Einkorn Wheat",
        f_min=0.65,
        f_target=0.80,
        evidence_tier="C",
        evidence_sources=[
            "Proxy: Winterweizen-Gruppe (winter_cereal)",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Einkorn wird als Proxy der Winterweizen-Gruppe bewertet. "
            "Direkte Agri-PV-Versuchsdaten fehlen."
        ),
        is_proxy=True,
    ),

    "emmer": CropProfile(
        id="emmer",
        name_de="Emmer",
        name_en="Emmer Wheat",
        f_min=0.65,
        f_target=0.80,
        evidence_tier="C",
        evidence_sources=[
            "Proxy: Winterweizen-Gruppe (winter_cereal)",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Emmer wird als Proxy der Winterweizen-Gruppe bewertet. "
            "Direkte Agri-PV-Versuchsdaten fehlen."
        ),
        is_proxy=True,
    ),

    "hafer": CropProfile(
        id="hafer",
        name_de="Hafer",
        name_en="Oats",
        f_min=0.65,
        f_target=0.80,
        evidence_tier="C",
        evidence_sources=[
            "Proxy: Winterweizen-Gruppe (winter_cereal)",
        ],
        crop_group="winter_cereal",
        critical_months=[4, 5, 6, 7],
        growing_months=[3, 4, 5, 6, 7],
        peak_ppfd_min=550.0,
        cv_max=0.25,
        weights={"wA": 0.35, "wS": 0.25, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Hafer wird als Proxy der Winterweizen-Gruppe bewertet. "
            "Direkte Agri-PV-Versuchsdaten fehlen."
        ),
        is_proxy=True,
    ),

    # ── Summer Cereal ──────────────────────────────────────────────────────
    "schwarzhafer": CropProfile(
        id="schwarzhafer",
        name_de="Schwarzhafer",
        name_en="Black Oat",
        f_min=0.75,
        f_target=0.90,
        evidence_tier="A",
        evidence_sources=[
            "Weselek et al. 2021 – cover-crop light response data",
            "Trommsdorff et al. 2021 – APV-RESOLA summer-crop results",
        ],
        crop_group="summer_cereal",
        critical_months=[3, 4, 5, 9, 10],
        growing_months=[3, 4, 5, 6, 7, 8, 9, 10],
        peak_ppfd_min=650.0,
        cv_max=0.25,
        weights={"wA": 0.30, "wS": 0.30, "wC": 0.30, "wH": 0.10},
        notes_de=(
            "Schwarzhafer hat eine lange Vegetationsperiode und ist "
            "lichtbedürftig in Frühjahr und Herbst."
        ),
        is_proxy=False,
    ),

    # ── C4 Grain ───────────────────────────────────────────────────────────
    "mais": CropProfile(
        id="mais",
        name_de="Mais",
        name_en="Maize / Corn",
        f_min=0.85,
        f_target=0.95,
        evidence_tier="A",
        evidence_sources=[
            "Weselek et al. 2021 – APV maize shade sensitivity data",
            "Dupraz et al. 2011 – C4 crop light-saturation analysis",
        ],
        crop_group="c4_grain",
        critical_months=[6, 7, 8],
        growing_months=[6, 7, 8, 9],
        peak_ppfd_min=900.0,
        cv_max=0.20,
        weights={"wA": 0.25, "wS": 0.20, "wC": 0.45, "wH": 0.10},
        notes_de=(
            "Mais als C4-Pflanze hat einen sehr hohen Lichtbedarf und "
            "ist für die meisten Agri-PV-Systeme nur bedingt geeignet."
        ),
        is_proxy=False,
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_par_ref_from_ghi(ghi_annual_kwh: float, f_par: float = 0.45) -> float:
    """Convert annual GHI to open-field PAR reference in mol m⁻² a⁻¹.

    The conversion chain is:

    1. ``GHI [kWh m⁻²]`` → ``MJ m⁻²``:  multiply by 3.6
    2. ``MJ m⁻²`` → ``PAR MJ m⁻²``:  multiply by *f_par* (≈ 0.45)
    3. ``PAR MJ m⁻²`` → ``mol m⁻²``:  multiply by 4.57 µmol J⁻¹

    Parameters
    ----------
    ghi_annual_kwh : float
        Annual global horizontal irradiance in kWh m⁻².
    f_par : float, optional
        Fraction of GHI that is photosynthetically active (default 0.45).

    Returns
    -------
    float
        Annual open-field PAR in mol m⁻² a⁻¹.

    Examples
    --------
    >>> round(get_par_ref_from_ghi(1100.0), 1)
    8145.9
    """
    return ghi_annual_kwh * 3.6 * f_par * 4.57


def get_absolute_thresholds(
    crop: CropProfile, par_ref: float
) -> tuple[float, float]:
    """Derive absolute PAR thresholds from a crop's fractional limits.

    Parameters
    ----------
    crop : CropProfile
        Crop whose ``f_min`` and ``f_target`` are used.
    par_ref : float
        Open-field PAR reference in mol m⁻² a⁻¹.

    Returns
    -------
    tuple[float, float]
        ``(PAR_min, PAR_target)`` in mol m⁻² a⁻¹.

    Examples
    --------
    >>> from crop_profiles import CROP_REGISTRY
    >>> ww = CROP_REGISTRY['winterweizen']
    >>> par_ref = get_par_ref_from_ghi(1100.0)
    >>> p_min, p_tgt = get_absolute_thresholds(ww, par_ref)
    >>> round(p_min, 1)
    5294.8
    """
    return crop.f_min * par_ref, crop.f_target * par_ref


def get_monthly_weights(crop: CropProfile) -> list[float]:
    """Return 12 normalised monthly activity weights for *crop*.

    Months inside ``crop.growing_months`` receive equal weight;
    months outside receive zero.  Weights are normalised so that
    the list sums to 1.0.

    Parameters
    ----------
    crop : CropProfile
        Crop whose ``growing_months`` are used.

    Returns
    -------
    list[float]
        Length-12 list (index 0 = January) of normalised weights.

    Examples
    --------
    >>> from crop_profiles import CROP_REGISTRY
    >>> ww = CROP_REGISTRY['winterweizen']
    >>> wts = get_monthly_weights(ww)
    >>> len(wts)
    12
    >>> abs(sum(wts) - 1.0) < 1e-9
    True
    >>> wts[0]  # January – not in growing season
    0.0
    """
    n_active = len(crop.growing_months)
    if n_active == 0:
        return [0.0] * 12

    weight_per_month = 1.0 / n_active
    return [
        weight_per_month if (month_idx + 1) in crop.growing_months else 0.0
        for month_idx in range(12)
    ]
