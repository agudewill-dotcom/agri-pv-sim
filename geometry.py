import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# ==============================================================================
# GEOMETRY PRESET REGISTRY (SUNfarming Cross-Section Definitions)
# ==============================================================================
GEOMETRY_PRESETS: Dict[str, Dict[str, Any]] = {
    "Predefined Table 12°": {
        "source_label": "SUNfarming Schnitt Agri-PV 12° / 8.28 m Pitch",
        "tilt_deg": 12.0,
        "clear_height_m": 2.70,
        "surface_azimuth_deg": 180.0,
        "table_length_m": 5.75,
        "row_pitch_m": 8.28,
        "structural_loss_percent": 0.0,
        "drawing_reference": {
            "table_projected_width_m": 5.62,
            "ground_gap_m": 2.63,
            "high_edge_height_m": 3.89,
            "note": "Referenz Schnitt 12°: LH 2,70 m | Tischlänge ca. 5,75 m | horizontale Projektion ca. 5,62 m | Pitch ca. 8,28 m | freier Gap ca. 2,63 m."
        }
    },
    "Predefined Table 15°": {
        "source_label": "SUNfarming / existing 15° default table",
        "tilt_deg": 15.0,
        "clear_height_m": 2.10,
        "surface_azimuth_deg": 180.0,
        "table_length_m": 5.63,
        "row_pitch_m": 8.63,
        "structural_loss_percent": 0.0,
        "drawing_reference": None
    }
}

# Legacy fallback defaults
DEFAULT_MODULE_LENGTH = 5.63
DEFAULT_LOWER_CLEARANCE = 2.10
DEFAULT_PITCH = 8.63
DEFAULT_FREE_GAP = 3.00


# ==============================================================================
# TABLE GEOMETRY DATA CLASS
# ==============================================================================
@dataclass
class TableGeometry:
    geometry_mode: str = "Predefined Table 12°"
    tilt_deg: float = 12.0
    clear_height_m: float = 2.70
    surface_azimuth_deg: float = 180.0
    table_length_m: float = 5.75
    row_pitch_m: float = 8.28
    structural_loss_percent: float = 0.0
    source_label: str = "SUNfarming Schnitt Agri-PV 12° / 8.28 m Pitch"

    @property
    def tilt_rad(self) -> float:
        return float(np.radians(self.tilt_deg))

    @property
    def table_projected_width_m(self) -> float:
        return float(self.table_length_m * np.cos(self.tilt_rad))

    @property
    def table_vertical_rise_m(self) -> float:
        return float(self.table_length_m * np.sin(self.tilt_rad))

    @property
    def h_low_m(self) -> float:
        return float(self.clear_height_m)

    @property
    def h_high_m(self) -> float:
        return float(self.clear_height_m + self.table_vertical_rise_m)

    @property
    def ground_gap_m(self) -> float:
        return float(self.row_pitch_m - self.table_projected_width_m)

    @property
    def ground_coverage_ratio(self) -> float:
        if self.row_pitch_m <= 0:
            return 0.0
        return float(self.table_projected_width_m / self.row_pitch_m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "geometry_mode": self.geometry_mode,
            "source_label": self.source_label,
            "tilt_deg": self.tilt_deg,
            "clear_height_m": self.clear_height_m,
            "surface_azimuth_deg": self.surface_azimuth_deg,
            "table_length_m": self.table_length_m,
            "table_projected_width_m": self.table_projected_width_m,
            "table_vertical_rise_m": self.table_vertical_rise_m,
            "h_low_m": self.h_low_m,
            "h_high_m": self.h_high_m,
            "row_pitch_m": self.row_pitch_m,
            "ground_gap_m": self.ground_gap_m,
            "ground_coverage_ratio": self.ground_coverage_ratio,
            "structural_loss_percent": self.structural_loss_percent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TableGeometry":
        if not data:
            preset = GEOMETRY_PRESETS["Predefined Table 15°"]
            return cls(
                geometry_mode="Predefined Table 15°",
                tilt_deg=preset["tilt_deg"],
                clear_height_m=preset["clear_height_m"],
                surface_azimuth_deg=preset["surface_azimuth_deg"],
                table_length_m=preset["table_length_m"],
                row_pitch_m=preset["row_pitch_m"],
                structural_loss_percent=preset.get("structural_loss_percent", 0.0),
                source_label=preset["source_label"]
            )
        
        return cls(
            geometry_mode=data.get("geometry_mode", "Custom Table Geometry"),
            tilt_deg=float(data.get("tilt_deg", 12.0)),
            clear_height_m=float(data.get("clear_height_m", 2.70)),
            surface_azimuth_deg=float(data.get("surface_azimuth_deg", 180.0)),
            table_length_m=float(data.get("table_length_m", 5.75)),
            row_pitch_m=float(data.get("row_pitch_m", 8.28)),
            structural_loss_percent=float(data.get("structural_loss_percent", 0.0)),
            source_label=data.get("source_label", "Custom")
        )


def calculate_derived_geometry(tilt_degrees, length=DEFAULT_MODULE_LENGTH, clearance=DEFAULT_LOWER_CLEARANCE):
    """Legacy compatibility helper."""
    tilt_rad = np.radians(tilt_degrees)
    projected_width = length * np.cos(tilt_rad)
    top_edge_height = clearance + length * np.sin(tilt_rad)
    return {
        'projected_width': float(projected_width),
        'top_edge_height': float(top_edge_height)
    }


def get_module_bounds(pitch, projected_width):
    """Legacy compatibility helper: returns (x_start, x_end) for module."""
    margin = (pitch - projected_width) / 2
    return float(margin), float(margin + projected_width)
