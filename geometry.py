import numpy as np

# Fixed Construction Defaults
DEFAULT_MODULE_LENGTH = 5.79
DEFAULT_LOWER_CLEARANCE = 2.10
DEFAULT_PITCH = 8.63
DEFAULT_FREE_GAP = 3.00

def calculate_derived_geometry(tilt_degrees, length=DEFAULT_MODULE_LENGTH, clearance=DEFAULT_LOWER_CLEARANCE):
    """
    Computes projected width and top edge height.
    """
    tilt_rad = np.radians(tilt_degrees)
    
    projected_width = length * np.cos(tilt_rad)
    top_edge_height = clearance + length * np.sin(tilt_rad)
    
    return {
        'projected_width': projected_width,
        'top_edge_height': top_edge_height
    }
