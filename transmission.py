import numpy as np

def calculate_avg_direct_transmission(projected_width, tau, free_gap, shadow_length, pitch):
    """
    Computes average direct transmission across one pitch.
    Incorporates the partially transparent modules and the unshaded gap.
    
    T_dir_avg = (projected_width * tau + max(0, free_gap - shadow_length)) / pitch
    """
    if pitch <= 0:
        return 0.0
    
    # Contribution from the modules (transmission tau)
    contribution_modules = projected_width * tau
    
    # Contribution from the gap (1.0 transmission where not shaded)
    unshaded_gap_width = max(0, free_gap - shadow_length)
    contribution_gap = unshaded_gap_width * 1.0
    
    t_avg = (contribution_modules + contribution_gap) / pitch
    
    return np.clip(t_avg, 0.0, 1.0)

def calculate_ground_irradiance(g_beam_horiz, t_dir_avg, g_diff_horiz, transmission_diffuse):
    """
    Compute total ground irradiance under the PV system.
    G_ground = G_beam_horiz * T_dir_avg + G_diff_horiz * transmission_diffuse
    """
    return g_beam_horiz * t_dir_avg + g_diff_horiz * transmission_diffuse

def calculate_par(g_ground, factor=2.1):
    """
    Converts irradiance (W/m2) to PAR (umol/m2/s).
    """
    return g_ground * factor
