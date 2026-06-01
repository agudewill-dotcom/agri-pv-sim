import numpy as np
import pandas as pd
import streamlit as st
import simulation
import geometry
import irradiance

@st.cache_data
def compute_spatial_grid_2d(config, res_a, resolution=0.5, field_length=20.0):
    """
    Computes a 2D spatio-temporal simulation grid of PAR and shading.
    
    Coordinates:
    - X axis: Parallel to PV rows (from -field_length/2 to field_length/2)
    - Y axis: Cross-row pitch direction (from 0 to 2 * pitch)
    
    Returns spatial matrices and KPI dictionary.
    """
    pitch = config['pitch']
    tau = config['tau']
    g_slope = config['g_slope']
    g_aspect = config['g_aspect']
    albedo = config['albedo']
    tilt = config.get('tilt', 15.0)
    clearance = 2.10
    
    # 1. Define 2D Grid
    x_coords = np.arange(-field_length/2, field_length/2 + resolution, resolution)
    y_coords = np.arange(0, 2 * pitch + resolution, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    # Shape of X and Y: (len(y), len(x))
    ny, nx = X.shape
    
    # 2. Extract Time Series
    elev = res_a['elevation'].values
    az = res_a['azimuth'].values
    dni = res_a['dni'].values
    dhi = res_a['dhi'].values
    ghi = res_a['ghi'].values
    
    # Calculate geometric details
    geo = geometry.calculate_derived_geometry(tilt, length=5.63, clearance=clearance)
    proj_w = geo['projected_width']
    h_top = geo['top_edge_height']
    
    # 3. Vectorized shadow offset projection (hourly)
    # This is how far the shadow shifts along the Y axis
    valid = elev > 0.5
    elev_rad = np.radians(elev[valid])
    slope_rad = np.radians(g_slope)
    az_diff_rad = np.radians(az[valid] - g_aspect)
    
    # Y-axis shift (cross row)
    tan_slope_eff_y = np.tan(slope_rad) * np.cos(az_diff_rad)
    denom_y = np.tan(elev_rad) + tan_slope_eff_y
    denom_y_mask = denom_y > 0.01
    
    sh_len_y = np.zeros_like(elev)
    sh_len_y[valid] = np.where(denom_y_mask, h_top / np.where(denom_y_mask, denom_y, 1.0) / np.cos(slope_rad), 1e6)
    sh_len_y[~valid] = 1e6
    
    # X-axis shift (along row)
    az_rad = np.radians(az[valid] - 180.0) # assuming rows face South (180)
    sh_len_x = np.zeros_like(elev)
    # Simplified X shadow projection
    sh_len_x[valid] = h_top / np.tan(elev_rad) * np.sin(az_rad)
    sh_len_x[~valid] = 1e6

    # 4. Shadow Polygon Overlaps
    # For a row centered at y=row_y, x from -L/2 to L/2
    # Shadow box bounds at hour t:
    # y_min(t) = row_y - proj_w/2 + sh_len_y(t)
    # y_max(t) = row_y + proj_w/2 + sh_len_y(t)
    # x_min(t) = -L/2 + sh_len_x(t)
    # x_max(t) = L/2 + sh_len_x(t)
    
    # We will simulate 3 rows to capture overlap: y = 0, pitch, 2*pitch
    row_centers_y = [0.0, pitch, 2.0 * pitch]
    row_length = field_length
    
    # 5. Broadcast memory-efficient calculation
    # We'll calculate annual/seasonal sums by looping over time blocks to save RAM
    # Or just loop over hours (8760 iterations is very fast for 2D numpy arrays)
    
    par_agri = np.zeros((ny, nx))
    par_open = np.zeros((ny, nx))
    shadow_freq = np.zeros((ny, nx))
    
    # Seasonal accumulators
    months = res_a.index.month.values
    # 1=Spring (M3,4,5), 2=Summer (M6,7,8), 3=Autumn (M9,10,11), 4=Winter (M12,1,2)
    seasons = np.where(np.isin(months, [3, 4, 5]), 1, 
              np.where(np.isin(months, [6, 7, 8]), 2, 
              np.where(np.isin(months, [9, 10, 11]), 3, 4)))
              
    par_season = {1: np.zeros((ny, nx)), 2: np.zeros((ny, nx)), 3: np.zeros((ny, nx)), 4: np.zeros((ny, nx))}
    
    # Background diff/refl (uniform for simplicity, tracking spatial diff is highly complex)
    tau_eff = max(0, (proj_w - 0.81) / proj_w) * tau if proj_w > 0 else 0
    svf_f = irradiance.sky_view_factor_periodic(h_top, proj_w, pitch, tau_eff, h_clearance=clearance)
    aoi_ground = irradiance.calculate_incidence_angle(res_a['zenith'], res_a['azimuth'], g_slope, g_aspect).values
    cos_g = np.cos(np.radians(aoi_ground))
    
    g_diff = dhi * svf_f * (1.0 + np.cos(np.radians(g_slope))) / 2.0
    g_refl = ghi * albedo * (1.0 - np.cos(np.radians(g_slope))) / 2.0 + ghi * albedo * (1.0 - svf_f) * 0.15
    g_diff_open = dhi * (1.0 + np.cos(np.radians(g_slope))) / 2.0
    g_refl_open = ghi * albedo * (1.0 - np.cos(np.radians(g_slope))) / 2.0
    
    total_daylight_hours = np.sum(valid)
    
    for t in range(len(elev)):
        if not valid[t]: continue
        
        # Determine shaded mask for the 2D grid at this hour
        is_shaded = np.zeros((ny, nx), dtype=bool)
        
        for row_y in row_centers_y:
            y_min = row_y - proj_w/2 + sh_len_y[t]
            y_max = row_y + proj_w/2 + sh_len_y[t]
            x_min = -row_length/2 + sh_len_x[t]
            x_max = row_length/2 + sh_len_x[t]
            
            row_mask = (Y >= y_min) & (Y <= y_max) & (X >= x_min) & (X <= x_max)
            is_shaded |= row_mask
            
        t_beam = np.where(is_shaded, tau, 1.0)
        
        # Irradiance
        g_beam = dni[t] * max(0.0, cos_g[t]) * t_beam
        g_ground = g_beam + g_diff[t] + g_refl[t]
        
        g_beam_open = dni[t] * max(0.0, cos_g[t])
        g_ground_open = g_beam_open + g_diff_open[t] + g_refl_open[t]
        
        # PAR mol/m2
        par_h = g_ground * 0.45 * 4.57 * 3600.0 / 1e6
        par_o = g_ground_open * 0.45 * 4.57 * 3600.0 / 1e6
        
        par_agri += par_h
        par_open += par_o
        shadow_freq += is_shaded
        
        season = seasons[t]
        par_season[season] += par_h

    # Convert shadow frequency to percentage of daylight hours
    shadow_freq = (shadow_freq / total_daylight_hours) * 100.0
    
    # Calculate derived layers
    rem_par = (par_agri / par_open) * 100.0
    par_loss = 100.0 - rem_par
    
    # Calculate spatial KPIs on the central block to avoid edge effects
    # Central block: Y between 0.5*pitch and 1.5*pitch, X between -L/4 and L/4
    mask_kpi = (Y >= 0.5*pitch) & (Y <= 1.5*pitch) & (X >= -field_length/4) & (X <= field_length/4)
    if not np.any(mask_kpi):
        mask_kpi = np.ones_like(Y, dtype=bool) # fallback
        
    rem_par_kpi = rem_par[mask_kpi]
    
    kpis = {
        'mean_rem': np.mean(rem_par_kpi),
        'median_rem': np.median(rem_par_kpi),
        'min_rem': np.min(rem_par_kpi),
        'max_rem': np.max(rem_par_kpi),
        'p10_rem': np.percentile(rem_par_kpi, 10),
        'p90_rem': np.percentile(rem_par_kpi, 90),
        'std_rem': np.std(rem_par_kpi),
        'cv_rem': np.std(rem_par_kpi) / np.mean(rem_par_kpi) if np.mean(rem_par_kpi) > 0 else 0,
        'below_50_pct': np.mean(rem_par_kpi < 50.0) * 100.0,
        'below_60_pct': np.mean(rem_par_kpi < 60.0) * 100.0,
        'mean_shadow_freq': np.mean(shadow_freq[mask_kpi])
    }
    
    # Add PV module overlay bounding boxes for visualizations
    pv_rects = []
    for row_y in row_centers_y:
        pv_rects.append({
            'x0': -row_length/2,
            'x1': row_length/2,
            'y0': row_y - proj_w/2,
            'y1': row_y + proj_w/2
        })
        
    layers = {
        'X': X, 'Y': Y,
        'par_agri': par_agri,
        'par_open': par_open,
        'rem_par': rem_par,
        'par_loss': par_loss,
        'shadow_freq': shadow_freq,
        'par_spring': par_season[1],
        'par_summer': par_season[2],
        'par_autumn': par_season[3],
        'par_winter': par_season[4],
        'pv_rects': pv_rects
    }
    
    return layers, kpis
