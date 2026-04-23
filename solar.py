import requests
import pandas as pd
import io
import numpy as np
from pvlib.location import Location

def fetch_pvgis_hourly(lat, lon, start_year, end_year):
    """
    Fetches hourly irradiance data from PVGIS 5.2 API.
    Ref: https://re.jrc.ec.europa.eu/api/v5_2/seriescalc
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        'lat': lat,
        'lon': lon,
        'startyear': start_year,
        'endyear': end_year,
        'outputformat': 'json',
        'components': 1,
        'angle': 0, # Horizontal plane for GHI components
        'aspect': 0
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    hourly_data = data['outputs']['hourly']
    
    df = pd.DataFrame(hourly_data)
    
    # Format time: 20240101:0010 -> pd.Timestamp
    # PVGIS uses HHMM format where MM is usually 10 (minutes)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
    df = df.set_index('time')
    
    # PVGIS columns: G(i) is global, Gb(i) is beam, Gd(i) is diffuse on horizontal (if horizontal requested)
    # Actually seriescalc output for horizontal:
    # Gb(n): DNI
    # Renaming based on actual response keys for horizontalplane
    # Gb(i) is beam on horizontal, Gd(i) is diffuse on horizontal, H_sun is elevation
    mapping = {
        'Gb(i)': 'g_beam_horiz',
        'Gd(i)': 'dhi',
        'T2m': 'temp_air',
        'WS10m': 'wind_speed',
        'H_sun': 'pvgis_elevation'
    }
    df = df.rename(columns=mapping)
    
    # Deriving DNI and GHI
    # elevation in degrees, DNI = beam_horiz / sin(elevation)
    elev_rad = np.radians(df['pvgis_elevation'])
    sin_elev = np.sin(elev_rad)
    
    # Avoid division by zero at sunrise/sunset/night
    df['dni'] = np.where(df['pvgis_elevation'] > 0.5, df['g_beam_horiz'] / sin_elev, 0)
    df['ghi'] = df['g_beam_horiz'] + df['dhi']
    
    # Ensure numerical stability
    df = df.fillna(0)
    
    return df

def get_solar_position_df(lat, lon, timestamps):
    """
    Computes solar elevation and azimuth using pvlib.
    """
    loc = Location(lat, lon)
    solpos = loc.get_solarposition(timestamps)
    return solpos[['elevation', 'azimuth', 'zenith']]
