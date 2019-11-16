import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from pathlib import Path
from hashlib import md5

import netCDF4 as nc
from datetime import datetime, timedelta
import pickle
import dill
import torch
import json


# Forcing and streamflow mean/std calculated over all basins in period 2000-01-01 until 2007-12-31
SCALER = {
    '2000-01-01': {
        '2007-12-31': {
            # TODO calculate
            '': {  # basin list hash, calculated as `md5(str(basin_list).encode('UTF-8')).hexdigest()`
                'input_means': np.array([]),
                'input_stds': np.array([]),
                'output_mean': np.array([]),
                'output_std': np.array([])
            }
        }
    }
}


def get_basin_list(data_root: Path, objectives: List, basin_type: str) -> List:
    """Returns a list of basin names for the given GRIP-GL objectives
    
    Parameters
    ----------
    data_root: Path
        Path to base data directory
    objectives: list
        List of objectives (as int)
    basin_type: str
        'C' to return calibration stations only, 'V' to return validation stations only, 
        '*' to return all stations
        
    Returns
    -------
    list
        list of basin name strings
    """
    if basin_type not in ['*', 'C', 'V']:
        raise ValueError('Illegal basin type')
        
    basins = np.array([], dtype=str)
    for o in objectives:
        if o not in [1, 2]:
            raise ValueError('Illegal GRIP objective.')
        gauge_info = pd.read_csv(data_root / f'gauge_info-obj{o}.csv')
        if basin_type != '*':
            gauge_info = gauge_info[gauge_info['Calibration/Validation'] == basin_type]
        basins = np.concatenate([basins, gauge_info['ID'].values])
        
    return np.unique(basins).tolist()        


def load_grip_gl_discharge(data_root: Path, basins: List = None,) -> pd.DataFrame:
    """Loads observed discharge for (calibration) gauging stations in GRIP-GL objective 1 & 2.
    
    Parameters
    ----------
    data_root: Path
        Path to base data directory
    basins: List
        Optional list of basins for which to return data. If None (default), all basins are returned.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns [date, basin, qobs], where 'qobs' contains the streamflow.
    """
    files = [data_root / 'grip-gl-calibration-obj1-all_gauges.nc',
             data_root / 'grip-gl-calibration-obj2-all_gauges.nc']
    
    data_streamflow = None
    for f in files:
        q_nc = nc.Dataset(f, 'r')
        obj_basins = q_nc['station_id'][:]
        time = nc.num2date(q_nc['time'][:], q_nc['time'].units, q_nc['time'].calendar)
        data = pd.DataFrame(q_nc['Q'][:].T, index=time, columns=obj_basins)
        q_nc.close()
        
        if data_streamflow is None:
            data_streamflow = data
        else:
            data = data[[s for s in obj_basins if s not in data_streamflow.columns]]
            data_streamflow = data_streamflow.join(data)
    
    data_streamflow = data_streamflow.loc['2000-01-01':'2016-12-31'].unstack().reset_index()\
            .rename({'level_0': 'basin', 'level_1': 'date', 0: 'qobs'}, axis=1)
    
    if basins is not None:
        print('x')
        data_streamflow = data_streamflow[data_streamflow['basin'].isin(basins)].reset_index(drop=True)
    return data_streamflow


def load_wfdei_gem_capa_forcings_lumped(data_root: Path, basins: List = None) -> Dict:
    """Loads basin-lumped WFDEI-GEM-CaPA forcings
    
    Parameters
    ----------
    data_root: Path
        Path to base data directory   
    basins: List
        List of basins for which to return data. Default (None) returns data for all basins.
    
    Returns
    -------
    dict
        Dictionary of forcings (pd.DataFrame) per basin
    """
    lumped_dir = data_root / 'lumped' / 'wfdei-gem-capa_forcing_ascii_by_gauge'
    basin_files = lumped_dir.glob('*.rvt')
    
    basin_forcings = {}
    for f in basin_files:
        basin = f.name.split('_')[-1][:-4]
        if basins is not None and basin not in basins:
            continue
            
        data = pd.read_csv(f, sep=',\s*', skiprows=4, skipfooter=1, 
                           header=None, usecols=range(4), engine='python')
        data.columns = ['precip', 'temp_daily_avg', 'temp_min', 'temp_max']
        data.index = pd.date_range('2000-01-01', periods=len(data), freq='D')
        basin_forcings[basin] = data
        
    return basin_forcings


def load_wfdei_gem_capa_forcings_gridded(data_root: Path, as_grid=False, include_lat_lon=False):
    """Loads gridded WFDEI-GEM-CaPA forcings.
    
    Loads hourly meteorological WFDEI-GEM-CaPA forcings.
    
    Parameters
    ----------
        data_root: Path
            Path to base data directory
        as_grid: bool, default False
            If False, will flatten returned forcing rows and columns into columns. 
        include_lat_lon: bool, default False
            If True and as_grid is True, will additionally return latitudes and longitudes of the forcing dataset.
    
    Returns
    -------
        If not as_grid: A pd.DataFrame with dates as index and one column per variable and forcing grid cell
        If as_grid: 
            A np.ndarray of shape (#timesteps, #vars, #rows, #cols) of forcing data
            A list of length #vars of variable names
            A pd.date_range of length #timesteps, and (if specified) lat and lon arrays)
            If include_lat_lon: An array of length #rows of latitudes and an array of length #cols of longitudes.
    """
    forcing_variables = ['hus', 'pr', 'ps', 'rlds', 'rsds_thresholded', 'wind_speed', 'ta']
    forcing_nc = nc.Dataset(data_root / 'grip-gl_wfdei-gem-capa_2000-2016_leap.nc', 'r')
    
    if as_grid:
        time_steps, nrows, ncols = forcing_nc[forcing_variables[0]].shape
        forcing_data = np.zeros((time_steps, len(forcing_variables), nrows, ncols))
        for i in range(len(forcing_variables)):
            forcing_data[:,i,:,:] = forcing_nc[forcing_variables[i]][:]
        
        if include_lat_lon:
            return_values = (forcing_data, forcing_variables, pd.Series(pd.date_range('1999-12-31 18:00', '2016-12-31 15:00', freq='3H')),
                             forcing_nc['lat'][:], forcing_nc['lon'][:])
        else:
            return_values = forcing_data, forcing_variables, pd.Series(pd.date_range('1999-12-31 18:00', '2016-12-31 15:00', freq='3H'))
        forcing_nc.close()
        return return_values
    else:
        # Using 18:00/15:00 because forcings are UTC, while streamflow is local time (we shift 6h back so it aligns with the 3h-steps)
        forcing_data = pd.DataFrame(index=pd.date_range('1999-12-31 18:00', '2016-12-31 15:00', freq='3H'))

        for var in forcing_variables:
            var_data = pd.DataFrame(forcing_nc[var][:].reshape(49680,86*171))
            var_data.columns = [var + '_' + str(c) for c in var_data.columns]
            forcing_data.reset_index(drop=True, inplace=True)
            forcing_data = forcing_data.reset_index(drop=True).join(var_data.reset_index(drop=True))
        forcing_data.index = pd.date_range('1999-12-31 18:00', '2016-12-31 15:00', freq='3H')

        forcing_nc.close()
        return forcing_data

    
def normalize_features(feature: np.ndarray, variable: str, 
                       start_date: str, end_date: str, basins: List) -> np.ndarray:
    """Normalize features using global pre-computed statistics.
    
    start_date, end_date, and basins are required to make sure we use the mean/std calculated over the training samples only.
    I.e., we don't want to use a mean/std that is calculated over train- & test-samples.
    
    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['input', 'output'], where `input` mean that the `feature` values are model inputs (forcings),
        and `output` means that the `feature` values are discharge.
    start_date: str
        Training period start date
    end_date: str
        Training period end date
    basins: List
        Training basins
        
    Returns
    -------
    np.ndarray
        Normalized features
        
    Raises
    ------
    RuntimeError
        If `variable` is neither 'input' nor 'output', or there is no precomputed mean/std for the passed date/basin combination.
    """
    if variable not in ['input', 'output']:
        raise RuntimeError(f"Unknown variable type {variable}")
    if start_date not in SCALER or end_date not in SCALER[start_date]:
        raise RuntimeError(f"No precomputed mean/std for {start_date}-{end_date}")
    
    basin_hash = md5(str(basins).encode('UTF-8')).hexdigest()
    if basin_hash not in SCALER[start_date][end_date]:
        raise RuntimeError(f"No precomputed mean/std for basins {basins}")
    
    scaler = SCALER[start_date][end_date][basin_hash]
    feature = (feature - scaler[f"{variable}_means"]) / scaler[f"{variable}_stds"]
    
    return feature