from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from datautils import (load_grip_gl_discharge,
                       load_wfdei_gem_capa_forcings_lumped,
                       normalize_features)


class LumpedBasin(Dataset):
    """PyTorch data set to work with the raw text files for lumped (daily basin-aggregated) forcings and streamflow.
       
    Parameters
    ----------
    data_root : Path
        Path to the main directory of the data set
    basin : str
        Gauge-id of the basin
    dates : List
        Start and end date of the period.
    is_train : bool
        If True, discharge observations are normalized and invalid discharge samples are removed
    seq_length : int, optional
        Length of the input sequence
    with_attributes : bool, optional
        If True, loads and returns addtionaly attributes, by default False
    attribute_means : pd.Series, optional
        Means of catchment characteristics, used to normalize during inference, by default None
    attribute_stds : pd.Series, optional
        Stds of catchment characteristics, used to normalize during inference, by default None
    concat_static : bool, optional
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    db_path : str, optional
        Path to sqlite3 database file containing the catchment characteristics, by default None
    """

    def __init__(self,
                 data_root: Path,
                 basin: str,
                 dates: List,
                 is_train: bool,
                 seq_length: int,
                 with_attributes: bool = False,
                 attribute_means: pd.Series = None,
                 attribute_stds: pd.Series = None,
                 concat_static: bool = False,
                 db_path: str = None):
        self.data_root = data_root
        self.basin = basin
        self.seq_length = seq_length
        self.is_train = is_train
        self.dates = dates
        self.with_attributes = with_attributes
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.concat_static = concat_static
        self.db_path = db_path

        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        self.x, self.y = self._load_data()

        if self.with_attributes:
            self.attributes = self._load_attributes()

        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.with_attributes:
            if self.concat_static:
                x = torch.cat([self.x[idx], self.attributes.repeat((self.seq_length, 1))], dim=-1)
                return x, self.y[idx]
            else:
                return self.x[idx], self.attributes, self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load input and output data from text files."""
        df = load_wfdei_gem_capa_forcings_lumped(self.data_root, [self.basin])[self.basin]
        qobs = load_grip_gl_discharge(self.data_root, basins=[self.basin]).set_index('date')['qobs']
        if len(qobs) != len(df):
            print(f"Length of forcings and observations doesn't match for basin {self.basin}")
        df['qobs'] = qobs

        # we use (seq_len) time steps before start for warmup
        start_date = self.dates[0] - pd.DateOffset(days=self.seq_length - 1)
        end_date = self.dates[1]
        df = df[start_date:end_date]

        # store first and last date of the selected period
        self.period_start = df.index[0]
        self.period_end = df.index[-1]

        # use all meteorological variables as inputs
        x = np.array([
            df['precip'].values, df['temp_daily_avg'].values, df['temp_min'].values,
            df['temp_max'].values
        ]).T

        y = np.array([df['qobs'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = normalize_features(x, variable='input')

        x, y = reshape_data(x, y, self.seq_length)

        if self.is_train:
            # Deletes all records with invalid discharge
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
            
            # Delete all samples where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs in basin {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # store std of discharge before normalization
            self.q_std = np.std(y)

            y = normalize_features(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _load_attributes(self) -> torch.Tensor:
        raise NotImplementedError()