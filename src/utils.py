from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import h5py

from .datasets import LumpedBasin


def create_h5_files(data_root: Path,
                    out_file: Path,
                    basins: List,
                    dates: List,
                    seq_length: int):
    """Create H5 training and 
    
    Parameters
    ----------
    data_root: Path
        Path to the main directory of the data set
    out_file : Path
        Path of the location where the hdf5 file should be stored
    basins : List
        List containing the gauge ids
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    seq_length : int, optional
        Length of the requested input sequences
    
    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    num_forcing_vars = 4
    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset('input_data',
                                          shape=(0, seq_length, num_forcing_vars),
                                          maxshape=(None, seq_length, num_forcing_vars),
                                          chunks=True,
                                          dtype=np.float32,
                                          compression='gzip')
        target_data = out_f.create_dataset('target_data',
                                           shape=(0, 1),
                                           maxshape=(None, 1),
                                           chunks=True,
                                           dtype=np.float32,
                                           compression='gzip')

        q_stds = out_f.create_dataset('q_stds',
                                      shape=(0, 1),
                                      maxshape=(None, 1),
                                      dtype=np.float32,
                                      compression='gzip',
                                      chunks=True)

        sample_2_basin = out_f.create_dataset('sample_2_basin',
                                              shape=(0, ),
                                              maxshape=(None, ),
                                              dtype="S10",
                                              compression='gzip',
                                              chunks=True)

        for basin in tqdm(basins, file=sys.stdout):

            dataset = LumpedBasin(data_root=data_root,
                                  basin=basin,
                                  is_train=True,
                                  seq_length=seq_length,
                                  dates=dates)

            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples

            # store input and output samples
            input_data.resize((total_samples, seq_length, num_forcing_vars))
            target_data.resize((total_samples, 1))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :] = dataset.y

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            sample_2_basin.resize((total_samples, ))
            str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
            sample_2_basin[-num_samples:] = str_arr

            out_f.flush()
