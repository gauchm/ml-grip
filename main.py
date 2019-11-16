import argparse
import json
import pickle
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.linear_models import LinearRegression
import torch

from src.datautils import (load_grip_gl_discharge,
                           load_wfdei_gem_capa_lumped_forcings)


GLOBAL_SETTINGS = {
    
}


def get_args() -> Dict:
    """Parse input arguments
    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('model_type', choices=["xgboost", "linearRegression"])
    parser.add_argument('--seq_length', type=int, help="Number of historical time steps to feed the model.")
    
    parser.add_argument('--data_root', type=str, help="Root directory of data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--use_mse',
                        action='store_true',
                        help="If provided, uses MSE as objective/loss function.")
    parser.add_argument('--run_dir_base', type=str, default="runs", help="For training mode. Path to store run directories in.")
    parser.add_argument('--run_name', type=str, required=False, help="For training mode. Name of the run.")
    parser.add_argument('--train_start', type=str, help="Training start date (ddmmyyyy).")
    parser.add_argument('--train_end', type=str, help="Training end date (ddmmyyyy).")
    parser.add_argument('--basins', 
                        nargs='+',
                        help='List of basins')
    cfg = vars(parser.parse_args())
    
    cfg["train_start"] = pd.to_datetime(cfg["train_start"], format='%d%m%Y')
    cfg["train_end"] = pd.to_datetime(cfg["train_end"], format='%d%m%Y')

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] == "evaluate") and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")
        
    if cfg["seq_length"] < 0:
        raise ValueError("Sequence length can not be negative.")
        
    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert str paths to Path objects
    cfg["data_root"] = Path(cfg["data_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    if cfg["run_dir_base"] is not None:
        cfg["run_dir_base"] = Path(cfg["run_dir_base"])
    
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run
    
    Parameters
    ----------
    cfg: dict
        Dictionary containing the run config
        
    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    second = f"{now.second}".zfill(2)
    if cfg["run_name"] is None:
        run_name = f'run_{cfg["model_type"]}_{day}{month}_{hour}{minute}{second}_seed{cfg["seed"]}'
    else:
        run_name = cfg["run_name"]
    cfg['run_dir'] = Path(__file__).absolute().parent / cfg["run_dir_base"] / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError('There is already a folder at {}'.format(cfg["run_dir"]))

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            elif 'param_dist' in key:
                temp_dict = {}
                for k, v in val.items():
                    if isinstance(v, sp.stats._distn_infrastructure.rv_frozen):
                        temp_dict[k] = f"{v.dist.name}{v.args}, *kwds={v.kwds}"
                    else:
                        temp_dict[k] = str(v)
                temp_cfg[key] = str(temp_dict)
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.
    
    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the gauge ids
        
    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(data_root=cfg["data_root"],
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])

    return cfg


def train(cfg):
    """Train model.
    
    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    basins = cfg["basins"]
    
    
    
if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
