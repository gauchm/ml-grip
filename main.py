import json
from ast import literal_eval
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mlstream.experiment import Experiment
from mlstream.utils import store_results
from mlstream.datautils import get_basin_list
from mlstream.models.base_models import LumpedModel
from mlstream.models.sklearn_models import LumpedSklearnRegression
from mlstream.models.lstm import LumpedLSTM


def get_args() -> Dict:
    """Parses input arguments.

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "predict"])
    parser.add_argument('--model_type', type=str, help="Model to train.")
    parser.add_argument('--use_mse', action='store_true',
                        help="If provided, uses MSE as objective/loss function.")
    parser.add_argument('--no_static', action='store_true',
                        help="If True, trains without static features")
    parser.add_argument('--concat_static', action='store_true',
                        help="If True, train with static features concatenated at each time step")
    parser.add_argument('--model_args', nargs='+', required=False, type=str,
                        help="Additional arguments to pass to the model, \
                        provided as list, e.g.: --model_args dropout 0.1.")

    parser.add_argument('--seq_length', type=int, default=10,
                        help="Number of historical time steps to feed the model.")

    parser.add_argument('--data_root', type=str, help="Root directory of data set")
    parser.add_argument('--run_dir', type=str,
                        help="Path to run directory (folder will be created in training mode).")

    parser.add_argument('--start_date', type=str, help="Start date (training start date in \
        training mode, validation start date in prediction mode) (ddmmyyyy).")
    parser.add_argument('--end_date', type=str, help="End date (training end date in \
        training mode, validation end date in prediction mode) (ddmmyyyy).")

    parser.add_argument('--basins', nargs='+', required=False,
                        help='List of basins to train or predict. Default in training: subset of \
                        calibration basins; in prediction: all basins.')
    parser.add_argument('--forcing_attributes', nargs='+',
                        default=["PRECIP", "TEMP_DAILY_AVE", "TEMP_MIN", "TEMP_MAX"],
                        help='List of forcing attributes to use. Default is all attributes.')
    parser.add_argument('--static_attributes', nargs='+', required=False,
                        default=["Area2", "Lat_outlet", "Lon_outlet", "RivSlope", "Rivlen",
                                 "BasinSlope", "BkfWidth", "BkfDepth", "MeanElev", "FloodP_n",
                                 "Q_Mean", "Ch_n", "Perim_m"],
                        help='List of basin attributes to use.')

    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--cache_data', action='store_true',
                        help="If True, loads all data into memory")

    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] == "predict") and (cfg["run_dir"] is None):
        raise ValueError("In prediction mode a run directory (--run_dir) has to be specified")

    if cfg["seq_length"] is not None and cfg["seq_length"] < 0:
        raise ValueError("Sequence length can not be negative.")

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert str paths to Path objects
    cfg["data_root"] = Path(cfg["data_root"])
    cfg["run_dir"] = Path(cfg["run_dir"])

    if cfg["model_args"] is None:
        cfg["model_args"] = {}
    else:
        if len(cfg["model_args"]) % 2 != 0:
            raise ValueError("model_args needs to be list of <key> <value> pairs.")
        cfg["model_args"] = {k: literal_eval(v) for k, v in zip(cfg["model_args"][::2],
                                                                cfg["model_args"][1::2])}

    return cfg


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg: Dict
        Dictionary containing the run configuration
    """
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if cfg["basins"] is None:
        print("Using random subset of calibration basins")
        cal_basins = get_basin_list(cfg["data_root"], "C")
        cfg["basins"] = np.random.choice(cal_basins,
                                         size=int(len(cal_basins) * 0.8),
                                         replace=False)

    run_metadata = {"model_type": cfg["model_type"],
                    "use_mse": cfg["use_mse"]}
    run_metadata.update(cfg["model_args"])
    print("Setting up experiment.")
    exp = Experiment(cfg["data_root"], is_train=True, run_dir=cfg["run_dir"],
                     start_date=cfg["start_date"], end_date=cfg["end_date"],
                     basins=cfg["basins"], forcing_attributes=cfg["forcing_attributes"],
                     static_attributes=cfg["static_attributes"], seq_length=cfg["seq_length"],
                     concat_static=cfg["concat_static"], no_static=cfg["no_static"],
                     cache_data=cfg["cache_data"], n_jobs=cfg["num_workers"], seed=cfg["seed"],
                     run_metadata=run_metadata)

    exp.set_model(_get_model(cfg))
    print("Starting training.")
    exp.train()


def predict(user_cfg: Dict):
    """Generate predictions with a trained model

    Parameters
    ----------
    user_cfg: Dict
        Dictionary containing the user entered prediction config
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if user_cfg["basins"] is None:
        user_cfg["basins"] = get_basin_list(user_cfg["data_root"], "*")

    exp = Experiment(user_cfg["data_root"], is_train=False,
                     run_dir=Path(user_cfg["run_dir"]), basins=user_cfg["basins"],
                     start_date=user_cfg["start_date"], end_date=user_cfg["end_date"])

    # load model
    model = _get_model(run_cfg)
    model_filename = 'model_epoch30.pt' if run_cfg["model_type"] == 'lstm' else 'model.pkl'
    model.load(Path(run_cfg["run_dir"]) / model_filename)
    exp.set_model(model)

    results = exp.predict()
    store_results(user_cfg, run_cfg, results)

    nses = exp.get_nses()
    nse_list = list(nses.values())
    print("Overall NSEs:", nses, np.median(nse_list),
          np.min(nse_list), np.max(nse_list))
    train_nses = {basin: n for basin, n in nses.items() if basin in run_cfg["basins"]}
    train_nse_list = list(train_nses.values())
    print("Training basins:", train_nses, np.median(train_nse_list),
          np.min(train_nse_list), np.max(train_nse_list))


def _get_model(cfg: Dict) -> LumpedModel:
    """Creates model to train or evaluate.

    Parameters
    ----------
    cfg : Dict
        Run configuration

    Returns
    -------
    model : LumpedModel
        Model to train or evaluate

    Raises
    ------
    ValueError
        If ``cfg["model_type"]`` is invalid.
    """
    n_jobs = cfg["num_workers"] if "num_workers" in cfg else 1
    model_args = cfg["model_args"] if "model_args" in cfg else {}
    model = None
    # sklearn models
    if cfg["model_type"] == 'linearRegression':
        model = LinearRegression(n_jobs=n_jobs)
    elif cfg["model_type"] == 'randomForest':
        model = RandomForestRegressor(n_jobs=n_jobs, **model_args)
    if model is not None:
        model = LumpedSklearnRegression(model, no_static=cfg["no_static"],
                                        concat_static=cfg["concat_static"],
                                        run_dir=cfg["run_dir"],
                                        n_jobs=n_jobs)
    # other models
    elif cfg["model_type"] == 'lstm':
        model = LumpedLSTM(len(cfg["forcing_attributes"]),
                           len(cfg["static_attributes"]) - 2,  # lat/lon is not part of training
                           use_mse=cfg["use_mse"],
                           no_static=cfg["no_static"],
                           concat_static=cfg["concat_static"],
                           run_dir=cfg["run_dir"],
                           n_jobs=n_jobs,
                           **model_args)

    else:
        raise ValueError(f'Unknown model type {cfg["model_type"]}')

    return model


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
