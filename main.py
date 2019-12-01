import pickle
import json
import itertools
from ast import literal_eval
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mlstream.experiment import Experiment
from mlstream.utils import store_results, nse
from mlstream.datautils import get_basin_list
from mlstream.models.base_models import LumpedModel
from mlstream.models.sklearn_models import LumpedSklearnRegression
from mlstream.models.lstm import LumpedLSTM
from mlstream.models.xgboost import LumpedXGBoost


XGB_PARAM_DIST = {
    'learning_rate': [0.25],
    'gamma': stats.uniform(0, 5),
    'max_depth': stats.randint(2, 8),
    'min_child_weight': stats.randint(1, 15),
    'subsample': [0.8],
    'colsample_bytree': stats.uniform(0.4, 0.6),
    'colsample_bylevel': stats.uniform(0.4, 0.6)
}
XGB_REG_PARAM_DIST = {
    'reg_alpha': stats.expon(0,20),
    'reg_lambda': stats.expon(0,20)
}


def get_args() -> Dict:
    """Parses input arguments.

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "predict", "combine_ensemble"])
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

    # set training basins
    if cfg["basins"] is None:
        #print("Using random subset of calibration basins")
        cal_basins = get_basin_list(cfg["data_root"], "C")
        #cfg["basins"] = np.random.choice(cal_basins,
        #                                 size=int(len(cal_basins) * 0.8),
        #                                 replace=False)
        cal_basins = np.random.choice(cal_basins, len(cal_basins), replace=False)
        basin_chunks = int(len(cal_basins)/4)

    run_metadata = {"model_type": cfg["model_type"],
                    "use_mse": cfg["use_mse"]}
    run_dir = cfg["run_dir"]
    run_metadata.update(cfg["model_args"])
    seq_lengths = [10, 30, 100, 200]
    dropouts = [0.0, 0.2, 0.4, 0.5]
    hidden_sizes = [64, 128, 256]
    learning_rates = [{1: 1e-3},
                      {1: 5e-3, 11: 1e-3, 21: 5e-4},
                      {1: 5e-3, 21: 1e-3, 61: 5e-4}]
    epochs = [30, 50, 100]
    s, d, h, l, e =list(itertools.product(seq_lengths, dropouts,
                                          hidden_sizes, learning_rates,
                                          epochs))[run_metadata["config"]]
    print(run_metadata["config"], s, d, h, l, e)
    i = run_metadata["config"]
    del run_metadata["config"]
    mins,meds,maxs = [],[],[]
    for j in range(4):
        basins = [b for b in cal_basins if b not in cal_basins[j*basin_chunks : (j+1)*basin_chunks]]
        print(j, basins)
        print("Setting up experiment.")
        cfg["run_dir"] = run_dir / str(i) / str(j)
        run_metadata["learning_rates"] = l
        run_metadata["hidden_size"] = h
        run_metadata["epochs"] = e
        run_metadata["dropout"] = d
        exp = Experiment(cfg["data_root"], is_train=True, run_dir=cfg["run_dir"],
                         start_date=cfg["start_date"], end_date=cfg["end_date"],
                         basins=basins, forcing_attributes=cfg["forcing_attributes"],
                         static_attributes=cfg["static_attributes"], seq_length=s,
                         concat_static=cfg["concat_static"], no_static=cfg["no_static"],
                         cache_data=cfg["cache_data"], n_jobs=cfg["num_workers"], seed=cfg["seed"],
                         run_metadata=run_metadata)

        exp.set_model(_get_model(cfg, is_train=True))
        print("Starting training.")
        exp.train()
        exp.cfg["basins"] = cal_basins
        exp.cfg["start_date"] = pd.to_datetime("2008-01-01")
        exp.cfg["end_date"] = pd.to_datetime("2010-12-31")
        results = exp.predict()
        nses = exp.get_nses()
        nse_list = list(nses.values())
        print("Overall NSEs:", np.median(nse_list),
              np.min(nse_list), np.max(nse_list))
        mins.append(np.min(nse_list))
        maxs.append(np.max(nse_list))
        meds.append(np.median(nse_list))

        store_results(cfg, cfg, results)
    print(mins, medians, maxs)
    print(np.mean(medians))


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
    model = _get_model(run_cfg, is_train=False)
    epoch = run_cfg["epochs"] if "epochs" in run_cfg else 30
    model_filename = f'model_epoch{epoch}.pt' if run_cfg["model_type"] == 'lstm' else 'model.pkl'
    model.load(Path(user_cfg["run_dir"]) / model_filename)
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


def combine_ensemble(cfg: Dict):
    """Combines predictions of multiple runs into one.

    Parameters
    ----------
    cfg : Dict
        Dict with entry "run_dir".
        Will combine all runs inside this directory.
    """
    overall = None
    for f in cfg["run_dir"].glob("*/results*.p"):
        print(f)
        results = pickle.load(f, 'rb')
        if overall is None:
            overall = results
        else:
            if len(overall) != len(results):
                print("Length of predictions not equal.")
            overall['qsim'] + results['qsim']
    overall['qsim'] /= len(run_dirs)
    print(f"Ensemble NSE: {nse(overall['qsim'].values, overall['qobs'].values)}.")


def _get_model(cfg: Dict, is_train: bool) -> LumpedModel:
    """Creates model to train or evaluate.

    Parameters
    ----------
    cfg : Dict
        Run configuration
    is_train : bool
        Whether the model should be loaded for training or prediction.

    Returns
    -------
    model : LumpedModel
        Model to train or evaluate

    Raises
    ------
    ValueError
        If ``cfg["model_type"]`` is invalid 
        or ``cfg["model_args"]`` is insufficient for the model type.
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
        if "config" in model_args:
            del model_args["config"]
        model = LumpedLSTM(len(cfg["forcing_attributes"]),
                           len(cfg["static_attributes"]) - 2,  # lat/lon is not part of training
                           use_mse=cfg["use_mse"],
                           no_static=cfg["no_static"],
                           concat_static=cfg["concat_static"],
                           run_dir=cfg["run_dir"],
                           n_jobs=n_jobs,
                           **model_args)

    elif cfg["model_type"] == 'xgb':
        if is_train:
            model_args["param_dist"] = XGB_PARAM_DIST
            model_args["reg_search_param_dist"] = XGB_REG_PARAM_DIST
            if any([c not in model_args for c in ["n_estimators", "learning_rate",
                                                  "early_stopping_rounds", "n_cv",
                                                  "param_search_n_estimators",
                                                  "param_search_early_stopping_rounds",
                                                  "param_search_n_iter",
                                                  "reg_search_n_iter"]]):
                raise ValueError("XGBoost configuration incomplete.")
        model = LumpedXGBoost(no_static=cfg["no_static"],
                              concat_static=cfg["concat_static"],
                              use_mse=cfg["use_mse"],
                              run_dir=cfg["run_dir"],
                              n_jobs=n_jobs,
                              seed=cfg["seed"],
                              **model_args)

    else:
        raise ValueError(f'Unknown model type {cfg["model_type"]}')

    return model


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
