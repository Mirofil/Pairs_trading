import os
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import distance, distance_spread
from pairs.helpers import data_path, prefilter, preprocess
from pairs.cointmethod import coint_spread, cointegration, find_integrated
from pairs.config import NUMOFPROCESSES, data_path, end_date, save, start_date, version, TradingUniverse
from pairs.simulation import simulate
from pairs.simulations_database import *
from pairs.pairs_trading_engine import (
    calculate_profit,
    pick_range,
    propagate_weights,
    signals,
    sliced_norm,
    weights_from_signals,
)
from ray import tune
import ray
import mlflow
from pairs.datasets.crypto_dataset import CryptoDataset
from pairs.datasets.us_dataset import USDataset

trading_univ = TradingUniverse(
            name="scenarioX",
            data_path=paper1_data_path,
            save_path_results=paper1_results,
            show_progress_bar=False,
            saving_method="pkl",
        )
config=generate_scenario(
            freq="1D",
            lag=1,
            txcost=0.003,
            training_delta=[2, 0, 0],
            formation_delta=[4, 0, 0],
            jump=[1, 0, 0],
            method="dist",
            dist_num=20,
            threshold=2,
            stoploss=100,
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=trading_univ,
            dataset=CryptoDataset(config=trading_univ)
        )
simulate(config)

if __name__ == "__main__":
    ray.init(num_cpus=3, include_webui=True, local_mode=True)
    mlflow.set_experiment("pairs_simulation")

    analysis = tune.run(
        simulate,
        name="results",
        config=generate_scenario(
            threshold=tune.grid_search(list(np.arange(0.5, 6, 0.5))),
            lag=tune.grid_search(list(np.arange(0, 3, 1))),
            dist_num=tune.grid_search(list(np.arange(10, 100, 20))),
            txcost=tune.grid_search([0.003]),
        ),
    )

    # TESTING BACKWARDS COMPAT
    trading_univ = TradingUniverse(
                name="scenarioX",
                data_path=paper1_data_path,
                save_path_results=paper1_results,
                show_progress_bar=False,
                saving_method="pkl",
            )
    univ = TradingUniverse(data_path='/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/hist/amex/')

    analysis = tune.run(
        simulate,
        name="results_backwards_compat",
        config=generate_scenario(
            freq="1D",
            lag=1,
            txcost=0.003,
            training_delta=[2, 0, 0],
            formation_delta=[4, 0, 0],
            jump=[1, 0, 0],
            method="dist",
            dist_num=20,
            threshold=2,
            stoploss=100,
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ)
        ),
    )

#%%
analysis = tune.run(
    simulate,
    name="results_backwards_compat",
    config=generate_scenario(
        freq="1D",
        lag=1,
        txcost=0.003,
        training_delta=[2, 0, 0],
        formation_delta=[4, 0, 0],
        jump=[1, 0, 0],
        method="dist",
        dist_num=20,
        threshold=2,
        stoploss=100,
        redo_prefiltered=True,
        redo_preprocessed=True,
        truncate=True,
        trading_univ=univ,
        dataset=USDataset(config=univ)
    ),
)