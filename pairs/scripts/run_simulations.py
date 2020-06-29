import os
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pairs.helpers import data_path
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

# univ = TradingUniverse(data_path='/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/hist/amex/')

# config=generate_scenario(
#         freq="1D",
#         lag=1,
#         txcost=[0,0.003],
#         training_delta=[2, 0, 0],
#         formation_delta=[4, 0, 0],
#         jump=[1, 0, 0],
#         method="dist",
#         dist_num=20,
#         threshold=2,
#         stoploss=100,
#         redo_prefiltered=True,
#         redo_preprocessed=True,
#         truncate=True,
#         trading_univ=univ,
#         dataset=USDataset(config=univ)
#     )
# simulate(config)



if __name__ == "__main__":
    ray.init(num_cpus=3, include_webui=True, log_to_driver=True)

    analysis = tune.run(
        simulate,
        name="results",
        config=generate_scenario(
            threshold=tune.grid_search([1,2,3]),
            dist_num=tune.grid_search([5,10,20,40,70,100]),
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
    univ = TradingUniverse(data_path='/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/hist/nyse/')

    analysis = tune.run(
        simulate,
        name="results_backwards_compat",
        config=generate_scenario(
            freq="1D",
            lag=1,
            txcost=[0.003],
            training_delta=[2, 0, 0],
            formation_delta=[4, 0, 0],
            jump=[1, 0, 0],
            method="dist",
            dist_num=tune.grid_search([5]),
            threshold=[1],
            stoploss=100,
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ)
        ),
    )
