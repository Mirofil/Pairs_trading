import os
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import distance
from pairs.helpers import data_path
from pairs.cointmethod import cointegration, find_integrated
from pairs.config import (
    TradingUniverse,
)
from pairs.simulation import simulate, simulate_mlflow
from pairs.simulations_database import generate_scenario
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

univ = TradingUniverse(data_path='/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/hist/amex/', tracking_uri="http://0.0.0.0:5000",
        start_date=[1996, 1, 1],
        end_date=[2020, 1, 1],show_progress_bar=True)

config=generate_scenario(
        freq="1D",
        lag=1,
        txcost=0.003,
        pairs_deltas={'formation_delta':[6,0,0], 'training_delta':[3,0,0]},
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
    )
#%%
simulate(config)

if __name__ == "__main__":
    ray.init(
        num_cpus=39,
        # include_webui=True,
        log_to_driver=True,
        
    )

    univ = TradingUniverse(
        data_path="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/hist/amex/",
        tracking_uri="http://0.0.0.0:5000",
        start_date=[1990, 1, 1],
        end_date=[1991, 1, 1],
        volume_cutoff=[0.1, 1],
        name="nyse_dist_big8",
    )

    analysis = tune.run(
        simulate_mlflow,
        local_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/ray_results/",
        name="simulate_retries3",
        max_failures=3,
        config=generate_scenario(
            freq="1D",
            lag=tune.grid_search([0]),
            txcost=0.003,
            pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]}]),
            jump=[1, 0, 0],
            method="dist",
            dist_num=tune.grid_search([5]),
            threshold=tune.grid_search([3,2]),
            stoploss=0,
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ),
        ),
    )