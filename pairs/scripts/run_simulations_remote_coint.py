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
from pairs.simulation import simulate
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


if __name__ == "__main__":
    ray.init(
        num_cpus=5,
        # include_webui=True,
        log_to_driver=False,
        
    )

    univ = TradingUniverse(
        data_path="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/hist/nyse/",
        # tracking_uri="file:/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/mlruns",
        tracking_uri="http://0.0.0.0:5000",
        start_date=[1990, 1, 1],
        end_date=[2020, 1, 1],
        volume_cutoff=[0.1, 1],
        name="nyse_coint_big_nomlflow7",
    )

    analysis = tune.run(
        simulate,
        local_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/",
        name="simulate_coint_retries_nomlflow7",
        max_failures=3,
        config=generate_scenario(
            freq="1D",
            lag=tune.grid_search([0]),
            txcost=0.003,
            pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]},
                {"training_delta":[6,0,0], "formation_delta":[12,0,0]},
                {"training_delta":[1,0,0], "formation_delta":[2,0,0]}]),
            jump=[2, 0, 0],
            method="coint",
            dist_num=tune.grid_search([5, 10, 20, 40]),
            threshold=tune.grid_search([0.5,1,1.5,2,2.5,3]),
            confidence=tune.grid_search([0.01,0.05,0.1]),
            stoploss=100,
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ),
        ),
    )





