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


if __name__ == "__main__":
    ray.init(num_cpus=3, include_webui=True)
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
    analysis = tune.run(
        simulate,
        name="results_backwards_compat",
        config=generate_scenario(
            freq="1D",
            lag=1,
            txcost=0.003,
            training_delta=[2, 0, 0],
            volume_cutoff=0.7,
            formation_delta=[4, 0, 0],
            jump=[1, 0, 0],
            method="dist",
            dist_num=20,
            threshold=2,
            stoploss=100,
            redo_prefiltered=False,
            redo_preprocessed=False,
            truncate=True,
            trading_univ=TradingUniverse(
                name="scenarioX",
                data_path=paper1_data_path,
                save_path_results=paper1_results,
                show_progress_bar=False,
                saving_method="pkl",
            ),
        ),
    )
