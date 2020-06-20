import os
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import distance, distance_spread
from pairs.helpers import data_path
from pairs.cointmethod import coint_spread, cointegration, find_integrated
from pairs.config import (
    NUMOFPROCESSES,
    data_path,
    end_date,
    save,
    start_date,
    version,
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


# univ = TradingUniverse(data_path='/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/hist/nyse/', tracking_uri="file:/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/mlruns",
#         start_date=[1990, 1, 1],
#         end_date=[2020, 1, 1],)

# config=generate_scenario(
#         freq="1D",
#         lag=1,
#         txcost=[0.003],
#         training_delta=[3, 0, 0],
#         formation_delta=[6, 0, 0],
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
    ray.init(
        num_cpus=36,
        # include_webui=True,
        log_to_driver=True,
        
    )

    univ = TradingUniverse(
        data_path="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/hist/nyse/",
        tracking_uri="file:/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/mlruns",
        start_date=[1990, 1, 1],
        end_date=[2020, 1, 1],
        volume_cutoff=[0.1, 1],
        name="nyse_dist_big4",
    )

    # run_ids = []

    # for _ in step2_arg_product:
    #     time.sleep(1)
    #     run = mlflow.start_run(experiment_id=experiment_id)
    #     run_ids.append(run.info.run_id)
    #     time.sleep(0.2)
    #     mlflow.end_run()

    analysis = tune.run(
        simulate,
        local_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/ray_results/",
        name="big_run",
        max_failures=3,
        config=generate_scenario(
            freq="1D",
            lag=tune.grid_search([0,1]),
            txcost=0.003,
            pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]},
                {"training_delta":[6,0,0], "formation_delta":[12,0,0]},
                {"training_delta":[1,0,0], "formation_delta":[2,0,0]}]),
            jump=[1, 0, 0],
            method="dist",
            dist_num=tune.grid_search([5, 10, 20, 40]),
            threshold=tune.grid_search([[0.5, 1], [1.5,2], [2.5,3]]),
            stoploss=0,
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ),
        ),
    )

    #TODO
    analysis = tune.run(
        simulate,
        local_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/ray_results/",
        name="big_run",
        max_failures=3,
        config=generate_scenario(
            freq="1D",
            lag=[0,1],
            txcost=[0, 0.003, 0.005],
            pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]},
                {"training_delta":[6,0,0], "formation_delta":[12,0,0]},
                {"training_delta":[1,0,0], "formation_delta":[2,0,0]}]),
            jump=[1, 0, 0],
            method="dist",
            dist_num=tune.grid_search([5, 10, 20, 40, 70, 100]),
            threshold=[1.5,2],
            stoploss=tune.grid_search([0, 1, 2]),
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ),
        ),
    )

    analysis = tune.run(
        simulate,
        local_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/ray_results/",
        name="big_run",
        max_failures=3,
        config=generate_scenario(
            freq="1D",
            lag=[0,1],
            txcost=[0, 0.003, 0.005],
            pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]},
                {"training_delta":[6,0,0], "formation_delta":[12,0,0]},
                {"training_delta":[1,0,0], "formation_delta":[2,0,0]}]),
            jump=[1, 0, 0],
            method="dist",
            dist_num=tune.grid_search([5, 10, 20, 40, 70, 100]),
            threshold=[2.5,3],
            stoploss=tune.grid_search([0, 1, 2]),
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ),
        ),
    )
    analysis = tune.run(
        simulate,
        local_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading/ray_results/",
        name="big_run",
        max_failures=3,
        config=generate_scenario(
            freq="1D",
            lag=[0,1],
            txcost=[0, 0.003, 0.005],
            pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]},
                {"training_delta":[6,0,0], "formation_delta":[12,0,0]},
                {"training_delta":[1,0,0], "formation_delta":[2,0,0]}]),
            jump=[1, 0, 0],
            method="dist",
            dist_num=tune.grid_search([5, 10, 20, 40]),
            threshold=[3.5,4],
            stoploss=tune.grid_search([0, 1, 2]),
            redo_prefiltered=True,
            redo_preprocessed=True,
            truncate=True,
            trading_univ=univ,
            dataset=USDataset(config=univ),
        ),
    )




