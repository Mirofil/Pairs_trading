import os
import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import distance, distance_spread
from pairs.helpers import data_path, prefilter, preprocess
from pairs.cointmethod import coint_spread, cointegration, find_integrated
from pairs.config import NUMOFPROCESSES, data_path, end_date, save, start_date, version
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
    analysis = tune.run(
        simulate,
        name="results",
        config=generate_scenario(
            threshold=tune.grid_search(list(np.arange(0.5, 2, 0.5))),
            lag=tune.grid_search(list(np.arange(1,3,1)))
        ),
    )
