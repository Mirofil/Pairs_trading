import datetime
import os
import timeit

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
import ray
import scipy
import seaborn as sns
import statsmodels
from dateutil.relativedelta import relativedelta
from ray import tune

from pairs.analysis import descriptive_stats
from pairs.cointmethod import coint_spread, cointegration, find_integrated
from pairs.config import TradingUniverse
from pairs.datasets.crypto_dataset import CryptoDataset
from pairs.distancemethod import distance, distance_spread
from pairs.helpers import data_path, prefilter, preprocess
from pairs.pairs_trading_engine import (
    calculate_profit,
    pick_range,
    propagate_weights,
    signals,
    sliced_norm,
    weights_from_signals,
)
from pairs.simulation import simulate
from pairs.simulations_database import *


def test_distance():
    data_path = "paper1/NEWconcatenated_price_data/"
    files = os.listdir(data_path)
    # we exclude CLOAKBTC because theres some data-level mixed types mistake that breaks prefilter and it would get deleted anyways
    # it also breakts at ETHBTC (I manually deleted the first wrong part in Excel)
    paths = [
        data_path + x
        for x in files
        if x not in ["BTCUSDT.csv", "ETHUSDT.csv", "CLOAKBTC.csv"]
    ]
    prefiltered = prefilter(paths, cutoff=0.7)
    preprocessed = preprocess(prefiltered[:, 0], first_n=0, freq="1D")

    pd.options.mode.chained_assignment = None
    formation = (datetime.date(*[2018, 1, 1]), datetime.date(*[2018, 1, 7]))
    trading = (formation[1], formation[1] + relativedelta(months=2))

    # we take timeframe corresponding to Formation period when finding the lowest SSDs
    start = datetime.datetime.now()
    head = pick_range(preprocessed, formation[0], formation[1])
    distances = distance(head, num=20)
    end = datetime.datetime.now()
    print("Distances were found in: " + str(end - start))
    #%%
    start = datetime.datetime.now()
    short_preprocessed = pick_range(preprocessed, formation[0], trading[1])
    spreads = distance_spread(short_preprocessed, distances["viable_pairs"], formation)
    end = datetime.datetime.now()
    print("Distance spreads were found in: " + str(end - start))
    # this is some technical detail needed later?
    spreads.sort_index(inplace=True)
    #%%
    start = datetime.datetime.now()
    dist_signal = signals(
        spreads, timeframe=trading, formation=formation, lag=1, num_of_processes=3
    )
    weights_from_signals(dist_signal, cost=0.003)
    end = datetime.datetime.now()
    print("Distance signals were found in: " + str(end - start))
    #%%
    start = datetime.datetime.now()
    propagate_weights(dist_signal, formation)
    end = datetime.datetime.now()
    print("Weight propagation was done in: " + str(end - start))
    #%%
    start = datetime.datetime.now()
    calculate_profit(dist_signal, cost=0.003)
    end = datetime.datetime.now()
    print("Profit calculation was done in: " + str(end - start))

    parent_dir = os.path.dirname(__file__)

    dist_signal_reference = pd.read_parquet(
        os.path.join(parent_dir, "dist_signal_reference.parquet")
    )

    assert (
        descriptive_stats(dist_signal)
        .astype(np.float32)
        .round(2)
        .equals((descriptive_stats(dist_signal_reference).astype(np.float32).round(2)))
    )


def test_configs_in_distance():
    ray.init(num_cpus=3, include_webui=True)

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
            start=start_date,
            end=end_date,
            jump=[1, 0, 0],
            method="dist",
            dist_num=20,
            threshold=2,
            stoploss=100,
            redo_prefiltered=False,
            redo_preprocessed=False,
            truncate=True,
            name="scenarioX",
            data_path=paper1_data_path,
            save=paper1_results,
            show_progress_bar=False,
        ),
    )
    assert analysis.dataframe()["Traded pairs"].astype(np.float32).round(2) == 0.82
    assert (
        analysis.dataframe()["Monthly profit"].astype(np.float32).round(5) == -0.00066
    )
