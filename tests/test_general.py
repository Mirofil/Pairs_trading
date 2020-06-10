import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import timeit
import datetime
import scipy
import statsmodels
import seaborn as sns
import multiprocess as mp
from dateutil.relativedelta import relativedelta
from distancemethod import distance, distance_spread
from helpers import data_path, prefilter, preprocess
from cointmethod import coint_spread, cointegration, find_integrated
from simulation import simulate
from simulations_database import *
from pairs_trading_engine import (calculate_profit, pick_range,
                                  propagate_weights, signals, sliced_norm,
                                  weights_from_signals)

from analysis import descriptive_stats


def test_distance():
    data_path = 'paper1/NEWconcatenated_price_data/'
    files = os.listdir(data_path)
    #we exclude CLOAKBTC because theres some data-level mixed types mistake that breaks prefilter and it would get deleted anyways
    #it also breakts at ETHBTC (I manually deleted the first wrong part in Excel)
    paths = [data_path + x for x in files if x not in ['BTCUSDT.csv', 'ETHUSDT.csv', 'CLOAKBTC.csv']]
    prefiltered=prefilter(paths, cutoff=0.7)
    preprocessed=preprocess(prefiltered[:,0], first_n=0, freq='1D')

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
    spreads = distance_spread(short_preprocessed, distances[2], formation)
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

    dist_signal_reference = pd.read_parquet("dist_signal_reference.parquet")

    assert dist_signal.equals(dist_signal_reference)


