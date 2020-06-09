#%%
import pandas as pd
from helpers import data_path, save
from cointmethod import find_integrated, cointegration, coint_spread
from distancemethod import (
    distance,
    distance_spread,
)
import os
from config import data_path, save, version, NUMOFPROCESSES, enddate, startdate
import datetime
from typing import *
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import numpy as np

from pairs_trading_engine import sliced_norm, pick_range, signals, weights_from_signals, propagate_weights, calculate_profit

num_of_processes = NUMOFPROCESSES
pd.options.mode.chained_assignment = None
#%%
def simulate(
    params: Dict,
    data_path: str = data_path,
    save: str = save,
    num_of_processes: int = num_of_processes,
    redo_prefiltered = False,
    redo_preprocessed = False,
    truncate = True,
    volume_cutoff=0.7
):
    freq = params["freq"]
    lag = params["lag"]
    txcost = params["txcost"]
    training_delta = params["training_delta"]
    cutoff = params["cutoff"]
    formation_delta = params["formation_delta"]
    start = params["start"]
    end = params["end"]
    jump = params["jump"]
    methods = params["methods"]
    dist_num = params["dist_num"]
    threshold = params["threshold"]
    stoploss = params["stoploss"]
    scenario = params["name"]

    files = os.listdir(data_path)
    paths = [
        os.path.join(data_path, x)
        for x in files
        if x not in ["BTCUSDT.csv", "ETHUSDT.csv", "CLOAKBTC.csv"]
    ]
    names = [file.partition(".")[0] for file in files]

    formationdelta = relativedelta(
        months=formation_delta[0], days=formation_delta[1], hours=formation_delta[2]
    )
    trainingdelta = relativedelta(
        months=training_delta[0], days=training_delta[1], hours=training_delta[2]
    )
    jumpdelta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    # 5000 is arbirtrarily high limit that will never be reached - but the
    print("Starting " + scenario)
    print("\n")
    if not os.path.isdir(os.path.join(save, scenario)):
        os.mkdir(os.path.join(save, scenario))
    with open(os.path.join(save, scenario, "parameters" + ".txt"), "w") as tf:
        print(params, file=tf)

    for i in tqdm(range(50000), desc = 'Starting nth iteration of the formation-trading loop'):
        formation = (start + i * jumpdelta, start + formationdelta + i * jumpdelta)
        trading = (formation[1], formation[1] + trainingdelta)
        print("Starting: " + str(formation) + " at " + str(datetime.datetime.now()))
        if trading[1] > end:
            if truncate == True:
                trading = (trading[0], end)
            else:
                break
        if trading[1] < formation[1]:
            break

        if redo_prefiltered == True:
            prefiltered = prefilter(paths, cutoff=cutoff)
            np.save(os.path.join(save, str(i) + "x" + str(cutoff), prefiltered))
        else:
            prefiltered_fpath = os.path.join(save, version + "prefiltered" + str(cutoff).replace(".", "_") + ".npy")
            if not os.path.isfile(prefiltered_fpath):
                prefiltered = prefilter(paths, cutoff=cutoff)
                np.save(prefiltered_fpath, prefiltered)
            else:
                prefiltered = np.load(
                    prefiltered_fpath
                )
        if redo_preprocessed == True:
            preprocessed = preprocess(prefiltered[:, 0], first_n=0, freq=freq)
            preprocessed.to_pickle(os.path.join(save, str(i) + "y" + str(freq)))

        else:
            preprocessed_fpath = os.path.join(
                    save, version + "preprocessed" + str(freq) + str(cutoff).replace(".", "_") + ".pkl"
                )
            if not os.path.isfile(preprocessed_fpath):
                preprocessed = preprocess(prefiltered[:, 0], first_n=0, freq=freq)
                preprocessed.to_pickle(preprocessed_fpath)
            else:
                preprocessed = pd.read_pickle(
                    preprocessed_fpath
                )
        if "coint" in methods:
            coint_head = pick_range(preprocessed, formation[0], formation[1])
            k = cointegration(
                find_integrated(coint_head), num_of_processes=num_of_processes
            )
            short_y = pick_range(preprocessed, formation[0], trading[1])
            coint_spreads = coint_spread(
                short_y,
                [item[0] for item in k],
                timeframe=formation,
                betas=[item[1] for item in k],
            )
            coint_spreads.sort_index(inplace=True)
            coint_signal = signals(
                coint_spreads,
                timeframe=trading,
                formation=formation,
                lag=lag,
                threshold=threshold,
                stoploss=stoploss,
                num_of_processes=num_of_processes,
            )
            #I think this is useless so let me comment it out
            # coint_signal = signals_numeric(coint_signal)
            weights_from_signals(coint_signal, cost=txcost)
            propagate_weights(coint_signal, formation)
            calculate_profit(coint_signal, cost=txcost)
            coint_signal.to_pickle(
                os.path.join(save, scenario, str(i) + "coint_signal.pkl")
            )
        if "dist" in methods:
            head = pick_range(preprocessed, formation[0], formation[1])
            distances = distance(head, num=dist_num)
            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = distance_spread(short_y, distances[2], formation)
            spreads.sort_index(inplace=True)
            dist_signal = signals(
                spreads,
                timeframe=trading,
                formation=formation,
                lag=lag,
                threshold=threshold,
                stoploss=stoploss,
                num_of_processes=num_of_processes,
            )
            weights_from_signals(dist_signal, cost=txcost)
            propagate_weights(dist_signal, formation)
            calculate_profit(dist_signal, cost=txcost)
            dist_signal.to_pickle(
                os.path.join(save, scenario, str(i) + "dist_signal.pkl")
            )
        if trading[1] == enddate:
            break


def stoploss(freqs=["daily"], thresh=[1, 2, 3], stoploss=[2, 3, 4, 5, 6], save=save):

    scenariod = {
        "freq": "1D",
        "lag": 1,
        "txcost": 0.003,
        "training_delta": [2, 0, 0],
        "cutoff": 0.7,
        "formation_delta": [4, 0, 0],
        "start": datetime.date(*[2018, 1, 1]),
        "end": datetime.date(*[2019, 9, 1]),
        "jump": [1, 0, 0],
        "methods": ["dist", "coint"],
        "dist_num": 20,
        "threshold": 2,
        "stoploss": 100,
        "name": "scenariosd12",
    }
    scenarioh = {
        "freq": "1H",
        "lag": 1,
        "txcost": 0.003,
        "training_delta": [0, 10, 0],
        "cutoff": 0.7,
        "formation_delta": [0, 20, 0],
        "start": datetime.date(*[2018, 1, 1]),
        "end": datetime.date(*[2019, 9, 1]),
        "jump": [0, 10, 0],
        "methods": ["dist", "coint"],
        "dist_num": 20,
        "threshold": 2,
        "stoploss": 100,
        "name": "scenariosh12",
    }
    if "daily" in freqs:
        for i in range(len(thresh)):
            for j in range(len(stoploss)):
                newnamed = scenariod["name"][:-2] + str(thresh[i]) + str(stoploss[j])
                if os.path.isfile(
                    os.path.join(save, newnamed, str(0) + "dist_signal.pkl")
                ):
                    continue
                scenariod.update(
                    {"threshold": thresh[i], "stoploss": stoploss[j], "name": newnamed}
                )
                simulate(scenariod)
    if "hourly" in freqs:
        for i in range(len(thresh)):
            for j in range(len(stoploss)):
                newnameh = scenarioh["name"][:-2] + str(thresh[i]) + str(stoploss[j])
                if os.path.isfile(
                    os.path.join(save, newnameh, str(0) + "coint_signal.pkl")
                ):
                    continue
                scenarioh.update(
                    {"threshold": thresh[i], "stoploss": stoploss[j], "name": newnameh}
                )
                simulate(scenarioh)
