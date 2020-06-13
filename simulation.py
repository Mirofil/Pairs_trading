#%%
import datetime
import os
from typing import *

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from ray import tune
import ray


from cointmethod import coint_spread, cointegration, find_integrated
from config import (
    NUMOFPROCESSES,
    data_path,
    end_date,
    save,
    start_date,
    version,
    standard_result_metrics_from_desc_stats,
)
from distancemethod import distance, distance_spread
from helpers import data_path, prefilter, preprocess
from pairs_trading_engine import (
    calculate_profit,
    pick_range,
    propagate_weights,
    signals,
    sliced_norm,
    weights_from_signals,
)
from analysis import (
    descriptive_stats,
    descriptive_frame,
    summarize,
    aggregate,
    compute_period_length,
    compute_cols_from_freq,
)
from pandas.io.json._normalize import nested_to_record    

num_of_processes = 1


def simulate(
    params
    # data_path: str = data_path,
    # save: str = save,
    # num_of_processes: int = num_of_processes,
    # redo_prefiltered:bool = False,
    # redo_preprocessed:bool = False,
    # truncate:bool = True,
    # volume_cutoff:int=0.7
):
    freq = params["freq"]
    lag = params["lag"]
    txcost = params["txcost"]
    training_delta_raw = params["training_delta"]
    volume_cutoff = params["volume_cutoff"]
    formation_delta = params["formation_delta"]
    start = params["start"]
    end = params["end"]
    jump = params["jump"]
    method = params["method"]
    dist_num = params["dist_num"]
    threshold = params["threshold"]
    stoploss = params["stoploss"]
    scenario = params["name"]
    data_path = params["data_path"]
    save = params["save"]
    num_of_processes = 1
    redo_prefiltered = params["redo_prefiltered"]
    redo_preprocessed = params["redo_preprocessed"]
    truncate = params["truncate"]
    volumne_cutoff = params["volume_cutoff"]
    show_progress_bar = params["show_progress_bar"]

    files = os.listdir(data_path)
    paths = [
        os.path.join(data_path, x)
        for x in files
        if x not in ["BTCUSDT.csv", "ETHUSDT.csv", "CLOAKBTC.csv"]
    ]
    names = [file.partition(".")[0] for file in files]

    formation_delta = relativedelta(
        months=formation_delta[0], days=formation_delta[1], hours=formation_delta[2]
    )
    training_delta = relativedelta(
        months=training_delta_raw[0], days=training_delta_raw[1], hours=training_delta_raw[2]
    )
    jump_delta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    # 5000 is arbirtrarily high limit that will never be reached - but the
    print("Starting " + scenario)
    print("\n")
    if not os.path.isdir(os.path.join(save, scenario)):
        os.mkdir(os.path.join(save, scenario))
    with open(os.path.join(save, scenario, "parameters" + ".txt"), "w") as tf:
        print(params, file=tf)

    backtests = []

    for i in tqdm(
        range(50000), desc="Starting nth iteration of the formation-trading loop", disable = not show_progress_bar
    ):
        formation = (start + i * jump_delta, start + formation_delta + i * jump_delta)
        trading = (formation[1], formation[1] + training_delta)
        if trading[1] > end:
            if truncate == True:
                trading = (trading[0], end)
            else:
                break
        if trading[1] < formation[1]:
            break

        if redo_prefiltered == True:
            prefiltered = prefilter(paths, cutoff=volume_cutoff)
            np.save(os.path.join(save, str(i) + "x" + str(volume_cutoff), prefiltered))
        else:
            prefiltered_fpath = os.path.join(
                save,
                version + "prefiltered" + str(volume_cutoff).replace(".", "_") + ".npy",
            )
            if not os.path.isfile(prefiltered_fpath):
                prefiltered = prefilter(paths, cutoff=volume_cutoff)
                np.save(prefiltered_fpath, prefiltered)
            else:
                prefiltered = np.load(prefiltered_fpath)
        if redo_preprocessed == True:
            preprocessed = preprocess(prefiltered[:, 0], first_n=0, freq=freq)
            preprocessed.to_pickle(os.path.join(save, str(i) + "y" + str(freq)))

        else:
            preprocessed_fpath = os.path.join(
                save,
                version
                + "preprocessed"
                + str(freq)
                + str(volume_cutoff).replace(".", "_")
                + ".pkl",
            )
            if not os.path.isfile(preprocessed_fpath):
                preprocessed = preprocess(prefiltered[:, 0], first_n=0, freq=freq)
                preprocessed.to_pickle(preprocessed_fpath)
            else:
                preprocessed = pd.read_pickle(preprocessed_fpath)
        if "coint" in method:
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
            # I think this is useless so let me comment it out
            # coint_signal = signals_numeric(coint_signal)
            weights_from_signals(coint_signal, cost=txcost)
            propagate_weights(coint_signal, formation)
            calculate_profit(coint_signal, cost=txcost)
            if save is not None:
                coint_signal.to_pickle(
                    os.path.join(save, scenario, str(i) + "coint_signal.pkl")
                )
            backtests.append(coint_signal)
        if "dist" in method:
            head = pick_range(preprocessed, formation[0], formation[1])
            distances = distance(head, num=dist_num)
            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = distance_spread(short_y, distances["viable_pairs"], formation)
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
            if save is not None:
                dist_signal.to_pickle(
                    os.path.join(save, scenario, str(i) + "dist_signal.pkl")
                )
            backtests.append(dist_signal)
        if trading[1] == end_date:
            break

    # backtests = [descriptive_stats(backtest) for backtest in backtests]
    descriptive_frames = descriptive_frame(
        pd.concat(backtests, keys=range(len(backtests))), show_progress_bar=show_progress_bar
    )
    trading_period_days = compute_period_length(training_delta_raw)
    multiindex_from_product_cols = compute_cols_from_freq([freq], [method])
    aggregated = aggregate(
        [descriptive_frames],
        columns_to_pick=standard_result_metrics_from_desc_stats,
        trading_period_days=[trading_period_days],
        multiindex_from_product_cols=multiindex_from_product_cols,
        returns_nonzero=True,
        trades_nonzero=True,
    )
    serializable_columns = ['/'.join(x) for x in aggregated.columns.to_flat_index().values]
    aggregated.columns = serializable_columns
    for col in aggregated.columns:
        tune.track.log(name=col, **aggregated[col].to_dict())


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


