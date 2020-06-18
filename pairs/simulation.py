#%%
import datetime
import os
from typing import *

import numpy as np
import pandas as pd
import ray
from dateutil.relativedelta import relativedelta
from pandas.io.json._normalize import nested_to_record
from ray import tune
from tqdm import tqdm

from pairs.analysis import (
    aggregate, compute_cols_from_freq, compute_period_length,
    descriptive_frame, descriptive_stats, summarize)

from pairs.cointmethod import coint_spread, cointegration, find_integrated

from pairs.distancemethod import distance, distance_spread
from pairs.helpers import data_path, prefilter, preprocess
from pairs.pairs_trading_engine import (calculate_profit, pick_range,
                                  propagate_weights, signals, sliced_norm,
                                  weights_from_signals)
import mlflow
import mlflowhelper

import uuid
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
    start_date = params["start_date"]
    end_date = params["end_date"]
    jump = params["jump"]
    method = params["method"]
    dist_num = params["dist_num"]
    threshold = params["threshold"]
    stoploss = params["stoploss"]
    scenario = params["name"]
    data_path = params["data_path"]
    save_path_results = params["save_path_results"]

    redo_prefiltered = params["redo_prefiltered"]
    redo_preprocessed = params["redo_preprocessed"]
    truncate = params["truncate"]
    volume_cutoff = params["volume_cutoff"]
    show_progress_bar = params["show_progress_bar"]
    saving_method = params["saving_method"]
    dataset = params["dataset"]
    trading_univ = params["trading_univ"]

    UNIQUE_ID = str(uuid.uuid4())


    formation_delta = relativedelta(
        months=formation_delta[0], days=formation_delta[1], hours=formation_delta[2]
    )
    training_delta = relativedelta(
        months=training_delta_raw[0], days=training_delta_raw[1], hours=training_delta_raw[2]
    )
    jump_delta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    trading_period_days = compute_period_length(training_delta_raw)
    multiindex_from_product_cols = compute_cols_from_freq([freq], [method])


    print("Starting " + scenario)
    print("\n")
    # if not os.path.isdir(os.path.join(save, scenario)):
    #     os.mkdir(os.path.join(save, scenario))
    # with open(os.path.join(save, scenario, "parameters" + ".txt"), "w") as tf:
    #     print(params, file=tf)

    backtests = []

    mlflow.set_experiment("Simulation")
    mlflow.set_tracking_uri("file:/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/mlruns/")
    with mlflowhelper.start_run():
        for i in tqdm(
            range(50000), desc="Starting nth iteration of the formation-trading loop", disable = not show_progress_bar
        ):
            artifacts = {}
            with mlflowhelper.start_run(nested=True):
                formation = (start_date + i * jump_delta, start_date + formation_delta + i * jump_delta)
                trading = (formation[1], formation[1] + training_delta)
                if trading[1] > end_date:
                    if truncate == True:
                        trading = (trading[0], end_date)
                    else:
                        break
                if trading[1] < formation[1]:
                    break

                if redo_prefiltered == True:
                    prefiltered = dataset.prefilter(start_date=formation[0], end_date=formation[1])
                    if save_path_results is not None:
                        # np.save(os.path.join(save, str(i) + "x" + str(volume_cutoff), prefiltered))
                        prefiltered.to_parquet(os.path.join(save_path_results, str(i) + "x" + str(volume_cutoff)+".parquet"))
                else:
                    prefiltered_fpath = os.path.join(
                        save_path_results,
                        "prefiltered" + str(volume_cutoff).replace(".", "_") + ".parquet",
                    )
                    if not os.path.isfile(prefiltered_fpath):
                        prefiltered = dataset.prefilter()
                        # np.save(prefiltered_fpath, prefiltered)
                        prefiltered.to_parquet(prefiltered_fpath)
                    else:
                        # prefiltered = np.load(prefiltered_fpath)
                        prefiltered = pd.read_parquet(prefiltered_fpath)

                if redo_preprocessed == True:
                    preprocessed = dataset.preprocess(start_date=formation[0], end_date=formation[1])
                    if save_path_results is not None:
                        preprocessed.to_parquet(os.path.join(save_path_results, str(i) + "y" + str(freq)))
                else:
                    preprocessed_fpath = os.path.join(
                        save_path_results,
                        "preprocessed"
                        + str(freq)
                        + str(volume_cutoff).replace(".", "_")
                        + f".{saving_method}",
                    )
                    if not os.path.isfile(preprocessed_fpath):
                        preprocessed = dataset.prefilter()

                        if saving_method == 'parquet':
                            preprocessed.to_parquet(preprocessed_fpath)
                        elif saving_method == 'pkl':
                            preprocessed.to_pickle(preprocessed_fpath)
                    else:
                        if saving_method == 'parquet':
                            preprocessed = pd.read_parquet(preprocessed_fpath)
                        elif saving_method == 'pkl':
                            preprocessed = pd.read_pickle(preprocessed_fpath)

                if "coint" == method:
                    coint_head = pick_range(preprocessed, formation[0], formation[1])
                    k = cointegration(
                        find_integrated(coint_head), num_of_processes=1
                    )
                    short_y = pick_range(preprocessed, formation[0], trading[1])
                    spreads = coint_spread(
                        short_y,
                        [item[0] for item in k],
                        timeframe=formation,
                        betas=[item[1] for item in k],
                    )
                    spreads.sort_index(inplace=True)

                if "dist" == method:
                    head = pick_range(preprocessed, formation[0], formation[1])
                    distances = distance(head, num=dist_num, show_progress_bar=show_progress_bar)
                    short_y = pick_range(preprocessed, formation[0], trading[1])
                    spreads = distance_spread(short_y, distances["viable_pairs"], formation, show_progress_bar=show_progress_bar)
                    spreads.sort_index(inplace=True)

                trading_signals = signals(
                    spreads,
                    timeframe=trading,
                    formation=formation,
                    lag=lag,
                    threshold=threshold,
                    stoploss=stoploss,
                    num_of_processes=1,
                )
                weights_from_signals(trading_signals, cost=txcost)
                propagate_weights(trading_signals, formation)
                calculate_profit(trading_signals, cost=txcost)
                if ssave_path_resultsave is not None:
                    trading_signals.to_parquet(
                        os.path.join(ssave_path_resultsave, scenario, str(i) + f"{method}_signal.parquet")
                    )
                backtests.append(trading_signals)

                artifacts["trading_signals"]=trading_signals
                artifacts["preprocessed"]=preprocessed
                artifacts["prefiltered"]= prefiltered



                aggregated = method_independent_part(signals=[trading_signals], keys=[len(backtests)-1], trading_period_days=trading_period_days, multiindex_from_product_cols=multiindex_from_product_cols)
                #NOTE there should be only one column - something like Daily/Dist
                for col in aggregated.columns:
                    mlflow.log_metrics(aggregated[col].to_dict())
                mlflow.log_params(params)
                mlflow.log_param("formation", str(formation))
                mlflow.log_param("trading", str(trading))
                mlflow.log_param("UNIQUE_ID", UNIQUE_ID+f"_{i}")

                for artifact_name, artifact_df in artifacts.items():

                    with mlflowhelper.managed_artifact(f"{artifact_name}.parquet") as artifact:
            
                        artifact_df.to_parquet(artifact.get_path())

                if trading[1] == end_date:
                    break

        # descriptive_frames = descriptive_frame(
        #     pd.concat(backtests, keys=range(len(backtests))), show_progress_bar=show_progress_bar
        # )
        # trading_period_days = compute_period_length(training_delta_raw)
        # multiindex_from_product_cols = compute_cols_from_freq([freq], [method])
        # aggregated = aggregate(
        #     [descriptive_frames],
        #     columns_to_pick=standard_result_metrics_from_desc_stats,
        #     trading_period_days=[trading_period_days],
        #     multiindex_from_product_cols=multiindex_from_product_cols,
        #     returns_nonzero=True,
        #     trades_nonzero=True,
        # )
        # serializable_columns = ['/'.join(x) for x in aggregated.columns.to_flat_index().values]
        # aggregated.columns = serializable_columns
        aggregated = method_independent_part(signals=backtests, keys=range(len(backtests)), trading_period_days=trading_period_days, multiindex_from_product_cols=multiindex_from_product_cols)
        #NOTE there should be only one column - something like Daily/Dist
        for col in aggregated.columns:
            tune.track.log(name=col, **aggregated[col].to_dict())
            mlflow.log_metrics(aggregated[col].to_dict())
        mlflow.log_params(params)
        mlflow.log_param("UNIQUE_ID", UNIQUE_ID + "_MASTER")

def method_independent_part(signals:List[pd.DataFrame], keys, trading_period_days, multiindex_from_product_cols):
    descriptive_frames = descriptive_frame(
        pd.concat(signals, keys=keys), show_progress_bar=False
        )
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
    return aggregated


# def stoploss(freqs=["daily"], thresh=[1, 2, 3], stoploss=[2, 3, 4, 5, 6], save=save):

#     scenariod = {
#         "freq": "1D",
#         "lag": 1,
#         "txcost": 0.003,
#         "training_delta": [2, 0, 0],
#         "cutoff": 0.7,
#         "formation_delta": [4, 0, 0],
#         "start": datetime.date(*[2018, 1, 1]),
#         "end": datetime.date(*[2019, 9, 1]),
#         "jump": [1, 0, 0],
#         "methods": ["dist", "coint"],
#         "dist_num": 20,
#         "threshold": 2,
#         "stoploss": 100,
#         "name": "scenariosd12",
#     }
#     scenarioh = {
#         "freq": "1H",
#         "lag": 1,
#         "txcost": 0.003,
#         "training_delta": [0, 10, 0],
#         "cutoff": 0.7,
#         "formation_delta": [0, 20, 0],
#         "start": datetime.date(*[2018, 1, 1]),
#         "end": datetime.date(*[2019, 9, 1]),
#         "jump": [0, 10, 0],
#         "methods": ["dist", "coint"],
#         "dist_num": 20,
#         "threshold": 2,
#         "stoploss": 100,
#         "name": "scenariosh12",
#     }
#     if "daily" in freqs:
#         for i in range(len(thresh)):
#             for j in range(len(stoploss)):
#                 newnamed = scenariod["name"][:-2] + str(thresh[i]) + str(stoploss[j])
#                 if os.path.isfile(
#                     os.path.join(save, newnamed, str(0) + "dist_signal.pkl")
#                 ):
#                     continue
#                 scenariod.update(
#                     {"threshold": thresh[i], "stoploss": stoploss[j], "name": newnamed}
#                 )
#                 simulate(scenariod)
#     if "hourly" in freqs:
#         for i in range(len(thresh)):
#             for j in range(len(stoploss)):
#                 newnameh = scenarioh["name"][:-2] + str(thresh[i]) + str(stoploss[j])
#                 if os.path.isfile(
#                     os.path.join(save, newnameh, str(0) + "coint_signal.pkl")
#                 ):
#                     continue
#                 scenarioh.update(
#                     {"threshold": thresh[i], "stoploss": stoploss[j], "name": newnameh}
#                 )
#                 simulate(scenarioh)

