#%%
import datetime
import os
from typing import *
import time
import numpy as np
import pandas as pd
import ray
import ray.tune.track
from dateutil.relativedelta import relativedelta
from pandas.io.json._normalize import nested_to_record
from ray import tune
from tqdm import tqdm

from pairs.analysis import (
    aggregate,
    compute_cols_from_freq,
    compute_period_length,
    descriptive_frame,
    descriptive_stats,
    summarize,
)
from pairs.config import standard_result_metrics_from_desc_stats
from pairs.cointmethod import cointegration, find_integrated, cointegration_mixed

from pairs.distancemethod import distance
from pairs.helpers import data_path
from pairs.pairs_trading_engine import (
    calculate_profit,
    pick_range,
    propagate_weights,
    signals,
    sliced_norm,
    weights_from_signals,
    calculate_spreads,
)
from pairs.helpers import remake_into_lists
import mlflow
import mlflowhelper
import itertools
import uuid
from retry import retry


def simulate(
    params,
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
    training_delta_raw = params["pairs_deltas"]["training_delta"]
    volume_cutoff = params["volume_cutoff"]
    formation_delta = params["pairs_deltas"]["formation_delta"]
    start_date = params["start_date"]
    end_date = params["end_date"]
    jump = params["jump"]
    method = params["method"]
    dist_num = params["dist_num"]
    threshold = params["threshold"]
    stoploss = params["stoploss"]
    scenario = params["name"]
    save_path_results = params["save_path_results"]
    confidence = params["confidence"]

    redo_prefiltered = params["redo_prefiltered"]
    redo_preprocessed = params["redo_preprocessed"]
    truncate = params["truncate"]
    volume_cutoff = params["volume_cutoff"]
    show_progress_bar = params["show_progress_bar"]
    saving_method = params["saving_method"]
    dataset = params["dataset"]
    trading_univ = params["trading_univ"]
    tracking_uri = params["tracking_uri"]
    # run_ids = params["run_ids"]

    UNIQUE_ID = str(uuid.uuid4())

    formation_delta = relativedelta(
        months=formation_delta[0], days=formation_delta[1], hours=formation_delta[2]
    )
    training_delta = relativedelta(
        months=training_delta_raw[0],
        days=training_delta_raw[1],
        hours=training_delta_raw[2],
    )
    jump_delta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    trading_period_days = compute_period_length(training_delta_raw)
    multiindex_from_product_cols = compute_cols_from_freq([freq], [method])

    backtests = []
    params = []
    metrics = []
    total_iters = 0

    for i in tqdm(
        range(50000),
        desc="Starting nth iteration of the formation-trading loop",
        disable=not show_progress_bar,
    ):
        total_iters = total_iters + 1 
        artifacts = {}
        if i % 2 == 0:
            try:
                ray.tune.track.log(iteration=str(i))
            except:
                pass
        formation = (
            start_date + i * jump_delta,
            start_date + formation_delta + i * jump_delta,
        )
        trading = (formation[1], formation[1] + training_delta)
        if trading[1] > end_date:
            if truncate == True:
                trading = (trading[0], end_date)
            else:
                break
        if trading[1] < formation[1]:
            break

        if redo_prefiltered == True:
            prefiltered = dataset.prefilter(
                start_date=formation[0], end_date=formation[1]
            )
            if save_path_results is not None:
                # np.save(os.path.join(save, str(i) + "x" + str(volume_cutoff), prefiltered))
                prefiltered.to_parquet(
                    os.path.join(
                        save_path_results,
                        str(i) + "x" + str(volume_cutoff) + ".parquet",
                    )
                )
        else:
            prefiltered_fpath = os.path.join(
                save_path_results,
                "prefiltered" + str(volume_cutoff).replace(".", "_") + ".parquet",
            )
            if not os.path.isfile(prefiltered_fpath):
                prefiltered = dataset.prefilter(
                    start_date=formation[0], end_date=formation[1]
                )
                # np.save(prefiltered_fpath, prefiltered)
                prefiltered.to_parquet(prefiltered_fpath)
            else:
                # prefiltered = np.load(prefiltered_fpath)
                prefiltered = pd.read_parquet(prefiltered_fpath)

        if redo_preprocessed == True:
            preprocessed = dataset.preprocess(
                start_date=formation[0], end_date=trading[1]
            )
            if save_path_results is not None:
                preprocessed.to_parquet(
                    os.path.join(save_path_results, str(i) + "y" + str(freq))
                )
        else:
            preprocessed_fpath = os.path.join(
                save_path_results,
                "preprocessed"
                + str(freq)
                + str(volume_cutoff).replace(".", "_")
                + f".{saving_method}",
            )
            if not os.path.isfile(preprocessed_fpath):
                preprocessed = dataset.prefilter(
                    start_date=formation[0], end_date=trading[1]
                )

                if saving_method == "parquet":
                    preprocessed.to_parquet(preprocessed_fpath)
                elif saving_method == "pkl":
                    preprocessed.to_pickle(preprocessed_fpath)
            else:
                if saving_method == "parquet":
                    preprocessed = pd.read_parquet(preprocessed_fpath)
                elif saving_method == "pkl":
                    preprocessed = pd.read_pickle(preprocessed_fpath)

        if "coint" == method:
            head = pick_range(preprocessed, formation[0], formation[1])
            # k = cointegration(find_integrated(coint_head), num_of_processes=1)
            distances = distance(head, num=20000, method="modern")
            cointed = find_integrated(head)

            k = cointegration_mixed(
                cointed,
                distances["viable_pairs"],
                desired_num=dist_num,
                confidence=confidence,
                show_progress_bar=show_progress_bar,
            )

            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = calculate_spreads(
                short_y,
                [item[0] for item in k],
                timeframe=formation,
                betas=[item[1] for item in k],
            )
            spreads.sort_index(inplace=True)

        if "dist" == method:
            head = pick_range(preprocessed, formation[0], formation[1])
            distances = distance(
                head, num=dist_num, show_progress_bar=show_progress_bar
            )
            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = calculate_spreads(
                short_y,
                distances["viable_pairs"],
                formation,
                show_progress_bar=show_progress_bar,
            )
            spreads.sort_index(inplace=True)

        trading_signals = signals(
            spreads,
            start_date=start_date,
            end_date=end_date,
            trading_timeframe=trading,
            formation=formation,
            lag=lag,
            threshold=threshold,
            stoploss=stoploss,
            num_of_processes=1,
        )
        weights_from_signals(trading_signals, cost=txcost)
        propagate_weights(trading_signals, formation_timeframe=formation)
        calculate_profit(trading_signals, cost=txcost)

        if save_path_results is not None:
            trading_signals.to_parquet(
                os.path.join(
                    save_path_results,
                    scenario,
                    str(i) + f"{method}_signal.parquet",
                )
            )

        backtests.append(trading_signals)

        artifacts["trading_signals"] = trading_signals
        artifacts["preprocessed"] = preprocessed
        artifacts["prefiltered"] = prefiltered

        aggregated = method_independent_part(
            signals=[trading_signals],
            keys=[len(backtests) - 1],
            trading_period_days=trading_period_days,
            multiindex_from_product_cols=multiindex_from_product_cols,
        )
        # NOTE there should be only one column - something like Daily/Dist

        def log_metrics_with_retries(aggregated):
            result = []
            for col in aggregated.columns:
                result.append(aggregated[col].to_dict())
            return result

        @retry(delay=1, jitter=(0.25, 0.5))
        def log_params_with_retries(
            aggregated,
            params,
            threshold,
            lag,
            txcost,
            stoploss,
            formation,
            trading,
            UNIQUE_ID,
        ):
            for col in aggregated.columns:
                mlflow.log_metrics(aggregated[col].to_dict())

            all_params = {
                **params,
                "specific_threshold": threshold,
                "specific_lag": lag,
                "specific_txcost": txcost,
                "specific_stoploss": stoploss,
                "formation": formation,
                "trading": trading,
                "UNIQUE_ID": UNIQUE_ID,
            }
            return all_params
        
        params.append(log_params_with_retries)
        metrics.append(log_metrics_with_retries)


        # for col in aggregated.columns:
        #     mlflow.log_metrics(aggregated[col].to_dict())
        # mlflow.log_params(params)
        # mlflow.log_param("specific_threshold", threshold)
        # mlflow.log_param("specific_lag", lag)
        # mlflow.log_param("specific_txcost", txcost)
        # mlflow.log_param("specific_stoploss", stoploss)
        # mlflow.log_param("formation", str(formation))
        # mlflow.log_param("trading", str(trading))
        # mlflow.log_param("UNIQUE_ID", UNIQUE_ID + f"_{i}")

        # mlflow.log_params(log_params_with_retries(
        #     aggregated,
        #     params,
        #     threshold,
        #     lag,
        #     txcost,
        #     stoploss,
        #     formation,
        #     trading,
        #     UNIQUE_ID,
        # ))
        # for item in log_metrics_with_retries(aggregated):
        #     mlflow.log_metrics(item)

        # NOTE Saving turns out to be taking too much spave so no saving..
        # for artifact_name, artifact_df in artifacts.items():
        #     artifact_df.to_parquet(f"{artifact_name}.parquet")
        #     mlflow.log_artifact(f"{artifact_name}.parquet")
        #     os.remove(f"{artifact_name}.parquet")

        if trading[1] == end_date:
            break

    for iteration in range(total_iters):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params[iteration])
            for model_metrics in metrics[iteration]:
                mlflow.log_metrics(model_metrics)

    aggregated = method_independent_part(
        signals=backtests,
        keys=range(len(backtests)),
        trading_period_days=trading_period_days,
        multiindex_from_product_cols=multiindex_from_product_cols,
    )
    # NOTE there should be only one column - something like Daily/Dist
    for col in aggregated.columns:
        # ray.tune.track.log(name=col, **aggregated[col].to_dict())
        mlflow.log_metrics(aggregated[col].to_dict())
    mlflow.log_params(params)
    mlflow.log_param("UNIQUE_ID", UNIQUE_ID + "_MASTER")

    # NOTE this can be used later to process the resutls dataframe from MLflow

    # x=mlflow.search_runs([experiment_id])
    # x = x.loc[x["tags.mlflow.parentRunId"].isin([run_ids])]
    # for parent_run_id, group in x.groupby(by='tags.mlflow.parentRunId'):
    #     with mlflow.start_run(run_id=parent_run_id):


def simulate_smart(
    params,
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
    training_delta_raw = params["pairs_deltas"]["training_delta"]
    volume_cutoff = params["volume_cutoff"]
    formation_delta = params["pairs_deltas"]["formation_delta"]
    start_date = params["start_date"]
    end_date = params["end_date"]
    jump = params["jump"]
    method = params["method"]
    dist_num = params["dist_num"]
    threshold = params["threshold"]
    stoploss = params["stoploss"]
    scenario = params["name"]
    save_path_results = params["save_path_results"]
    confidence = params["confidence"]

    redo_prefiltered = params["redo_prefiltered"]
    redo_preprocessed = params["redo_preprocessed"]
    truncate = params["truncate"]
    volume_cutoff = params["volume_cutoff"]
    show_progress_bar = params["show_progress_bar"]
    saving_method = params["saving_method"]
    dataset = params["dataset"]
    trading_univ = params["trading_univ"]
    tracking_uri = params["tracking_uri"]
    # run_ids = params["run_ids"]

    lag, stoploss, threshold, txcost = remake_into_lists(
        lag, stoploss, threshold, txcost
    )

    UNIQUE_ID = str(uuid.uuid4())

    formation_delta = relativedelta(
        months=formation_delta[0], days=formation_delta[1], hours=formation_delta[2]
    )
    training_delta = relativedelta(
        months=training_delta_raw[0],
        days=training_delta_raw[1],
        hours=training_delta_raw[2],
    )
    jump_delta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    trading_period_days = compute_period_length(training_delta_raw)
    multiindex_from_product_cols = compute_cols_from_freq([freq], [method])

    step2_arg_product = list(itertools.product(lag, threshold, stoploss, txcost))

    print("Starting " + scenario)
    print("\n")
    # if not os.path.isdir(os.path.join(save, scenario)):
    #     os.mkdir(os.path.join(save, scenario))
    # with open(os.path.join(save, scenario, "parameters" + ".txt"), "w") as tf:
    #     print(params, file=tf)

    backtests_store = [[] for prod in step2_arg_product]
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    experiment_id = mlflowhelper.set_experiment(scenario)

    run_ids = []

    for _ in step2_arg_product:
        time.sleep(0.1)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_ids.append(run.info.run_id)
            time.sleep(0.1)

    for i in tqdm(
        range(50000),
        desc="Starting nth iteration of the formation-trading loop",
        disable=not show_progress_bar,
    ):
        artifacts = {}
        if i % 2 == 0:
            ray.tune.track.log(iteration=str(i))
        formation = (
            start_date + i * jump_delta,
            start_date + formation_delta + i * jump_delta,
        )
        trading = (formation[1], formation[1] + training_delta)
        if trading[1] > end_date:
            if truncate == True:
                trading = (trading[0], end_date)
            else:
                break
        if trading[1] < formation[1]:
            break

        if redo_prefiltered == True:
            prefiltered = dataset.prefilter(
                start_date=formation[0], end_date=formation[1]
            )
            if save_path_results is not None:
                # np.save(os.path.join(save, str(i) + "x" + str(volume_cutoff), prefiltered))
                prefiltered.to_parquet(
                    os.path.join(
                        save_path_results,
                        str(i) + "x" + str(volume_cutoff) + ".parquet",
                    )
                )
        else:
            prefiltered_fpath = os.path.join(
                save_path_results,
                "prefiltered" + str(volume_cutoff).replace(".", "_") + ".parquet",
            )
            if not os.path.isfile(prefiltered_fpath):
                prefiltered = dataset.prefilter(
                    start_date=formation[0], end_date=formation[1]
                )
                # np.save(prefiltered_fpath, prefiltered)
                prefiltered.to_parquet(prefiltered_fpath)
            else:
                # prefiltered = np.load(prefiltered_fpath)
                prefiltered = pd.read_parquet(prefiltered_fpath)

        if redo_preprocessed == True:
            preprocessed = dataset.preprocess(
                start_date=formation[0], end_date=trading[1]
            )
            if save_path_results is not None:
                preprocessed.to_parquet(
                    os.path.join(save_path_results, str(i) + "y" + str(freq))
                )
        else:
            preprocessed_fpath = os.path.join(
                save_path_results,
                "preprocessed"
                + str(freq)
                + str(volume_cutoff).replace(".", "_")
                + f".{saving_method}",
            )
            if not os.path.isfile(preprocessed_fpath):
                preprocessed = dataset.prefilter(
                    start_date=formation[0], end_date=trading[1]
                )

                if saving_method == "parquet":
                    preprocessed.to_parquet(preprocessed_fpath)
                elif saving_method == "pkl":
                    preprocessed.to_pickle(preprocessed_fpath)
            else:
                if saving_method == "parquet":
                    preprocessed = pd.read_parquet(preprocessed_fpath)
                elif saving_method == "pkl":
                    preprocessed = pd.read_pickle(preprocessed_fpath)

        if "coint" == method:
            head = pick_range(preprocessed, formation[0], formation[1])
            # k = cointegration(find_integrated(coint_head), num_of_processes=1)
            distances = distance(head, num=20000, method="modern")
            cointed = find_integrated(head)

            k = cointegration_mixed(
                cointed,
                distances["viable_pairs"],
                desired_num=dist_num,
                confidence=confidence,
                show_progress_bar=show_progress_bar,
            )

            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = calculate_spreads(
                short_y,
                [item[0] for item in k],
                timeframe=formation,
                betas=[item[1] for item in k],
            )
            spreads.sort_index(inplace=True)

        if "dist" == method:
            head = pick_range(preprocessed, formation[0], formation[1])
            distances = distance(
                head, num=dist_num, show_progress_bar=show_progress_bar
            )
            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = calculate_spreads(
                short_y,
                distances["viable_pairs"],
                formation,
                show_progress_bar=show_progress_bar,
            )
            spreads.sort_index(inplace=True)

        for arg_tuple, run_id, backtests, idx in zip(
            itertools.product(lag, threshold, stoploss, txcost),
            run_ids,
            backtests_store,
            range(len(run_ids)),
        ):
            lag_, threshold_, stoploss_, txcost_ = list(arg_tuple)
            with mlflowhelper.start_run(run_id=run_id):
                time.sleep(0.25)
                with mlflowhelper.start_run(nested=True):
                    time.sleep(0.25)
                    # The best way would be to have a regular pipeline DAG here and somehow do plate-like notation for redistribution of parameters.
                    # since this part is independent of the previous
                    trading_signals = signals(
                        spreads,
                        start_date=start_date,
                        end_date=end_date,
                        trading_timeframe=trading,
                        formation=formation,
                        lag=lag_,
                        threshold=threshold_,
                        stoploss=stoploss_,
                        num_of_processes=1,
                    )
                    weights_from_signals(trading_signals, cost=txcost_)
                    propagate_weights(trading_signals, formation_timeframe=formation)
                    calculate_profit(trading_signals, cost=txcost_)

                    if save_path_results is not None:
                        trading_signals.to_parquet(
                            os.path.join(
                                save_path_results,
                                scenario,
                                str(i) + f"{method}_signal.parquet",
                            )
                        )

                    backtests_store[idx].append(trading_signals)

                    artifacts["trading_signals"] = trading_signals
                    artifacts["preprocessed"] = preprocessed
                    artifacts["prefiltered"] = prefiltered

                    aggregated = method_independent_part(
                        signals=[trading_signals],
                        keys=[len(backtests) - 1],
                        trading_period_days=trading_period_days,
                        multiindex_from_product_cols=multiindex_from_product_cols,
                    )
                    # NOTE there should be only one column - something like Daily/Dist
                    for col in aggregated.columns:
                        mlflow.log_metrics(aggregated[col].to_dict())
                    mlflow.log_params(params)
                    mlflow.log_param("specific_threshold", threshold_)
                    mlflow.log_param("specific_lag", lag_)
                    mlflow.log_param("specific_txcost", txcost_)
                    mlflow.log_param("specific_stoploss", stoploss_)
                    mlflow.log_param("formation", str(formation))
                    mlflow.log_param("trading", str(trading))
                    mlflow.log_param("UNIQUE_ID", UNIQUE_ID + f"_{i}")

                    # for artifact_name, artifact_df in artifacts.items():

                    #     with mlflowhelper.managed_artifact(
                    #         f"{artifact_name}.parquet"
                    #     ) as artifact:

                    #         artifact_df.to_parquet(artifact.get_path())

                    # for artifact_name, artifact_df in artifacts.items():
                    #     artifact_df.to_parquet(f"{artifact_name}.parquet")
                    #     mlflow.log_artifact(f"{artifact_name}.parquet")
                    #     os.remove(f"{artifact_name}.parquet")

        if trading[1] == end_date:
            break

    for backtests, run_id, idx in zip(backtests_store, run_ids, range(len(run_ids))):
        # experiment_id = mlflow.get_experiment_by_name(scenario).experiment_id
        with mlflowhelper.start_run(run_id=run_id):
            print(backtests)
            print(run_id)
            aggregated = method_independent_part(
                signals=backtests_store[idx],
                keys=range(len(backtests_store[idx])),
                trading_period_days=trading_period_days,
                multiindex_from_product_cols=multiindex_from_product_cols,
            )
            print(aggregated)
            # NOTE there should be only one column - something like Daily/Dist
            for col in aggregated.columns:
                # ray.tune.track.log(name=col, **aggregated[col].to_dict())
                mlflow.log_metrics(aggregated[col].to_dict())
            mlflow.log_params(params)
            mlflow.log_param("UNIQUE_ID", UNIQUE_ID + "_MASTER")

    # NOTE this can be used later to process the resutls dataframe from MLflow

    # x=mlflow.search_runs([experiment_id])
    # x = x.loc[x["tags.mlflow.parentRunId"].isin([run_ids])]
    # for parent_run_id, group in x.groupby(by='tags.mlflow.parentRunId'):
    #     with mlflow.start_run(run_id=parent_run_id):


def method_independent_part(
    signals: List[pd.DataFrame], keys, trading_period_days, multiindex_from_product_cols
):
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
    serializable_columns = [
        "/".join(x) for x in aggregated.columns.to_flat_index().values
    ]
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

