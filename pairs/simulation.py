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
    trading_delta_raw = params["pairs_deltas"]["training_delta"]
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
    trading_delta = relativedelta(
        months=trading_delta_raw[0],
        days=trading_delta_raw[1],
        hours=trading_delta_raw[2],
    )
    jump_delta = relativedelta(months=jump[0], days=jump[1], hours=jump[2])
    trading_period_days = compute_period_length(trading_delta_raw)
    multiindex_from_product_cols = compute_cols_from_freq([freq], [method])

    backtests = []
    logging_params = []
    logging_metrics = []
    total_iters = 0

    for i in tqdm(
        range(50000),
        desc="Starting nth iteration of the formation-trading loop",
        disable=not show_progress_bar,
    ):
        total_iters = total_iters + 1 
        artifacts = {}
        formation = (
            start_date + i * jump_delta,
            start_date + formation_delta + i * jump_delta,
        )
        trading = (formation[1], formation[1] + trading_delta)
        print(i)
        print(formation)
        print(trading)
        if trading[1] > end_date:
            if truncate == True:
                trading = (trading[0], end_date)
            else:
                break
        if trading[1] < formation[1]:
            break

        if trading[0] >= end_date:
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
            distances = distance(head, num=200000, method="modern")
            cointed = find_integrated(head, confidence=confidence, show_progress_bar=show_progress_bar)
            if len(cointed) == 0:
                continue
            k = cointegration_mixed(
                cointed,
                distances["viable_pairs"],
                desired_num=dist_num,
                confidence=confidence,
                show_progress_bar=show_progress_bar,
            )
            if len(k) == 0:
                continue

            short_y = pick_range(preprocessed, formation[0], trading[1])
            spreads = calculate_spreads(
                short_y,
                [item[0] for item in k],
                timeframe=formation,
                betas=[item[1] for item in k],
                show_progress_bar=show_progress_bar
            )
            spreads.sort_index(inplace=True)

        elif "dist" == method:
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
            trading_delta=trading_delta,

        )

        def log_metrics_with_retries(aggregated):
            result = []
            for col in aggregated.columns:
                result.append(aggregated[col].to_dict())
            return result

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
        
        logging_params.append(log_params_with_retries(aggregated,params, threshold,lag,txcost,stoploss,formation,trading,UNIQUE_ID))
        logging_metrics.append(log_metrics_with_retries(aggregated))

        if i % 100 == 0:
            try:
                ray.tune.track.log(**logging_params[i],**logging_metrics[i][0],iteration=str(i))
            except:
                pass
        # NOTE Saving turns out to be taking too much spave so no saving..
        # for artifact_name, artifact_df in artifacts.items():
        #     artifact_df.to_parquet(f"{artifact_name}.parquet")
        #     mlflow.log_artifact(f"{artifact_name}.parquet")
        #     os.remove(f"{artifact_name}.parquet")

        if trading[1] == end_date:
            break

    # for iteration in range(total_iters):
    #     with mlflow.start_run(nested=True):
    #         mlflow.log_params(logging_params[iteration])
    #         for model_metrics in logging_metrics[iteration]:
    #             mlflow.log_metrics(model_metrics)

    aggregated = method_independent_part(
        signals=backtests,
        keys=range(len(backtests)),
        trading_period_days=trading_period_days,
        multiindex_from_product_cols=multiindex_from_product_cols,
        trading_delta=trading_delta,

    )

    (pd.concat(backtests, keys=range(len(backtests)))).to_parquet('aggregated.parquet')

    return {"backtests":backtests, 'aggregated':aggregated}


def method_independent_part(
    signals: List[pd.DataFrame], keys, trading_period_days, multiindex_from_product_cols, trading_delta=None
):
    descriptive_frames = descriptive_frame(
        pd.concat(signals, keys=keys), show_progress_bar=False, trading_delta=trading_delta
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
