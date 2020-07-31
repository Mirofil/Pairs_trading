import datetime
import glob
import os
import pickle
import timeit
from typing import *
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import beautify, standardize_results
from pairs.helpers import *
from pairs.helpers import latexsave
from pairs.scripts.latex.helpers import *
from pairs.pairs_trading_engine import (
    pick_range,
    backtests_up_to_date,
    change_txcost_in_backtests,
    calculate_new_experiments_txcost,
)
from pairs.scripts.paper2.loaders import join_backtests_by_id
from pairs.scripts.paper2.helpers import ts_stats, nya_stats, method_mapping, format_aggregateds, select_rolling_best, find_closest_params
from pairs.scripts.paper2.loaders import process_experiment, load_experiment
from pairs.analysis import find_scenario
from pairs.ray_analysis import compute_aggregated_and_sort_by, analyse_top_n

def rolling_best_summary(teacher_summary, available_params, subperiods, univ=None, should_beautify=True, metric='avg', suffix=""):
    optimals = [
        select_rolling_best(teacher_summary, available_params, metric=metric)[col]
        for col in select_rolling_best(teacher_summary, available_params).columns
    ]
    analyses = []
    for period, optimal in zip(subperiods, optimals):
        # period.analysis["config/pairs_deltas"] = period.analysis["config/pairs_deltas"].apply(convert_params_deltas_to_multiplier)
        fixed_params_base = {"lag": 1, "config/txcost": 0.003}
        fixed_params = {**fixed_params_base, **optimal.to_dict()}
        fixed_params = {k: round(v, 3) for k, v in fixed_params.items()}
        #NOTE when taking year-by-year, the first year wont have many of the pairs_deltas scenarios because they are too long. 0.5 is a quick hack to make sure we take the next closest
        if fixed_params["pairs_deltas"] not in period.analysis["pairs_deltas"].values:
            fixed_params["pairs_deltas"] = 0.5
        result = analyse_top_n(
            analysis=period.analysis,
            best_configs=compute_aggregated_and_sort_by(
                period.analysis, sort_by="Annualized Sharpe"
            ),
            top_n=1,
            fixed_params_before=fixed_params,
        )
        analyses.append(result)

    for analysis in analyses:
        # analysis["agg_avg"] = standardize_results(analysis["agg_avg"])
        if should_beautify:
            analysis["agg_avg"] = beautify(analysis["agg_avg"])

    aggregateds = pd.concat([analysis["agg_avg"] for analysis in analyses], axis=1, keys = [subperiod.table_name for subperiod in subperiods])
    param_avgs = pd.concat([pd.Series(analysis["param_avg"]) for analysis in analyses], axis=1, keys = [subperiod.table_name for subperiod in subperiods])

    if univ is not None:
        if not should_beautify:
            latexsave(
                beautify(aggregateds),
                os.path.join(univ.save_path_tables, "all_periods_aggregateds"+suffix),
            )
        else:
            latexsave(
                aggregateds,
                os.path.join(univ.save_path_tables, "all_periods_aggregateds"+suffix),
            )
        latexsave(
            param_avgs, os.path.join(univ.save_path_tables, "all_periods_param_avgs"+suffix)
        )

    return {"aggregateds":aggregateds, "param_avgs":param_avgs}


def all_periods_summary(
    subperiods,
    new_txcosts,
    ids=None,
    top_n=3,
    fixed_params_before={"lag": 1, "config/txcost": 0.003},
    experiment_dir=[
        "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
        "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow_covid/",
    ],
    sort_by="Annualized Sharpe",
    univ=None,
    force=False,
    should_beautify=True,
    suffix=""
):

    aggregateds = []
    param_avgs = []
    param_medians = []
    value_counts = []
    for period in subperiods:
        analysis = load_experiment(
            subperiod=period,
            ids=ids,
            experiment_dir=experiment_dir,
            new_txcosts=new_txcosts,
            force=force
        )
        period.analysis = analysis
        fixed_params_before["config/txcost"] = period.preferred_txcost

        top_n_analysis = analyse_top_n(
            analysis=analysis,
            best_configs=compute_aggregated_and_sort_by(analysis, sort_by=sort_by),
            top_n=top_n,
            fixed_params_before=fixed_params_before,
        )
        aggregateds.append(top_n_analysis["agg_avg"])
        param_avgs.append(pd.Series(top_n_analysis["param_avg"]))
        param_medians.append(pd.Series(top_n_analysis["param_median"]))
        value_counts.append(top_n_analysis["value_counts"])

    aggregateds = pd.concat(
        aggregateds, axis=1, keys=[subperiod.table_name for subperiod in subperiods]
    )
    param_avgs = pd.concat(
        param_avgs, axis=1, keys=[subperiod.table_name for subperiod in subperiods]
    )
    param_medians = pd.concat(param_medians, axis = 1, keys=[subperiod.table_name for subperiod in subperiods])

    # aggregateds.to_parquet(
    #     os.path.join(experiment_dir, "all_periods_aggregateds.parquet")
    # )
    # param_avgs.to_parquet(
    #     os.path.join(experiment_dir, "all_periods_param_avgs.parquet")
    # )

    if univ is not None:
        latexsave(
            beautify(aggregateds),
            os.path.join(univ.save_path_tables, "all_periods_aggregateds"+suffix),
        )
        latexsave(
            param_avgs, os.path.join(univ.save_path_tables, "all_periods_param_avgs"+suffix)
        )
    if should_beautify is True:

        aggregateds = pd.concat([beautify(aggregateds[[col]]) for col in aggregateds.columns], axis=1)
    
    else:
        aggregateds = pd.concat([aggregateds[[col]] for col in aggregateds.columns], axis=1)


    return {"aggregateds": aggregateds, "param_avgs": param_avgs, "param_medians":param_medians}


def subperiods_summary(subperiods: List, univ=None):
    table = pd.DataFrame(
        [],
        index=["Starting date", "Ending date", "Tx. cost", "Short cost"],
        columns=[subperiod.table_name for subperiod in subperiods],
    )
    for subperiod in subperiods:
        table.loc[
            "Starting date", subperiod.table_name
        ] = subperiod.start_date.strftime("%Y-%m-%d")
        table.loc["Ending date", subperiod.table_name] = subperiod.end_date.strftime(
            "%Y-%m-%d"
        )
        table.loc["Tx. cost", subperiod.table_name] = subperiod.preferred_txcost
        table.loc["Short cost", subperiod.table_name] = "0.6%"
    if univ is not None:
        latexsave(table, os.path.join(univ.save_path_tables, "subperiods_summary"))
    return table


def lag_txcost_summary(
    subperiods,
    new_txcosts: List[float],
    all_lags: List[int],
    top_n: int = 3,
    experiment_dir: List[str] = [
        "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
        "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow_covid/",
    ],
    sort_by: str = "Annualized Sharpe",
    univ=None,
    convert_to_multipliers=True,
):
    pairs_deltas_2id = {
        "{'training_delta': [6 ,0, 0], 'formation_delta': [12, 0, 0]}": 1,
        "{'training_delta': [3 ,0, 0], 'formation_delta': [6, 0, 0]}": 0.5,
        "{'training_delta': [1 ,0, 0], 'formation_delta': [2, 0, 0]}": 0.16,
        "{'training_delta': [10 ,0, 0], 'formation_delta': [20, 0, 0]}": 1.66,
        "{'formation_delta': [12, 0, 0], 'training_delta': [6, 0, 0]}": 1,
        "{'formation_delta': [6, 0, 0], 'training_delta': [3, 0, 0]}": 0.5,
        "{'formation_delta': [2, 0, 0], 'training_delta': [1, 0, 0]}": 0.16,
        "{'formation_delta': [20, 0 0], 'training_delta': [10, 0, 0]}": 1.66,
    }

    id_2pairs_deltas = {
        0.5: '{"training_delta":[3,0,0], "formation_delta":[6,0,0]}',
        1: '{"training_delta":[6,0,0], "formation_delta":[12,0,0]}',
        0.16: '{"training_delta":[1,0,0], "formation_delta":[2,0,0]}',
        1.66: '{"training_delta":[10,0,0], "formation_delta":[20,0,0]}',
    }
    new_tables = []
    for period in subperiods:
        table = pd.DataFrame([], index=all_lags, columns=new_txcosts)

        aggregateds = []
        param_avgs = []
        analysis = load_experiment(
            subperiod=period, experiment_dir=experiment_dir, new_txcosts=new_txcosts
        )
        for lag in all_lags:
            for txcost in new_txcosts:
                top_n_analysis = analyse_top_n(
                    analysis=analysis,
                    best_configs=compute_aggregated_and_sort_by(
                        analysis, sort_by=sort_by
                    ),
                    top_n=top_n,
                    fixed_params_before={"lag": lag, "txcost": txcost},
                )
                table.loc[lag, txcost] = [find_closest_params(pd.Series(top_n_analysis["param_avg"]).astype(np.float32).round(2))]
        table = table.applymap(lambda x: x[0])
        rows = []
        for lag in table.index:
            interim = []
            for txcost in table.columns:
                interim.append(table.loc[lag, txcost])
            interim = pd.concat(interim, axis=1, keys=table.columns)
            rows.append(interim)

        if convert_to_multipliers:
            for i in range(len(rows)):
                if (isinstance(rows[i].loc["pairs_deltas"], (dict, str))):
                    delta = str(rows[i].loc["pairs_deltas"].astype(str).iloc[0])
                    rows[i].loc["pairs_deltas"] = pairs_deltas_2id[
                        delta
                    ]

        for i in range(len(rows)):
            rows[i] = rows[i].rename(
                {
                    "freq": "Frequency",
                    "lag": "Lag",
                    "dist_num": "# of pairs",
                    "pairs_deltas": "Period length mult.",
                    "confidence": "Confidence",
                    "threshold": "Threshold",
                }
            )
        new_table = pd.concat(rows, keys=table.index)
        new_table = pd.concat([new_table], keys=["Lag"])
        new_table = pd.concat([new_table], keys=["Transaction cost"], axis=1)
        if univ is not None:
            latexsave(new_table, os.path.join(univ.save_path_tables, "lag_txcost"))
        new_tables.append(new_table)
    
    return new_tables