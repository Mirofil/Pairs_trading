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
import ray
from dateutil.relativedelta import relativedelta
from p_tqdm import p_map
from ray import tune
from tqdm import tqdm
from joblib import Parallel, delayed

from pairs.analysis import (
    aggregate,
    descriptive_frame,
    descriptive_stats,
    infer_periods,
    compute_cols_from_freq,
    compute_period_length_in_days,
)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import beautify, standardize_results
from pairs.helpers import *
from pairs.helpers import latexsave
from pairs.scripts.latex.helpers import *
from pairs.pairs_trading_engine import pick_range, backtests_up_to_date, change_txcost_in_backtests, calculate_new_experiments_txcost
from pairs.scripts.latex.loaders import join_backtests_by_id
from pairs.scripts.paper2.helpers import ts_stats, nya_stats
from pairs.scripts.paper2.loaders import process_experiment, load_experiment
from pairs.scripts.paper2.subperiods import nineties, dotcom, financial_crisis, inbetween_crises, modern



# /mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/paper2/analysis
# analysis = ray.tune.Analysis(
#     experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/"
# ).dataframe()
all_subperiods = [nineties,dotcom, inbetween_crises, financial_crisis, modern]

for period in all_subperiods:   
    exp = process_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=[0,0.005])


period = dotcom
analysis = load_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=[0,0.005])


analysis.loc[0, 'backtests'].loc[0].loc['ZTRxNUV'].loc['1990/07/01':'1990-09-28']


change_txcost_in_backtests(generated["backtests"], old_txcost=generated["txcost"], new_txcost=new_txcost)
calculate_new_experiments_txcost(analysis, [0.000])
change_txcost_in_backtest(backtest,0.05,0)


def calculate_timeframes(start_date, i, jump_delta, formation_delta, training_delta):
    formation = (
        start_date + i * jump_delta,
        start_date + formation_delta + i * jump_delta,
    )
    trading = (formation[1], formation[1] + training_delta)
    return {"formation": formation, "trading": trading}


def sort_aggs_by_stat(aggs, stat):
    """Returns the indices that would sort aggregateds by the desired statistic (Monthly profit, ..) """
    aggs = aggs.apply(lambda y: y.loc[stat]).values
    aggs = (-np.array([x[0] for x in aggs])).argsort()
    return aggs


def compute_aggregated_and_find_best_config(
    analysis: pd.DataFrame,
    sort_by="Monthly profit",
    descs: pd.DataFrame = None,
):

    if descs is None:
        descs = analysis["descs"]

    stats_df = pd.DataFrame(index=analysis.index)

    stats_df["trading_days"] = analysis["config/pairs_deltas"].apply(
        lambda x: x["training_delta"][0] * 30 + x["training_delta"][1]
    )
    stats_df["number_of_trades_monthly"] = analysis["config/pairs_deltas"].apply(
        lambda x: (x["training_delta"][0] * 30 + x["training_delta"][1]) / 30
    )
    stats_df["aggregate_multiindex_cols"] = analysis.apply(
        lambda row: compute_cols_from_freq([row["freq"]], [row["config/method"]]),
        axis=1,
    )
    stats_df["one_period_in_days"] = analysis.apply(
        lambda row: compute_period_length_in_days(row["freq"]), axis=1
    )
    stats_df = stats_df.join(descs)

    aggs = []
    for idx in tqdm(stats_df.index, desc="Constructing aggregated statistics"):
        row = stats_df.loc[idx]
        aggs.append(
            aggregate(
                [row["descs"]],
                None,
                [row["trading_days"]],
                row["aggregate_multiindex_cols"],
                returns_nonzero=True,
                trades_nonzero=True,
            )
        )
    stats_df["aggregated"] = aggs
    analysis["aggregated"] = aggs

    stats_df = stats_df.loc[sort_aggs_by_stat(stats_df["aggregated"], sort_by)]
    return stats_df


def analyse_top_n(
    analysis: pd.DataFrame,
    best_configs: pd.DataFrame,
    top_n: int = 20,
    fixed_params: Dict = None,
):
    sorted_analysis = analysis.loc[best_configs.index]
    top_n_results = sorted_analysis.iloc[:top_n]
    if fixed_params is not None:
        sorted_analysis = find_scenario(top_n_results, params=fixed_params)
        print(
            f"Analysis was narrowed down to {len(sorted_analysis)} due to fixed params"
        )
    average_aggregated = top_n_results["aggregated"].sum() / len(
        top_n_results["aggregated"]
    )

    important_params = [
        "freq",
        "lag",
        "txcost",
        "jump",
        "method",
        "dist_num",
        "pairs_deltas",
        "confidence",
        "threshold",
        "stoploss",
    ]

    for param in important_params:
        print(sorted_analysis["config/" + param].astype(str).value_counts())
    return average_aggregated


analysis = join_backtests_by_id(analysis, ids=range(144))

trimmed_backtests = [
    backtests_up_to_date(
        backtests,
        min_formation_period_start="1990/1/1",
        max_trading_period_end="2000/01/01",
    )
    for backtests in tqdm(analysis["backtests"], desc="Trimming backtests")
]

backtests_up_to_date(
    analysis,
    min_formation_period_start="1990/1/1",
    max_trading_period_end="2000/01/01",
)

trimmed_backtests = pd.DataFrame(
    pd.Series(trimmed_backtests, index=analysis.index, name="trimmed_backtests")
)
results = p_map(descriptive_frame, trimmed_backtests, num_cpus=40)
results = [
    descriptive_frame(backtests)
    for backtests in tqdm(trimmed_backtests, desc="Desc frames")
]
results = Parallel(n_jobs=8,verbose=10)(delayed(descriptive_frame)(backtests) for backtests in tqdm(trimmed_backtests))
results = pd.DataFrame(pd.Series(results, index=analysis.index, name="descs"))



find_scenario(
    analysis,
    {
        "lag": 1,
        "txcost": 0.003,
        "dist_num": 20,
        "pairs_deltas/formation_delta": "[6, 0, 0]",
    },
)
best_configs = compute_aggregated_and_find_best_config(analysis)

aggregate(
    [results[78]], None, [60], [["Daily"], ["Dist"]]
)  # this is benchmark literature conf


sort_aggs_by_stat(best_configs["aggregated"], "Monthly profit")
analyse_top_n(
    analysis,
    compute_aggregated_and_find_best_config(analysis, "1995/1/1", "2000/1/1", "Annualized Sharpe"),
    20,
    {"lag": 0},
)


nya_stats(start_date='1990/1/1', end_date='2000/1/1')
