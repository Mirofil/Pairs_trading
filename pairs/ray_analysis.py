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
from pairs.scripts.paper2.loaders import join_backtests_by_id
from pairs.scripts.paper2.helpers import ts_stats, nya_stats
from pairs.scripts.paper2.loaders import process_experiment, load_experiment
from pairs.scripts.paper2.subperiods import nineties, dotcom, financial_crisis, inbetween_crises, modern
from pairs.analysis import find_scenario

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
    fixed_params_before: Dict = None,
    fixed_params_after: Dict = None,
):
    if fixed_params_before is not None:
        analysis = find_scenario(analysis, params=fixed_params_before)
        print(
            f"Analysis was apriori narrowed down to {len(analysis)} due to fixed params"
        )
    sorted_analysis = analysis.loc[best_configs.index.intersection(analysis.index)] # sorts the analysis DF

    if fixed_params_after is not None:
        sorted_analysis = find_scenario(sorted_analysis, params=fixed_params_after)
        print(
            f"Analysis was aposteriori narrowed down to {len(sorted_analysis)} due to fixed params"
        )

    top_n_results = sorted_analysis.iloc[:top_n]
    average_aggregated = top_n_results["aggregated"].sum() / max(len(
        top_n_results["aggregated"]
    ), 1)

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

    to_be_averaged = ['dist_num', 'threshold']

    value_counts = {}
    param_avg = {}
    for param in important_params:
        print(top_n_results["config/" + param].astype(str).value_counts())
        value_counts["config/" + param] = top_n_results["config/" + param].astype(str).value_counts()
        if param in to_be_averaged:
            param_avg[param] = top_n_results["config/" + param].mean()
    # for key in value_counts.keys():
    #     value_counts[key] = eval(value_counts[key])

    return {'agg_avg':average_aggregated, 'value_counts':value_counts, 'param_avg':param_avg}