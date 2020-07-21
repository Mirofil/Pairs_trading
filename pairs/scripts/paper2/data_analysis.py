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
from pairs.scripts.paper2.subperiods import nineties, dotcom, financial_crisis, inbetween_crises, modern, all_history
from pairs.analysis import find_scenario
from pairs.ray_analysis import compute_aggregated_and_find_best_config, analyse_top_n


# /mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/paper2/analysis
# analysis = ray.tune.Analysis(
#     experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/"
# ).dataframe()

subperiods = [nineties,dotcom, inbetween_crises, financial_crisis, modern, all_history]
new_txcosts = [0,0.005, 0.01, 0.02]
all_lags = [0, 1]

for period in subperiods:   
    process_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=new_txcosts)

for period in subperiods:
    period.analysis = load_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=new_txcosts)

period = modern
analysis = load_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=new_txcosts)

# sort_aggs_by_stat(best_configs["aggregated"], "Monthly profit")
anal = analyse_top_n(
    analysis,
    compute_aggregated_and_find_best_config(analysis, sort_by="Monthly profit"),
    top_n=10,
    fixed_params_before={"lag": 1, "config/txcost":0.003, "dist_num":20},
)


find_scenario(
    analysis,
    {
        "lag": 1,
        "txcost": 0.003,
        "dist_num": 20,
        "pairs_deltas/formation_delta": "[6, 0, 0]",
    },
)
stats_df = compute_aggregated_and_find_best_config(analysis=analysis, sort_by='Monthly profit')

aggregate(
    [results[78]], None, [60], [["Daily"], ["Dist"]]
)  # this is benchmark literature conf



def all_periods_summary(subperiods, new_txcosts, top_n=10, fixed_params_before={"lag": 1, "config/txcost":0.003}, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", sort_by='Annualized Sharpe'):

    aggregateds = []
    param_avgs = []
    for period in subperiods:
        analysis = load_experiment(subperiod=period, experiment_dir=experiment_dir, new_txcosts=new_txcosts)
        
        top_n_analysis = analyse_top_n(
            analysis=analysis,
            best_configs=compute_aggregated_and_find_best_config(analysis, sort_by=sort_by),
            top_n=top_n,
            fixed_params_before=fixed_params_before,
        )
        aggregateds.append(top_n_analysis['agg_avg'])
        param_avgs.append(pd.Series(top_n_analysis["param_avgs"]))

    aggregateds = pd.concat(aggregateds, axis=1, keys=[subperiod.name for subperiod in subperiods])
    param_avgs = pd.concat(param_avgs, axis=1, keys=[subperiod.name for subperiod in subperiods])

    return {"aggregateds":aggregateds, "param_avgs":param_avgs}

def lag_txcost_summary(subperiods, all_txcosts, all_lags, top_n=10, fixed_params_before={"lag": 1, "config/txcost":0.003}, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", sort_by='Annualized Sharpe'):

    table = pd.DataFrame([], index = all_lags, columns = all_txcosts)

    aggregateds = []
    param_avgs = []
    for period in subperiods:
        analysis = load_experiment(subperiod=period, experiment_dir=experiment_dir, new_txcosts=new_txcosts)
        for lag in all_lags:
            for txcost in all_txcosts:
                top_n_analysis = analyse_top_n(
                    analysis=analysis,
                    best_configs=compute_aggregated_and_find_best_config(analysis, sort_by=sort_by),
                    top_n=top_n,
                    fixed_params_before={"lag":lag, "txcost":txcost},
                )
                table.loc[lag, txcost] = [pd.Series(top_n_analysis["param_avg"])]
    table = table.applymap(lambda x: x[0])
    rows = []
    for lag in table.index:
        interim = []
        for txcost in table.columns:
            interim.append(table.loc[lag, txcost])
        interim = pd.concat(interim, axis=1, keys = table.columns)
        rows.append(interim)
    new_table = pd.concat(rows, keys=table.index)

    return table

lag_txcost_summary([modern], all_txcosts=[0.003, 0.005], all_lags=[0,1])

analyse_top_n(
    analysis=analysis,
    best_configs=compute_aggregated_and_find_best_config(analysis, sort_by="Annualized Sharpe"),
    top_n=10,
    fixed_params_before={"lag": 0, "config/txcost":0},
)


nya_stats(start_date='1990/1/1', end_date='2000/1/1')
