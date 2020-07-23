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
from pairs.config import paper1_univ, paper2_univ
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
from pairs.scripts.paper2.tables import all_periods_summary, lag_txcost_summary, subperiods_summary
from pairs.analysis import find_scenario
from pairs.ray_analysis import compute_aggregated_and_find_best_config, analyse_top_n


# /mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/paper2/analysis
# analysis = ray.tune.Analysis(
#     experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/"
# ).dataframe()

subperiods = [all_history,nineties,dotcom, inbetween_crises, financial_crisis, modern]
new_txcosts = [0,0.0015,0.005, 0.01, 0.02]

subperiods = [nineties,dotcom, inbetween_crises, financial_crisis, modern]
new_txcosts = [0]
all_lags = [0, 1]

for period in subperiods:   
    exp = process_experiment(subperiod=period,ids=range(143), experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=new_txcosts)

for period in subperiods:
    period.analysis = load_experiment(subperiod=period, ids=range(143), experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=new_txcosts)

period = modern
analysis = load_experiment(subperiod=period, ids=range(143), experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=new_txcosts)

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



all_periods_summary(subperiods = [modern], ids=range(143), new_txcosts = new_txcosts, univ=paper2_univ)
lag_txcost_summary(subperiods=[modern], all_txcosts=[0.003, 0.005], all_lags=[0,1], sort_by='Monthly profit', univ=paper2_univ)
subperiods_summary(subperiods, univ = paper2_univ)

analyse_top_n(
    analysis=analysis,
    best_configs=compute_aggregated_and_find_best_config(analysis, sort_by="Annualized Sharpe"),
    top_n=10,
    fixed_params_before={"lag": 1, "config/txcost":0},
)


nya_stats(period=financial_crisis)
