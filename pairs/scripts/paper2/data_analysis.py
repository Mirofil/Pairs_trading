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
from pairs.ray_analysis import compute_aggregated_and_find_best_config, analyse_top_n


# /mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/paper2/analysis
# analysis = ray.tune.Analysis(
#     experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/"
# ).dataframe()
all_subperiods = [nineties,dotcom, inbetween_crises, financial_crisis, modern]

for period in all_subperiods:   
    exp = process_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=[0,0.005, 0.01, 0.02])


period = dotcom
analysis = load_experiment(subperiod=period, experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/", new_txcosts=[0,0.005, 0.01, 0.02])
# sort_aggs_by_stat(best_configs["aggregated"], "Monthly profit")
analyse_top_n(
    analysis,
    compute_aggregated_and_find_best_config(analysis, sort_by="Annualized Sharpe"),
    top_n=20,
    fixed_params_before={"lag": 1, "config/txcost":0.005},
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


# sort_aggs_by_stat(best_configs["aggregated"], "Monthly profit")
analyse_top_n(
    analysis=analysis,
    best_configs=compute_aggregated_and_find_best_config(analysis, sort_by="Annualized Sharpe"),
    top_n=10,
    fixed_params_before={"lag": 1, "config/txcost":0},
)


nya_stats(start_date='1990/1/1', end_date='2000/1/1')
