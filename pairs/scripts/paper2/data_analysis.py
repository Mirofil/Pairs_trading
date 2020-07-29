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
from pairs.pairs_trading_engine import (
    pick_range,
    backtests_up_to_date,
    change_txcost_in_backtests,
    calculate_new_experiments_txcost,
    trim_backtests_with_trading_past_date,
)
from pairs.scripts.paper2.loaders import join_backtests_by_id
from pairs.scripts.paper2.helpers import ts_stats, nya_stats, join_summaries_by_period, take_closest, find_closest_params, select_rolling_best
from pairs.scripts.paper2.loaders import process_experiment, load_experiment
from pairs.scripts.paper2.subperiods import (
    nineties,
    dotcom,
    financial_crisis,
    inbetween_crises,
    modern,
    all_history,
    covid,
)
from pairs.scripts.paper2.tables import (
    all_periods_summary,
    lag_txcost_summary,
    subperiods_summary,
)
from pairs.analysis import find_scenario
from pairs.ray_analysis import compute_aggregated_and_sort_by, analyse_top_n


# /mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/paper2/analysis
# analysis = ray.tune.Analysis(
#     experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/"
# ).dataframe()

subperiods = [nineties, dotcom, inbetween_crises, financial_crisis, modern, covid]
new_txcosts = [0, 0.0026, 0.0035]
all_lags = [0, 1]
dist_nums = [5, 10, 20, 40]
thresholds = [0.5, 1, 1.5, 2, 2.5, 3]
confidences = [0.01,0.05,0.1]
pairs_deltas_mult = [1.66,1,0.5,0.166]
base_params = {
    "lag": 1,
    "config/txcost": 0.003,
    "dist_num": 20,
    "config/pairs_deltas": {"formation_delta": [12, 0, 0], "training_delta": [6, 0, 0]},
    "confidence": 0.05,
    "threshold": 2,
}
available_params = {"new_txcost":new_txcosts+[0.003], "dist_num":dist_nums, "threshold":thresholds, "confidence":confidences, "pairs_deltas":[0.16,0.5,1,1.66]}
dist_dirs = [
    "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
    "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow_covid/",
]

subperiods = [nineties, dotcom, inbetween_crises]

subperiods = [nineties, dotcom, inbetween_crises, financial_crisis, modern]
new_txcosts = [0]
all_lags = [0, 1]
pairs_deltas_2id = {
    '{"training_delta":[3,0,0], "formation_delta":[6,0,0]}': 0.5,
    '{"training_delta":[6,0,0], "formation_delta":[12,0,0]}': 1,
    '{"training_delta":[1,0,0], "formation_delta":[2,0,0]}': 0.16,
    '{"training_delta":[10,0,0], "formation_delta":[20,0,0]}': 1.66,
}

id_2pairs_deltas = {
    0.5: '{"training_delta":[3,0,0], "formation_delta":[6,0,0]}',
    1: '{"training_delta":[6,0,0], "formation_delta":[12,0,0]}',
    0.16: '{"training_delta":[1,0,0], "formation_delta":[2,0,0]}',
    1.66: '{"training_delta":[10,0,0], "formation_delta":[20,0,0]}',
}

for period in subperiods:
    exp = process_experiment(
        subperiod=period, experiment_dir=dist_dirs, new_txcosts=new_txcosts,
    )

for period in subperiods:
    period.analysis = load_experiment(
        subperiod=period, experiment_dir=dist_dirs, new_txcosts=new_txcosts,
    )

def trim_backtests_with_trading_past_date(backtests: pd.DataFrame, end_dates_past: str):
    """This is used for joining additional experiments with some other experiment run. THe experiment-to-be-joined should have end_date in excess of the other's experiments's end_date, and there can be some overla[
        The idea here is to assure that no backtests are counted twice
    """
    if type(end_dates_past) is str:
        end_dates_past = pd.to_datetime(end_dates_past)
    for backtest_idx in backtests.index.get_level_values(level=0).unique(0):
        periods = infer_periods(backtests.loc[backtest_idx])
        if pd.to_datetime(periods["trading"][1]) >= end_dates_past:
            backtests_trimmed.append(
                pick_range(
                    backtests.loc[backtest_idx],
                    start=periods["formation"][0],
                    end=periods["trading"][1],
                )
            )
            backtests_trimmed_idxs.append(backtest_idx)

    return pd.concat(backtests_trimmed, keys=backtests_trimmed_idxs)


anal = analyse_top_n(
    analysis,
    compute_aggregated_and_sort_by(analysis, sort_by="Monthly profit"),
    top_n=5,
    fixed_params_before=base_params,
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
stats_df = compute_aggregated_and_sort_by(analysis=analysis, sort_by="Monthly profit")

aggregate(
    [results[78]], None, [60], [["Daily"], ["Dist"]]
)  # this is benchmark literature conf


lag_txcost_summary(
    subperiods=subperiods,
    new_txcosts=new_txcosts,
    all_lags=[0, 1],
    sort_by="Monthly profit",
    univ=paper2_univ,
)
subperiods_summary(subperiods, univ=paper2_univ)

analyse_top_n(
    analysis=analysis,
    best_configs=compute_aggregated_and_sort_by(analysis, sort_by="Annualized Sharpe"),
    top_n=10,
    fixed_params_before={"lag": 1, "config/txcost": 0.003},
)


summary = all_periods_summary(
    subperiods=subperiods,
    top_n=3,
    new_txcosts=new_txcosts,
    univ=paper2_univ,
    force=False,
    experiment_dir=dist_dirs,
    sort_by="Annualized Sharpe",
)
summary_base = all_periods_summary(
    subperiods=subperiods,
    top_n=3,
    new_txcosts=new_txcosts,
    univ=paper2_univ,
    force=False,
    experiment_dir=dist_dirs,
    fixed_params_before=base_params,
    sort_by="Annualized Sharpe",
)

summary_profit = all_periods_summary(
    subperiods=subperiods,
    top_n=3,
    new_txcosts=new_txcosts,
    univ=paper2_univ,
    force=False,
    experiment_dir=dist_dirs,
    sort_by="Monthly profit",
)
summary_profit_base = all_periods_summary(
    subperiods=subperiods,
    top_n=3,
    new_txcosts=new_txcosts,
    univ=paper2_univ,
    force=False,
    experiment_dir=dist_dirs,
    fixed_params_before=base_params,
    sort_by="Monthly profit",
)


nya = nya_stats(periods=subperiods)
join_summaries_by_period(summaries=[nya, summary["aggregateds"]], periods=subperiods)

optimals = [select_rolling_best(summary, available_params)[col] for col in select_rolling_best(summary, available_params).columns]
analyses=[]
for period, optimal in zip(subperiods, optimals):
    # period.analysis["config/pairs_deltas"] = period.analysis["config/pairs_deltas"].apply(convert_params_deltas_to_multiplier)
    fixed_params_base = {"lag": 1, "config/txcost": 0.003}
    fixed_params = {**fixed_params_base, **optimal.to_dict()}
    fixed_params = {k:round(v, 3) for k,v in fixed_params.items()}
    result = analyse_top_n(
    analysis=period.analysis,
    best_configs=compute_aggregated_and_sort_by(period.analysis, sort_by="Annualized Sharpe"),
    top_n=1,
    fixed_params_before=fixed_params,
    )
    analyses.append(result)

[analysis["agg_avg"] for analysis in analyses]
[analysis["param_avg"] for analysis in analyses]