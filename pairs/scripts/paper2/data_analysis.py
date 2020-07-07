import datetime
import glob
import os
import pickle
import timeit
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from dateutil.relativedelta import relativedelta
from p_tqdm import p_map
from ray import tune
from tqdm import tqdm

from pairs.analysis import (aggregate, descriptive_frame, descriptive_stats,
                            infer_periods)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import beautify, standardize_results
from pairs.helpers import *
from pairs.helpers import latexsave
from pairs.scripts.latex.helpers import *
from pairs.pairs_trading_engine import pick_range

analysis = ray.tune.Analysis(
    experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/"
).dataframe()


def calculate_timeframes(start_date, i, jump_delta, formation_delta, training_delta):
    formation = (
        start_date + i * jump_delta,
        start_date + formation_delta + i * jump_delta,
    )
    trading = (formation[1], formation[1] + training_delta)
    return {"formation": formation, "trading": trading}


def backtest_up_to_date(
    backtests: pd.DataFrame, min_formation_period_start=None, max_trading_period_end: str = None, print_chosen_periods=False
):
    if type(max_trading_period_end) is str:
        max_trading_period_end = pd.to_datetime(max_trading_period_end)
    if type(min_formation_period_start) is str:
        min_formation_period_start = pd.to_datetime(min_formation_period_start)

    if min_formation_period_start is None:
        min_formation_period_start = backtests.iloc[0]

    backtests_trimmed = []

    last_valid_idx = 0
    last_valid_datetime = None
    for backtest_idx in backtests.index.get_level_values(0).unique(0):
        periods = infer_periods(backtests.loc[backtest_idx])
        if pd.to_datetime(periods["trading"][1]) < max_trading_period_end and pd.to_datetime(periods['formation'][0]) >= min_formation_period_start:
            
            backtests_trimmed.append(pick_range(backtests.loc[backtest_idx], start=min_formation_period_start, end=max_trading_period_end))
            # last_valid_idx = max(backtest_idx - 1, 0)
            # last_valid_datetime = infer_periods(backtests.loc[last_valid_idx])[
            #     "trading"
            # ][1]
            # if print_chosen_periods:
            #     print(infer_periods(backtests.loc[last_valid_idx]))
            # break

    # for backtest_idx in backtests.index.get_level_values(0).unique(0):
    #     backtests_trimmed.append(pick_range(backtests.loc[backtest_idx], start=back))
    
    # backtests_trimmed = pd.concat(backtests_trimmed, keys=range(len(backtests)))

    # assert len(backtests_trimmed.index.get_level_values(0).unique(0)) == len(backtests.index.get_level_values(0).unique(0))

    return { "backtests_trimmed": backtests_trimmed}


def join_results_by_id(analysis: pd.DataFrame, ids: List[int] = None):

    loaded_results = []
    if ids is None:
        ids = analysis.index
    for id in tqdm(ids, desc="Loading backtests by id"):
        # note that aggregated.parquet is acutally more like rhc/backtests kind of thing
        loaded_results.append(
            pd.read_parquet(
                os.path.join(analysis.loc[id, "logdir"], "aggregated.parquet")
            )
        )

    interim = pd.Series(loaded_results, index=ids)
    interim.name = "aggregated"
    return analysis.join(interim)


analysis = join_results_by_id(analysis)
[
    descriptive_frame(backtest)
    for backtest in tqdm(
        analysis["aggregated"].values, desc="Calculating descriptive frames"
    )
]
results = p_map(descriptive_frame, analysis["aggregated"].values, num_cpus=40)
backtest_up_to_date(analysis.loc[0, 'aggregated'],'1999/1/1' , '2000/02/02', True)