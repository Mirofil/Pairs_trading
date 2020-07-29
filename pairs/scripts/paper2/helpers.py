import datetime
import glob
import os
import timeit
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from tqdm import tqdm

from pairs.analysis import (
    aggregate,
    descriptive_frame,
    descriptive_stats,
    drawdown,
    infer_periods,
)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import standardize_results, beautify
from pairs.helpers import *


def take_closest(num,collection):
   return min(collection,key=lambda x:abs(x-num))

def find_closest_params(params: pd.Series, possibilities: Dict):
    for param in params.index:
        params.loc[param] = take_closest(params.loc[param], possibilities[param])
    return params
   
def select_rolling_best(
    summary,
    possibilities,
    base_params={
        "dist_num": 20,
        "confidence": 0.05,
        "threshold": 2,
        "pairs_deltas":1
    },
):
    schedule = []
    schedule.append(pd.Series(base_params))
    medians = summary["param_medians"].drop(['freq', 'lag'], errors='ignore').astype(np.float32)
    avgs = summary["param_avgs"].drop(['freq', 'lag'], errors='ignore').astype(np.float32)
    # rolling_mean = medians.rolling(2, axis=1).mean()
    for col in rolling_mean.columns[:-1]:
        schedule.append(find_closest_params(avgs[col], possibilities=possibilities))
    
    schedule = pd.concat(schedule, axis=1)
    return schedule

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


def convert_params_deltas_to_multiplier(params_deltas):
    if params_deltas == {"formation_delta": [20, 0, 0], "training_delta": [10, 0, 0]}:
        mult = 1.66
    elif params_deltas == {"formation_delta": [12, 0, 0], "training_delta": [6, 0, 0]}:
        mult = 1
    elif params_deltas == {"formation_delta": [6, 0, 0], "training_delta": [3, 0, 0]}:
        mult = 0.5
    elif params_deltas == {"formation_delta": [2, 0, 0], "training_delta": [1, 0, 0]}:
        mult = 0.16
    return mult


def convert_multiplier_to_params_deltas(mult):
    return {
        "formation_delta": [12 * mult, 0 * mult, 0 * mult],
        "training_delta": [6 * mult, 0 * mult, 0 * mult],
    }


def join_summaries_by_period(summaries: List, periods: List):
    new_table = []
    new_index = summaries[0].index
    all_cols = []

    for period in periods:
        for summary in summaries:
            all_cols.append(summary.loc[:, [period.table_name]])
    keys = []
    for period in periods:
        for i in range(len(summaries)):
            keys.append(period.table_name)
    all_cols = pd.concat(all_cols, axis=1)

    return all_cols


def ts_stats(ts: pd.DataFrame, feasible=None, riskfree=0.02):
    """
    Args:
        btc ([type]): BTC time series with cumProfit as cumulative profit. Should be DAILY frequency!
    """
    if feasible is None:
        feasible = [
            "Monthly profit",
            "Annual profit",
            "Total profit",
            "Annualized Sharpe",
            "Trading period Sharpe",
            "Number of trades",
            "Roundtrip trades",
            "Avg length of position",
            "Pct of winning trades",
            "Max drawdown",
            "Nominated pairs",
            "Traded pairs",
        ]
    profit = ts.iloc[-1]["cumProfit"] - ts.iloc[0]["cumProfit"]
    num_of_trading_days = (ts.index[-1] - ts.index[0]).days
    # num_of_trading_days = len(ts.index)
    num_of_trading_months = num_of_trading_days / 30
    monthly_profit = (1 + profit) ** (1 / num_of_trading_months) - 1
    max_drawdown = abs(drawdown(ts).min())
    annualized_sd = ts["cumProfit"].std() * (1 / ((num_of_trading_days / 360) ** 1 / 2))
    annualized_sharpe = (
        (1 + profit) ** (1 / (num_of_trading_days / 360)) - 1 - riskfree
    ) / annualized_sd

    result = pd.DataFrame(
        [
            monthly_profit,
            monthly_profit ** 12,
            monthly_profit,
            annualized_sharpe,
            None,
            None,
            None,
            None,
            None,
            max_drawdown,
            None,
            None,
        ],
        index=feasible,
    )
    result.columns = pd.MultiIndex.from_product([["Market"], ["NYA"]])
    result = beautify(result)
    result = result.drop(
        ["Annual profit", "Trading period profit", "Trading period Sharpe"]
    )
    result = result.rename(
        {
            "Number of trades": "Monthly number of trades",
            "Avg length of position": "Length of position (days)",
        }
    )
    result = result.replace("nan", "None").replace("nan\%", "None")

    return result


def nya_stats(
    start_date: str = None,
    end_date: str = None,
    periods=None,
    nya_path: pd.DataFrame = "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/hist/NYA.csv",
):
    results = []
    if periods is not None:
        for period in periods:
            start_date = period.start_date.strftime("%Y-%m-%d")
            end_date = period.end_date.strftime("%Y-%m-%d")
            nya = pd.read_csv(nya_path)
            nya = nya.set_index("Date")
            nya.index = pd.to_datetime(nya.index)
            if type(end_date) is str:
                end_date = pd.to_datetime(end_date)
            if type(start_date) is str:
                start_date = pd.to_datetime(start_date)

            nya = nya.loc[start_date:end_date]
            nya["Close"] = nya["Close"] / nya["Close"].iloc[0]
            nya["cumProfit"] = nya["Close"]
            result = ts_stats(nya)
            result = result.rename({"NYA": "NYSE"}, level=1, axis=1)
            result = result.droplevel(level=1, axis=1)

            results.append(result)
    results = pd.concat(
        results, axis=1, keys=[subperiod.table_name for subperiod in periods]
    )

    return results

