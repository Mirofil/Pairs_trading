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

from pairs.analysis import (aggregate, descriptive_frame, descriptive_stats,
                            drawdown, infer_periods)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import standardize_results, beautify
from pairs.helpers import *


def produce_trading_ts(
    rxx:pd.DataFrame, relevant_timeframes:List, take_every_nth:int=1, dxx: pd.DataFrame=None, keep_ts_continuity:bool=True, desired_mult_coef = None
):
    """Generates a trading timeseries for graphing

    #TODO the multiplicative coef is not great. Smoothing it too much is dangerous anyhow, Ill just drop it for now

    Args:
        rxx (): DF with all backtests in first level of multiindex
        relevant_timeframes ([type]): Trading timeframes for each backtest to consider as part of the final trading ts
        take_every_nth (int, optional): This needs to be setup properly with how the time series are overlapping. Defaults to 1.
        keep_ts_continuity (bool, optional): Whether to keep a running multiplier of the achieved profits. It is necessary to make the time series appear coherent as in reality it is stitched through many backtests which are disjoint. Defaults to True.
        desired_mult_coef ([type], optional): This needs to be found out from the returns Table - it should be the typical average value at the end of trading (ie. the monthly average profit to the power of how many months are there in the whole trading backtest timeframe, for example). Defaults to None.


    Returns:
        [type]: [description]
    """
    rxx_trading_ts = pd.concat(
        [
            rxx.loc[
                backtest_idx,
                :,
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                    0
                ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1],
                :,
            ].query('Signals != "Formation"')
            .dropna(subset=["cumProfit"])
            .groupby(level=2)
            .mean()
            .fillna(0)
            for backtest_idx in rxx.index.get_level_values(0).unique(0)
        ]
    )

    rxx_trading_ts = rxx_trading_ts.groupby(rxx_trading_ts.index).last()
    if keep_ts_continuity is True:

        if dxx is None:
            dxx = descriptive_frame(rxx)
        
        per_period_avg_profit = dxx.groupby(level=0).mean().mean()['Total profit']

        multiplicative_factors_levels=[
            rxx.loc[
                backtest_idx,
                :,
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                    0
                ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1],
                :,
            ].dropna(subset=["cumProfit"]).groupby(level=1).last().mean()["cumProfit"]
            for backtest_idx in rxx.index.get_level_values(0).unique(0)
        ]
        multiplicative_factors = pd.Series(multiplicative_factors_levels).cumprod()

        if desired_mult_coef is not None:

            multiplicative_factors_corrections = pd.Series([desired_mult_coef ** ((len(rxx.index.get_level_values(0).unique(0))-i)/len(rxx.index.get_level_values(0).unique(0))) for i in range(len(rxx.index.get_level_values(0).unique(0)))])
            multiplicative_factors_corrections = multiplicative_factors_corrections.iloc[:-1]
            multiplicative_factors_corrections = multiplicative_factors_corrections.append(pd.Series([1], index=[len(multiplicative_factors_corrections)]))
            
            multiplicative_factors = multiplicative_factors * (multiplicative_factors_corrections/multiplicative_factors_levels)

        for backtest_idx in rxx.index.get_level_values(0).unique(0)[1:]:
            len_of_linspace = len(
                rxx_trading_ts.loc[
                    relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                        0
                    ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
                    + relativedelta(days=-1),
                    "cumProfit",
                ]
            )
            rxx_trading_ts.loc[
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][0]
                + relativedelta(days=1) : relevant_timeframes[
                    int(int(backtest_idx) / take_every_nth)
                ][1]
                + relativedelta(days=0),
                "cumProfit",
            ] *= multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)]

            rxx_trading_ts.loc[
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][0]
                + relativedelta(days=1) : relevant_timeframes[
                    int(int(backtest_idx) / take_every_nth)
                ][1]
                + relativedelta(days=0),
                "multFactor",
            ] = multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)]

    return rxx_trading_ts


def preprocess_rdx(rdx, take_every_nth=1, should_ffill=False):
    """Ffill is important when constructing the trading TS as there is an average across days; if you dont ffill, the cumulative profit of a pair that converges before the period ends will have Nones in the later rows """
    rdx["Profit"] = rdx["Profit"].astype(np.float64)
    rdx["cumProfit"] = rdx["cumProfit"].astype(np.float64)
    # NOTE I THINK THIS PART SHOULD ONLY BE NEEDED IN THE GRAPH VERSION? ESP SINCE IT WASNT INCLUDED IN THE ORIGINAL GRAPHS
    if should_ffill:
        for backtest_idx in rdx.index.get_level_values(0).unique():
            for pair in rdx.loc[backtest_idx].index.get_level_values(0).unique():
                rdx.loc[(backtest_idx, pair), "cumProfit"] = (
                    rdx.loc[(backtest_idx, pair), "cumProfit"]
                    .fillna(method="ffill")
                    .values
                )
    rdx = rdx.loc[[i for i in rdx.index.levels[0] if i % take_every_nth == 0]]
    return rdx

def resample_multiindexed_backtests(backtests):
    result = []
    for backtest_idx in tqdm(backtests.index.get_level_values(0).unique()):
        interim = []
        for pair in backtests.loc[backtest_idx].index.get_level_values(0).unique():
            # backtests.loc[(backtest_idx, pair), :] = backtests.loc[(backtest_idx, pair), :].resample('1D').last()
            resampled = backtests.loc[(backtest_idx, pair), :].resample('1D').last()
            resampled.index = pd.MultiIndex.from_product([[pair], resampled.index])
            interim.append(resampled)
        result.append(pd.concat(interim))
    # return backtests
    return pd.concat(result, keys=range(len(result)))

def generate_timeframes(rdx:pd.DataFrame, jump_delta=relativedelta(months=2, days=0, hours=0)):
    """Should generate a non-overlapping list of trading timeframes for consecutive backtests so that we can use it for graphing
        It can also be overlapping to be averaged out later, but that is an unrealistic scenario

    Args:
        rdx (pd.DataFrame): DF with all backtests as first level of multiindex
        jump_delta ([type], optional): Should be the length of trading period for each backtest so that we always jump just by that length and we end up getting non-overlapping, consecutive trading timeframes with full coverage of history. Defaults to relativedelta(months=2, days=0, hours=0).

    Returns:
        [type]: [description]
    """

    relevant_timeframes = [
        (
            infer_periods(rdx.loc[backtest_idx])["trading"][0],
            infer_periods(rdx.loc[backtest_idx])["trading"][0] + jump_delta,
        )
        for backtest_idx in rdx.index.get_level_values(0).unique(0)
    ]
    return relevant_timeframes


def generate_stats_from_ts(pair_ts, market_ts=None, freq="daily", riskfree=0.02):
    num_of_trading_periods = len(pair_ts)
    total_profit = pair_ts.iloc[-1]["cumProfit"] - 1
    if freq == "daily":
        num_of_months = num_of_trading_periods / 30
        trading_days_per_backtest = 60
    monthly_return = pair_ts.iloc[-1]["cumProfit"] ** (1 / num_of_months) - 1
    annual_return = pair_ts.iloc[-1]["cumProfit"] ** (12 / num_of_months) - 1
    sd = pair_ts["cumProfit"].std()
    if freq == "daily":
        # sharpe = ((monthly_return-riskfree/12)/sd)*(360/30)**1/2
        sharpe = (total_profit - riskfree * num_of_months / 12) / sd
    if market_ts is not None:
        corr = pair_ts["cumProfit"].corr(other=market_ts["Close"])

    all_stats = {
        "Monthly profit": monthly_return,
        "Annual profit": annual_return,
        "Total profit": total_profit,
        "Standard deviation": sd,
        "Annualized Sharpe": sharpe,
    }

    if market_ts is not None:
        all_stats["Corr. to market"] = corr
    else:
        all_stats["Corr. to market"] = np.nan
    return all_stats


def prepare_random_scenarios(rxrs, should_ffill=False, workers=1):
    rxrs = [preprocess_rdx
        (rxr, take_every_nth=1, should_ffill=should_ffill) for rxr in rxrs
    ]
    rxrs = Parallel(n_jobs=workers, max_nbytes=None)(delayed(descriptive_frame)(rdr) for rdr in tqdm(rxrs))
    return rxrs


def load_random_scenarios(results_dir, prefix="scenario_randomd"):
    paths = glob.glob(os.path.join(results_dir, prefix + "*"))
    results = []
    for path in paths:
        if "parameters" not in path:
            results.append(
                load_results(
                    os.path.basename(os.path.normpath(path)), "random", results_dir
                )
            )
    return results

def btc_stats(btc: pd.DataFrame, feasible, riskfree=0.02):
    """
    Args:
        btc ([type]): BTC time series with cumProfit as cumulative profit. Should be DAILY frequency!
    """
    profit = btc.iloc[-1]["cumProfit"] - btc.iloc[0]["cumProfit"]
    num_of_trading_days = (btc.index[-1] - btc.index[0]).days
    num_of_trading_months = num_of_trading_days / 30
    monthly_profit = (1 + profit) ** (1 / num_of_trading_months)-1
    max_drawdown = abs(drawdown(btc).min())
    annualized_sd = profit ** ((num_of_trading_days / 360) ** 1 / 2)
    annualized_sharpe = (
        (1 + profit) ** (1 / (num_of_trading_days / 360)) - 1 - riskfree
    ) / annualized_sd

    result= pd.DataFrame(
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
    result.columns = pd.MultiIndex.from_product([['Market'], ['BTC']])
    result = beautify(result)
    result = result.drop(["Annual profit", "Trading period profit", "Trading period Sharpe"])
    result = result.rename({"Number of trades":"Monthly number of trades", "Avg length of position":"Length of position (days)"})
    result = result.replace('nan', 'None').replace('nan\%', 'None')

    return result
