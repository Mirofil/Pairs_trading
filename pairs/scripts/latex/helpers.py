import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import timeit
import datetime
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import *
from pairs.helpers import *
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.analysis import (
    descriptive_frame,
    descriptive_stats,
    infer_periods,
    aggregate,
)
from pairs.formatting import standardize_results
import glob
from tqdm import tqdm

# GRAPH OF BTC PRICE AND COMPARISON TO BUY AND HOLD
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


def produce_trading_ts(
    rdx, relevant_timeframes, take_every_nth=1, keep_ts_continuity=True
):
    rdx_trading_ts = pd.concat(
        [
            rdx.loc[
                backtest_idx,
                :,
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                    0
                ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1],
                :,
            ]
            .dropna(subset=["cumProfit"])
            .groupby(level=2)
            .mean()
            .fillna(0)
            for backtest_idx in rdx.index.get_level_values(0).unique(0)
        ]
    )

    # multiplicative_factors=[
    #     rdx.loc[
    #         backtest_idx,
    #         :,
    #         relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
    #             0
    #         ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1],
    #         :,
    #     ]
    #     .dropna(subset=["cumProfit"])
    #     .groupby(level=1)
    #     .last()
    #     .mean()["cumProfit"]
    #     for backtest_idx in rdx.index.get_level_values(0).unique(0)
    # ]
    rdx_trading_ts = rdx_trading_ts.groupby(rdx_trading_ts.index).last()
    if keep_ts_continuity is True:
        # ddx = descriptive_frame(rdx)
        # multiplicative_factors = (
        #     ddx.groupby(level=0).mean().iloc[1:].cumprod()["Cumulative profit"].tolist()
        # )

        multiplicative_factors=[
            rdx.loc[
                backtest_idx,
                :,
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                    0
                ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1],
                :,
            ]
            .dropna(subset=["cumProfit"])
            .groupby(level=1)
            .last()
            .mean()["cumProfit"]
            for backtest_idx in rdx.index.get_level_values(0).unique(0)
        ]
        multiplicative_factors = pd.Series(multiplicative_factors).cumprod()
        # multiplicative_factors = (
        #     pd.Series(
        #         [
        #             rdx_trading_ts.loc[
        #                 relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
        #             ]["cumProfit"]
        #             for backtest_idx in rdx.index.get_level_values(0).unique(0)[0:-1]
        #         ]
        #     )
        #     .cumprod()
        #     .to_list()
        # )
        # multiplicative_factors.append(multiplicative_factors[-1])
        # print(multiplicative_factors)
        for backtest_idx in rdx.index.get_level_values(0).unique(0)[1:]:
            len_of_linspace = len(
                rdx_trading_ts.loc[
                    relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                        0
                    ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
                    + relativedelta(days=-1),
                    "cumProfit",
                ]
            )
            rdx_trading_ts.loc[
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][0]
                + relativedelta(days=1) : relevant_timeframes[
                    int(int(backtest_idx) / take_every_nth)
                ][1]
                + relativedelta(days=0),
                "cumProfit",
            ] *= multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)]

            rdx_trading_ts.loc[
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][0]
                + relativedelta(days=1) : relevant_timeframes[
                    int(int(backtest_idx) / take_every_nth)
                ][1]
                + relativedelta(days=0),
                "multFactor",
            ] = multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)]

    return rdx_trading_ts


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

def generate_timeframes(rdx, jump_delta=relativedelta(months=2, days=0, hours=0)):
    # We must set up jump_delta depending on how we want to overlap the subsequent backtests. months=2 gives you as much overlap and hopefully as accurate results as possible.
    # If there is overlap, it will get averaged out later
    relevant_timeframes = [
        (
            infer_periods(rdx.loc[backtest_idx])["trading"][0],
            infer_periods(rdx.loc[backtest_idx])["trading"][0] + jump_delta,
        )
        for backtest_idx in rdx.index.get_level_values(0).unique(0)
    ]
    return relevant_timeframes


def generate_stats_from_ts(pair_ts, market_ts=None, freq="daily"):
    riskfree = 0.02
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


def prepare_random_scenarios(rxrs, should_ffill=False):
    rxrs = [
        preprocess_rdx(rdr, take_every_nth=1, should_ffill=should_ffill) for rdr in rxrs
    ]
    rxrs = [descriptive_frame(rdr) for rdr in tqdm(rxrs)]
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


# def produce_trading_ts(
#     rdx, relevant_timeframes, take_every_nth=1, keep_ts_continuity=True
# ):

#     rdx_trading_ts = pd.concat(
#         [
#             rdx.loc[
#                 backtest_idx,
#                 :,
#                 relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
#                     0
#                 ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1],
#                 :,
#             ]
#             .dropna(subset=["cumProfit"])
#             .groupby(level=2)
#             .mean()
#             .fillna(0)
#             for backtest_idx in rdx.index.get_level_values(0).unique(0)
#         ]
#     )
#     if keep_ts_continuity is True:
#         ddx = descriptive_frame(rdx)

#         multiplicative_factors = (
#             pd.Series(
#                 [
#                     rdx_trading_ts.loc[
#                         relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
#                     ]["cumProfit"]
#                     for backtest_idx in rdx.index.get_level_values(0).unique(0)[0:-1]
#                 ]
#             )
#             .cumprod()
#             .to_list()
#         )
#         # multiplicative_factors.append(multiplicative_factors[-1])
#         # print(multiplicative_factors)
#         for backtest_idx in rdx.index.get_level_values(0).unique(0)[1:]:
#             len_of_linspace = len(
#                 rdx_trading_ts.loc[
#                     relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
#                         0
#                     ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
#                     + relativedelta(days=-1),
#                     "cumProfit",
#                 ]
#             )
#             rdx_trading_ts.loc[
#                 relevant_timeframes[int(int(backtest_idx) / take_every_nth)][0]
#                 + relativedelta(days=1) : relevant_timeframes[
#                     int(int(backtest_idx) / take_every_nth)
#                 ][1]
#                 + relativedelta(days=0),
#                 "cumProfit",
#             ] *= multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)]

#             rdx_trading_ts.loc[
#                 relevant_timeframes[int(int(backtest_idx) / take_every_nth)][0]
#                 + relativedelta(days=1) : relevant_timeframes[
#                     int(int(backtest_idx) / take_every_nth)
#                 ][1]
#                 + relativedelta(days=0),
#                 "multFactor",
#             ] = multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)]

#     return rdx_trading_ts
