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
import pickle
from pairs.formatting import standardize_results, beautify
from pairs.helpers import latexsave


def load_random_scenarios(results_dir, prefix="scenario_randomd"):
    paths = glob.glob(os.path.join(results_dir, prefix + "*"))
    results = []
    for path in paths:
        if "parameters" not in path:
            results.append(
                load_results(
                    os.path.basename(os.path.normpath(path)), "random", newbase
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
    if keep_ts_continuity is True:
        ddx = descriptive_frame(rdx)
        # multiplicative_factors = (
        #     ddx.groupby(level=0).mean().iloc[1:].cumprod()["Cumulative profit"].tolist()
        # )
        multiplicative_factors = (
            pd.Series(
                [
                    rdx_trading_ts.loc[
                        relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
                    ]["cumProfit"]
                    for backtest_idx in rdx.index.get_level_values(0).unique(0)[0:-1]
                ]
            )
            .cumprod()
            .to_list()
        )
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
        #   np.linspace(
        #         multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)],
        #         multiplicative_factors[int(int(backtest_idx) / take_every_nth)],
        #         num=len_of_linspace,
        #     )

    return rdx_trading_ts


def preprocess_rdx(rdx, take_every_nth=1, should_ffill=True):
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


def generate_timeframes(rdx, jump_delta=relativedelta(months=2, days=0, hours=0)):
    # We must set up jump_delta depending on how we want to overlap the subsequent backtests. months=2 gives you as much overlap and hopefully as accurate results as possible.
    # If there is overlap, it will get averaged out later
    relevant_timeframes = [
        (
            infer_periods(rdd.loc[backtest_idx])["trading"][0],
            infer_periods(rdd.loc[backtest_idx])["trading"][0] + jump_delta,
        )
        for backtest_idx in rdx.index.get_level_values(0).unique(0)
    ]
    return relevant_timeframes


def prepare_random_scenarios(rxrs, should_ffill=False):
    rxrs = [
        preprocess_rdx(rdr, take_every_nth=1, should_ffill=should_ffill) for rdr in rxrs
    ]
    rxrs = [descriptive_frame(rdr) for rdr in tqdm(rxrs)]
    return rxrs


newbase = paper1_univ.save_path_results
tables_save = paper1_univ

rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rdd = preprocess_rdx(rdd, take_every_nth=1)
rdc = preprocess_rdx(rdc, take_every_nth=1)

rdrs = load_random_scenarios(newbase, prefix="scenario_randomd")
rhrs = load_random_scenarios(newbase, prefix="scenario_randomh")
rtrs = load_random_scenarios(newbase, prefix="scenario_randomt")


rdrs = prepare_random_scenarios(rdrs, should_ffill=False)
rhrs = prepare_random_scenarios(rhrs, should_ffill=False)
rtrs = prepare_random_scenarios(rtrs, should_ffill=False)

with open(os.path.join(paper1_univ.save_path_tables, "rdrs.pkl"), "wb") as f:
    pickle.dump(rdrs, f)

with open(os.path.join(paper1_univ.save_path_tables, "rhrs.pkl"), "wb") as f:
    pickle.dump(rhrs, f)

with open(os.path.join(paper1_univ.save_path_tables, "rtrs.pkl"), "wb") as f:
    pickle.dump(rtrs, f)


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

daily_aggs = [
    aggregate(
        [rdr],
        columns_to_pick=feasible,
        multiindex_from_product_cols=[["Daily"], ["Random"]],
        trades_nonzero=True,
        returns_nonzero=True,
    )
    for rdr in rdrs
]

hourly_aggs = [
    aggregate(
        [rhr],
        columns_to_pick=feasible,
        multiindex_from_product_cols=[["Hourly"], ["Random"]],
        trades_nonzero=True,
        returns_nonzero=True,
    )
    for rhr in rhrs
]

minute_aggs = [
    aggregate(
        [rtr],
        columns_to_pick=feasible,
        multiindex_from_product_cols=[["5-Minute"], ["Random"]],
        trades_nonzero=True,
        returns_nonzero=True,
    )
    for rtr in rtrs
]

daily_aggs = sum(daily_aggs) / len(daily_aggs)
hourly_aggs = sum(hourly_aggs) / len(hourly_aggs)
minute_aggs = sum(minute_aggs) / len(minute_aggs)

agg = pd.concat([daily_aggs, hourly_aggs, minute_aggs], axis=1)

# agg = aggregate(
#     [rdrs[1], rhrs[1], rtrs[1]], columns_to_pick=feasible, multiindex_from_product_cols=[["Daily", "Hourly", "5-Minute"], ["Random"]], trades_nonzero=True, returns_nonzero=True
# )
agg = standardize_results(agg, poslen=[1, 1 / 24, 1 / 288], numtrades=[1 / 2, 3, 10])
agg = beautify(agg)
latexsave(agg, os.path.join(paper1_univ.save_path_tables, "randomresultstable"))

