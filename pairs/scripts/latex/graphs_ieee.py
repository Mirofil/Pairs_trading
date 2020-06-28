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
def load_random_scenarios(results_dir):
    paths = glob.glob(os.path.join(results_dir, "scenario_random*"))
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
        multiplicative_factors = (
            ddx.groupby(level=0).mean().iloc[1:].cumprod()["Cumulative profit"].tolist()
        )
        multiplicative_factors.append(multiplicative_factors[-1])
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
                relevant_timeframes[int(int(backtest_idx) / take_every_nth)][
                    0
                ] : relevant_timeframes[int(int(backtest_idx) / take_every_nth)][1]
                + relativedelta(days=-1),
                "cumProfit",
            ] *= np.linspace(multiplicative_factors[int(int(backtest_idx) / take_every_nth - 1)], multiplicative_factors[int(int(backtest_idx) / take_every_nth)], num = len_of_linspace)

    #    rdx_trading_ts["cumProfit"] = rdx_trading_ts["Profit"] + 1
    #    rdx_trading_ts["cumProfit"] = rdx_trading_ts["cumProfit"].cumprod()
    return rdx_trading_ts


def preprocess_rdx(rdx, take_every_nth=1):
    rdx["Profit"] = rdx["Profit"].astype(np.float64)
    rdx["cumProfit"] = rdx["cumProfit"].astype(np.float64)
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


# for backtest_idx in rdc.index.get_level_values(0).unique(0):
#     for pair in rdc.loc[backtest_idx].index.get_level_values(0).unique(0):
#         rdc.loc[(backtest_idx, pair), "cumTotalProfit"] = rdc.loc[(backtest_idx,pair),"Profit"].cumsum().values

newbase = paper1_univ.save_path_results
rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rdrs = load_random_scenarios(newbase)

nth = 2

rdd = preprocess_rdx(rdd, take_every_nth=nth)
rdc = preprocess_rdx(rdc, take_every_nth=nth)
rdrs = [preprocess_rdx(rdr, take_every_nth=nth) for rdr in rdrs]

ddd = descriptive_frame(rdd)
ddc = descriptive_frame(rdc)

relevant_timeframes = generate_timeframes(
    rdd, jump_delta=relativedelta(months=2, days=0, hours=0)
)


rdd_trading_ts = produce_trading_ts(
    rdd, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)

rdc_trading_ts = produce_trading_ts(
    rdc, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)
# rdrs = [produce_trading_ts(rdr, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True) for rdr in tqdm(rdrs)]
rdr_trading_ts = pd.concat(rdrs).groupby(level=0).mean()

# rdd_trading_ts = rdd_trading_ts.groupby(level=0).mean()
# rdc_trading_ts = rdc_trading_ts.groupby(level=0).mean()
# feasible = [
#     "Monthly profit",
#     "Annual profit",
#     "Total profit",
#     "Annualized Sharpe",
#     "Trading period Sharpe",
#     "Number of trades",
#     "Roundtrip trades",
#     "Avg length of position",
#     "Pct of winning trades",
#     "Max drawdown",
#     "Nominated pairs",
#     "Traded pairs",
# ]
# agg = aggregate(
#     [ddd, ddc],
#     columns_to_pick=feasible,
#     multiindex_from_product_cols=[["Daily"], ["Dist.", "Coint."]],
#     trades_nonzero=True,
#     returns_nonzero=True,
#     trading_period_days=[60, 60],
# )
# agg = standardize_results(agg)
# agg = beautify(agg)
# latexsave(agg, save_path_tables + "resultstable")


plt.style.use("default")
fig, ax = plt.subplots(1, 1)
btcusd = pd.read_csv(
    "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/BTCUSDT.csv"
)
btcusd = pd.DataFrame(btcusd.set_index("Opened")["Close"], columns=["Close"])
btcusd.index = pd.to_datetime(btcusd.index)
btcusd = btcusd.loc[
    relevant_timeframes[0][0] : relevant_timeframes[-1][1]
]  # need to match up with the trading periods from our strategy
btcusd["Close"] = btcusd["Close"] / btcusd["Close"].iloc[0]
btcusd["Close"].plot(linewidth=1, color="k", ax=ax)

rdd_trading_ts["cumProfit"].plot(linewidth=0.5, color="b", ax=ax)
rdc_trading_ts["cumProfit"].plot(linewidth=0.5, color="k", ax=ax)
# rdr_trading_ts["cumProfit"].plot(linewidth=0.5, color="r", ax=ax)

plt.xlabel("Date")
plt.ylabel("BTC/USDT")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(paper1_univ.save_path_graphs, "btcprice.png"))

# %%
