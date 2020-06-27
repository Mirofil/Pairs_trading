#%%
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

# GRAPH OF BTC PRICE AND COMPARISON TO BUY AND HOLD
#%%
# GIANT RESULT TABLE
newbase = paper1_univ.save_path_results
rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rdd["Profit"] = rdd["Profit"].astype(np.float64)
rdc["Profit"] = rdc["Profit"].astype(np.float64)

rdc=rdc.loc[[i for i in rdc.index.levels[0] if i%2 ==0]]
rdd=rdd.loc[[i for i in rdd.index.levels[0] if i%2 ==0]]


ddd = descriptive_frame(rdd)
ddc = descriptive_frame(rdc)
# ddd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddd.pkl"))
# ddc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddc.pkl"))

jump_delta = relativedelta(
    months=2, days=0, hours=0
)  # the daily baseline scenarios have a jump_delta of 1 mth, but we need to have 2mth to make a continuous time series
earliest_trading_period = infer_periods(rdd.loc[0])["trading"]
relevant_timeframes = [
    (
        infer_periods(rdd.loc[backtest_idx])["trading"][0],
        infer_periods(rdd.loc[backtest_idx])["trading"][0] + jump_delta,
    )
    for backtest_idx in rdd.index.get_level_values(0).unique(0)
]
# relevant_timeframes[-1] = (
#     relevant_timeframes[-1][0],
#     relevant_timeframes[-1][1] + jump_delta,
# )

rdd_trading_ts = pd.concat(
    [
        rdd.loc[
            backtest_idx,
            :,
            relevant_timeframes[int(int(backtest_idx)/2)][0] : relevant_timeframes[int(int(backtest_idx)/2)][1],
            :,
        ]
        .groupby(level=2)
        .mean()
        .fillna(0)
        for backtest_idx in rdd.index.get_level_values(0).unique(0)
    ]
)
rdd_trading_ts["cumProfit"] = rdd_trading_ts["Profit"] + 1
rdd_trading_ts["cumProfit"] = rdd_trading_ts["cumProfit"].cumprod()

rdc_trading_ts = pd.concat(
    [
        rdc.loc[
            backtest_idx,
            :,
            relevant_timeframes[int(int(backtest_idx)/2)][0] : relevant_timeframes[int(int(backtest_idx)/2)][1],
            :,
        ]
        .groupby(level=2)
        .mean()
        .fillna(0)
        for backtest_idx in rdc.index.get_level_values(0).unique(0)
    ]
)

rdc_trading_ts["cumProfit"] = rdc_trading_ts["Profit"] + 1
rdc_trading_ts["cumProfit"] = rdc_trading_ts["cumProfit"].cumprod()

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
    "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWconcatenated_price_data/BTCUSDT.csv"
)
btcusd = pd.DataFrame(btcusd.set_index("Opened")["Close"], columns=["Close"])
btcusd.index = pd.to_datetime(btcusd.index)
btcusd = btcusd.loc[
    relevant_timeframes[0][0] : relevant_timeframes[-1][1]
]  # need to match up with the trading periods from our strategy
btcusd["Close"] = btcusd["Close"] / btcusd["Close"].iloc[0]
btcusd["Close"].plot(linewidth=0.5, color="k", ax=ax)

rdd_trading_ts["cumProfit"].plot(linewidth=2, color="k", ax=ax)
rdc_trading_ts["cumProfit"].plot(linewidth=1, color="k", ax=ax)
plt.xlabel("Date")
plt.ylabel("BTC/USDT")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(paper1_univ.save_path_graphs, "btcprice.png"))

# %%
