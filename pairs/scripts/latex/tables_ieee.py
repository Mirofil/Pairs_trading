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
from pairs.scripts.latex.helpers import *
from pairs.formatting import beautify, standardize_results

# BASELINE BENCHMARK DRESULTS TABLE
newbase = paper1_univ.save_path_results
rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rhd = load_results("scenario3", "dist", newbase)
rhc = load_results("scenario3", "coint", newbase)

rtd = load_results("scenario7", "dist", newbase)
rtc = load_results("scenario7", "coint", newbase)

rdrs = load_random_scenarios(newbase, prefix='scenario_randomd')
rhrs = load_random_scenarios(newbase, prefix='scenario_randomh')
rtrs = load_random_scenarios(newbase, prefix='scenario_randomt')

nth = 2

#NOTE TAKE CARE IF YOU WANT TO LOAD PREDEFINED rdrs, .. Here we should take every nth to form the timeseries but in the big results table, we want every backtest
rdd = preprocess_rdx(rdd, take_every_nth=2, should_ffill=False)
rdc = preprocess_rdx(rdc, take_every_nth=2,should_ffill=False)
rhd = preprocess_rdx(rhd, take_every_nth=1,should_ffill=False)
rhc = preprocess_rdx(rhc, take_every_nth=1,should_ffill=False)
rtd = preprocess_rdx(rtd, take_every_nth=1,should_ffill=False)
rtc = preprocess_rdx(rtc, take_every_nth=1,should_ffill=False)
rdrs = [preprocess_rdx(rdr, take_every_nth=2,should_ffill=False) for rdr in tqdm(rdrs)]
rhrs = [preprocess_rdx(rhr, take_every_nth=1,should_ffill=False) for rhr in tqdm(rhrs)]
rtrs = [preprocess_rdx(rtr, take_every_nth=nth,should_ffill=False) for rtr in tqdm(rtrs)]

ddd = descriptive_frame(rdd)
ddc = descriptive_frame(rdc)
dhd = descriptive_frame(rhd)
dhc = descriptive_frame(rhc)
dtd = descriptive_frame(rtd)
dtc = descriptive_frame(rtc)

relevant_timeframes = generate_timeframes(
    rdd, jump_delta=relativedelta(months=2, days=0, hours=0)
)

relevant_timeframes_h = generate_timeframes(rhd, jump_delta=relativedelta(months=0, days=10, hours=0))
relevant_timeframes_t = generate_timeframes(rtd, jump_delta=relativedelta(months=0, days=6, hours=0))

rdd_trading_ts = produce_trading_ts(
    rdd, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)

rdc_trading_ts = produce_trading_ts(
    rdc, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)
rhd_trading_ts = produce_trading_ts(
    rhd, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)
rhc_trading_ts = produce_trading_ts(
    rhc, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)
rtd_trading_ts = produce_trading_ts(
    rtd, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)
rtc_trading_ts = produce_trading_ts(
    rtc, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
)


rdrs = [
    produce_trading_ts(
        rdr, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True
    )
    for rdr in tqdm(rdrs)
]



rdr_trading_ts = pd.concat(rdrs).groupby(level=0).mean()




btcusd = pd.DataFrame(btcusd.set_index("Opened")["Close"], columns=["Close"])
btcusd.index = pd.to_datetime(btcusd.index)
btcusd = btcusd.resample("1D").last()
btcusd = btcusd.loc[
    relevant_timeframes[0][0] : relevant_timeframes[-1][1]
]  # need to match up with the trading periods from our strategy
btcusd["Close"] = btcusd["Close"] / btcusd["Close"].iloc[0]
btcusd["cumProfit"] = btcusd["Close"]
btcusd["Buy&Hold (BTC)"] = btcusd["Close"]
btcusd["Buy&Hold (BTC)"].plot(linewidth=1, color="k", ax=ax)

benchmark_table = pd.concat(
    [
        pd.Series(generate_stats_from_ts(rdd_trading_ts, btcusd)),
        pd.Series(generate_stats_from_ts(rdc_trading_ts, btcusd)),
        pd.Series(generate_stats_from_ts(rdr_trading_ts, btcusd)),
        pd.Series(generate_stats_from_ts(btcusd)),
    ],
    axis=1,
    keys=["Distance", "Cointegration", "Random", "Market"],
)

benchmark_table = beautify(benchmark_table)
latexsave(
    benchmark_table, os.path.join(paper1_univ.save_path_tables, "benchmarkresultstable")
)


# RANDOM RESULTS TABLE
newbase = paper1_univ.save_path_results
tables_save = paper1_univ

rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)

rhd = load_results("scenario3", "dist", newbase)
rhc = load_results("scenario3", "coint", newbase)

rtd = load_results("scenario7", "dist", newbase)
rtc = load_results("scenario7", "coint", newbase)


rdd = preprocess_rdx(rdd, take_every_nth=1)
rdc = preprocess_rdx(rdc, take_every_nth=1)
rhd = preprocess_rdx(rhd, take_every_nth=1)
rhc = preprocess_rdx(rhc, take_every_nth=1)
rtd = preprocess_rdx(rtd, take_every_nth=1)
rtc = preprocess_rdx(rtc, take_every_nth=1)

#NOTE USE THIS WHEN RERUNNING FROM SCRATCH ONLY

# rdrs = load_random_scenarios(newbase, prefix="scenario_randomd")
# rhrs = load_random_scenarios(newbase, prefix="scenario_randomh")
# rtrs = load_random_scenarios(newbase, prefix="scenario_randomt")


# rdrs = prepare_random_scenarios(rdrs, should_ffill=False)
# rhrs = prepare_random_scenarios(rhrs, should_ffill=False)
# rtrs = prepare_random_scenarios(rtrs, should_ffill=False)

# with open(os.path.join(paper1_univ.save_path_tables, "rdrs.pkl"), "wb") as f:
#     pickle.dump(rdrs, f)

# with open(os.path.join(paper1_univ.save_path_tables, "rhrs.pkl"), "wb") as f:
#     pickle.dump(rhrs, f)

# with open(os.path.join(paper1_univ.save_path_tables, "rtrs.pkl"), "wb") as f:
#     pickle.dump(rtrs, f)

rdrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rdrs.pkl"))
rhrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rhrs.pkl"))
rtrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rtrs.pkl"))


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

agg = standardize_results(agg, poslen=[1, 1 / 24, 1 / 288], numtrades=[1 / 2, 3, 10])
agg = beautify(agg)
latexsave(agg, os.path.join(paper1_univ.save_path_tables, "randomresultstable"))

