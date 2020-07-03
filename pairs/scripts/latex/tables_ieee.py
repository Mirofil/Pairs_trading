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
from pairs.scripts.latex.helpers import resample_multiindexed_backtests



# BASELINE BENCHMARK RESULTS TABLE
newbase = paper1_univ.save_path_results
rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rhd = load_results("scenario3", "dist", newbase)
rhc = load_results("scenario3", "coint", newbase)
rtd = load_results("scenario7_full_coverage", "dist", newbase)
rtc = load_results("scenario7_full_coverage", "coint", newbase)



rdrs = load_random_scenarios(newbase, prefix='scenario_randomd')
rhrs = load_random_scenarios(newbase, prefix='scenario_randomh')
rtrs = load_random_scenarios(newbase, prefix='scenario_randomt')



#NOTE TAKE CARE IF YOU WANT TO LOAD PREDEFINED rdrs, .. Here we should take every nth to form the timeseries but in the big results table, we want every backtest
rdd = preprocess_rdx(rdd, take_every_nth=2, should_ffill=False)
rdc = preprocess_rdx(rdc, take_every_nth=2,should_ffill=False)
rhd = preprocess_rdx(rhd, take_every_nth=1,should_ffill=False)
rhc = preprocess_rdx(rhc, take_every_nth=1,should_ffill=False)
rtd = preprocess_rdx(rtd, take_every_nth=1,should_ffill=False)
rtc = preprocess_rdx(rtc, take_every_nth=1,should_ffill=False)

rhd=resample_multiindexed_backtests(rhd)
rhc = resample_multiindexed_backtests(rhc)
rtd = resample_multiindexed_backtests(rtd)
rtc = resample_multiindexed_backtests(rtc)


rdrs = [preprocess_rdx(rdr, take_every_nth=2,should_ffill=False) for rdr in tqdm(rdrs)]
rhrs = [preprocess_rdx(rhr, take_every_nth=1,should_ffill=False) for rhr in tqdm(rhrs)]
rtrs = [preprocess_rdx(rtr, take_every_nth=1,should_ffill=False) for rtr in tqdm(rtrs)]

ddd = descriptive_frame(rdd)
ddc = descriptive_frame(rdc)
dhd = descriptive_frame(rhd)
dhc = descriptive_frame(rhc)
dtd = descriptive_frame(rtd)
dtc = descriptive_frame(rtc)

#NOTE CAREFUL WITH THIS LOADING! THE NTH PARAMETER MIGHT BE OUT OF SYNC
# ddd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddd.pkl"))
# dhd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dhd.pkl"))
# dhc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dhc.pkl"))
# ddc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddc.pkl"))
# dtd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtd.pkl"))
# dtc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtc.pkl"))

# rdrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rdrs.pkl"))
# rhrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rhrs.pkl"))
# rtrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rtrs.pkl"))


relevant_timeframes_d = generate_timeframes(
    rdd, jump_delta=relativedelta(months=2, days=0, hours=0)
)

relevant_timeframes_h = generate_timeframes(rhd, jump_delta=relativedelta(months=0, days=10, hours=0))
relevant_timeframes_t = generate_timeframes(rtd, jump_delta=relativedelta(months=0, days=3, hours=0))

rdd_trading_ts = produce_trading_ts(
    rdd, relevant_timeframes_d, take_every_nth=2, keep_ts_continuity=True
)

rdc_trading_ts = produce_trading_ts(
    rdc, relevant_timeframes_d, take_every_nth=2, keep_ts_continuity=True
)
rhd_trading_ts = produce_trading_ts(
    rhd, relevant_timeframes_h, take_every_nth=1, keep_ts_continuity=True
)
rhc_trading_ts = produce_trading_ts(
    rhc, relevant_timeframes_h, take_every_nth=1, keep_ts_continuity=True
)
rtd_trading_ts = produce_trading_ts(
    rtd, relevant_timeframes_t, take_every_nth=1, keep_ts_continuity=True
)
rtc_trading_ts = produce_trading_ts(
    rtc, relevant_timeframes_t, take_every_nth=1, keep_ts_continuity=True
)


rdrs = [
    produce_trading_ts(
        rdr, relevant_timeframes_d, take_every_nth=2, keep_ts_continuity=True
    )
    for rdr in tqdm(rdrs)
]
rhrs = [
    produce_trading_ts(
        rhr, relevant_timeframes_h, take_every_nth=1, keep_ts_continuity=True
    )
    for rhr in tqdm(rhrs)
]
rtrs = [
    produce_trading_ts(
        rtr, relevant_timeframes_t, take_every_nth=1, keep_ts_continuity=True
    )
    for rtr in tqdm(rtrs)
]



rdr_trading_ts = pd.concat(rdrs).groupby(level=0).mean()
rhr_trading_ts = pd.concat(rhrs).groupby(level=0).mean()
rtr_trading_ts = pd.concat(rtrs).groupby(level=0).mean()

with open(os.path.join(paper1_univ.save_path_tables, "rdr_trading_ts.pkl"), "wb") as f:
    pickle.dump(rdr_trading_ts, f)

with open(os.path.join(paper1_univ.save_path_tables, "rhr_trading_ts.pkl"), "wb") as f:
    pickle.dump(rhr_trading_ts, f)

with open(os.path.join(paper1_univ.save_path_tables, "rtr_trading_ts.pkl"), "wb") as f:
    pickle.dump(rtr_trading_ts, f)

rdr_trading_ts = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rdr_trading_ts.pkl"))
rhr_trading_ts = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rhr_trading_ts.pkl"))
rtr_trading_ts = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rtr_trading_ts.pkl"))



btcusd = pd.DataFrame(btcusd.set_index("Opened")["Close"], columns=["Close"])
btcusd.index = pd.to_datetime(btcusd.index)
btcusd = btcusd.resample("1D").last()
btcusd = btcusd.loc[
    relevant_timeframes_d[0][0] : relevant_timeframes_d[-1][1]
]  # need to match up with the trading periods from our strategy
btcusd["Close"] = btcusd["Close"] / btcusd["Close"].iloc[0]
btcusd["cumProfit"] = btcusd["Close"]
btcusd["Buy&Hold (BTC)"] = btcusd["Close"]

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


# BIG RESULTS TABLE WITH RANDOM RESULTS AND CORRELATIONS
newbase = paper1_univ.save_path_results
tables_save = paper1_univ

# rdd = load_results("scenario1", "dist", newbase)
# rdc = load_results("scenario1", "coint", newbase)

# rhd = load_results("scenario3", "dist", newbase)
# rhc = load_results("scenario3", "coint", newbase)

# rtd = load_results("scenario7", "dist", newbase)
# rtc = load_results("scenario7", "coint", newbase)


# rdd = preprocess_rdx(rdd, take_every_nth=1)
# rdc = preprocess_rdx(rdc, take_every_nth=1)
# rhd = preprocess_rdx(rhd, take_every_nth=1)
# rhc = preprocess_rdx(rhc, take_every_nth=1)
# rtd = preprocess_rdx(rtd, take_every_nth=1)
# rtc = preprocess_rdx(rtc, take_every_nth=1)

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

ddd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddd.pkl"))
dhd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dhd.pkl"))
dhc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables,"dhc.pkl"))
ddc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddc.pkl"))
dtd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtd.pkl"))
dtc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtc.pkl"))

rdrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddrs.pkl"))
rhrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dhrs.pkl"))
rtrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtrs.pkl"))


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
        trading_period_days=[60]
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
        trading_period_days=[10],
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
        trading_period_days=[3]
    )
    for rtr in rtrs
]

daily_aggs = sum(daily_aggs) / len(daily_aggs)
hourly_aggs = sum(hourly_aggs) / len(hourly_aggs)
minute_aggs = sum(minute_aggs) / len(minute_aggs)

agg_rand = pd.concat([daily_aggs, hourly_aggs, minute_aggs], axis=1)

agg_rand = standardize_results(agg_rand, poslen=[1, 1 / 24, 1 / 288], numtrades=[1 / 2, 3, 10])
agg_rand = beautify(agg_rand)

agg = aggregate(
    [ddd, ddc, dhd, dhc, dtd, dtc],
    columns_to_pick=feasible,
    trades_nonzero=True,
    returns_nonzero=True,
    trading_period_days=[60, 60, 10, 10, 3, 3],
    multiindex_from_product_cols=[["Daily", "Hourly", "5-Minute"], ["Dist.", "Coint."]],
)
agg = standardize_results(
    agg,
    poslen=[1, 1, 1 / 24, 1 / 24, 1 / 288, 1 / 288],
    numtrades=[1 / 2, 1 / 2, 3, 3, 10, 10],

)
agg = beautify(agg)

def nth_column(df, n):
    return df[[df.columns[n]]]

total_agg = pd.concat([nth_column(agg, 0), nth_column(agg, 1), nth_column(agg_rand, 0), nth_column(agg,2), nth_column(agg,3), nth_column(agg_rand,1), nth_column(agg,4), nth_column(agg,5), nth_column(agg_rand, 2)], axis=1)

latexsave(total_agg, os.path.join(paper1_univ.save_path_tables, "randomresultstable_full"))

