import datetime
import glob
import os
import pickle
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from pairs.analysis import (aggregate, descriptive_frame, descriptive_stats,
                            infer_periods)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import standardize_results
from pairs.helpers import *
from pairs.scripts.latex.helpers import *
from pairs.scripts.latex.helpers import resample_multiindexed_backtests

# GRAPH OF BTC PRICE AND COMPARISON TO BUY AND HOLD

newbase = paper1_univ.save_path_results
rdd = load_results("scenario1", "dist", newbase)
rdc = load_results("scenario1", "coint", newbase)
rdrs = load_random_scenarios(newbase)

nth = 2

rdd = preprocess_rdx(rdd, take_every_nth=nth, should_ffill=True)
rdc = preprocess_rdx(rdc, take_every_nth=nth, should_ffill=True)
rdrs = [preprocess_rdx(rdr, take_every_nth=nth, should_ffill=True) for rdr in rdrs]

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
rdrs = [produce_trading_ts(rdr, relevant_timeframes, take_every_nth=nth, keep_ts_continuity=True) for rdr in tqdm(rdrs)]
rdr_trading_ts = pd.concat(rdrs).groupby(level=0).mean()


plt.style.use("default")
fig, ax = plt.subplots(1, 1)
btcusd = pd.read_csv(
    "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/BTCUSDT.csv"
)

btcusd = pd.DataFrame(btcusd.set_index("Opened")["Close"], columns=["Close"])
btcusd.index = pd.to_datetime(btcusd.index)
btcusd = btcusd.resample('1D').last()
btcusd = btcusd.loc[
    relevant_timeframes[0][0] : relevant_timeframes[-1][1]
]  # need to match up with the trading periods from our strategy
btcusd["Close"] = btcusd["Close"] / btcusd["Close"].iloc[0]
btcusd['cumProfit'] = btcusd['Close']
btcusd['Buy&Hold (BTC)'] = btcusd['Close']
btcusd["Buy&Hold (BTC)"].plot(linewidth=1, color='k', ax=ax)


rdd_trading_ts["Distance"] = rdd_trading_ts['cumProfit']
rdc_trading_ts["Cointegration"] = rdc_trading_ts['cumProfit']
rdr_trading_ts["Random"] = rdr_trading_ts["cumProfit"]


rdd_trading_ts["Distance"].plot(linewidth=0.5, color='tab:red', ax=ax)
rdc_trading_ts["Cointegration"].plot(linewidth=0.5, color='purple', ax=ax)
rdr_trading_ts["Random"].plot(linewidth=0.5, color="mediumblue", ax=ax)

plt.xlabel("Date")
plt.ylabel("Cumulative profit")
plt.legend()
for item in ax.legend().get_texts():
    item.set_fontsize(11)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(13)
plt.tight_layout()

plt.savefig(os.path.join(paper1_univ.save_path_graphs, "btcprice_big.png"), dpi=300)

# %%
#SUPER GRAPH TODO let me forget this for now, I think ill go with just the basic graph. there are some subtle inconsistencies here
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

rhd = resample_multiindexed_backtests(rhd)
rhc = resample_multiindexed_backtests(rhc)
rtd = resample_multiindexed_backtests(rtd)
rtc = resample_multiindexed_backtests(rtc) #This takes like 50 mins

rhrs = [resample_multiindexed_backtests(rhr) for rhr in tqdm(rhrs)]
rtrs = [resample_multiindexed_backtests(rtr) for rtr in tqdm(rtrs)]

with open(os.path.join(paper1_univ.save_path_tables, 'rhrs_resampled.pkl'), 'wb') as f:
    pickle.dump(rhrs, f)

with open(os.path.join(paper1_univ.save_path_tables, 'rtrs_resampled.pkl'), 'wb') as f:
    pickle.dump(rtrs, f)


#NOTE TAKE CARE IF YOU WANT TO LOAD PREDEFINED rdrs, .. Here we should take every nth to form the timeseries but in the big results table, we want every backtest
rdd = preprocess_rdx(rdd, take_every_nth=2, should_ffill=True)
rdc = preprocess_rdx(rdc, take_every_nth=2,should_ffill=True)
rhd = preprocess_rdx(rhd, take_every_nth=1,should_ffill=True)
rhc = preprocess_rdx(rhc, take_every_nth=1,should_ffill=True)
rtd = preprocess_rdx(rtd, take_every_nth=1,should_ffill=True)
rtc = preprocess_rdx(rtc, take_every_nth=1,should_ffill=True)


rdrs = [preprocess_rdx(rdr, take_every_nth=2,should_ffill=True) for rdr in tqdm(rdrs)]
rhrs = [preprocess_rdx(rhr, take_every_nth=1,should_ffill=True) for rhr in tqdm(rhrs)]
rtrs = [preprocess_rdx(rtr, take_every_nth=1,should_ffill=True) for rtr in tqdm(rtrs)]



# ddd = descriptive_frame(rdd)
# ddc = descriptive_frame(rdc)
# dhd = descriptive_frame(rhd)
# dhc = descriptive_frame(rhc)
# dtd = descriptive_frame(rtd)
# dtc = descriptive_frame(rtc)

ddd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddd.pkl"))
dhd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dhd.pkl"))
dhc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables,"dhc.pkl"))
ddc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddc.pkl"))
dtd = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtd.pkl"))
dtc = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtc.pkl"))

ddrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "ddrs.pkl"))
dhrs = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dhrs.pkl"))
dtrs= pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "dtrs.pkl"))


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
    rhd, relevant_timeframes_h, take_every_nth=1, keep_ts_continuity=True,
)
rhc_trading_ts = produce_trading_ts(
    rhc, relevant_timeframes_h, take_every_nth=1, keep_ts_continuity=True, 
)
rtd_trading_ts = produce_trading_ts(
    rtd, relevant_timeframes_t, take_every_nth=1, keep_ts_continuity=True
)
rtc_trading_ts = produce_trading_ts(
    rtc, relevant_timeframes_t, take_every_nth=1, keep_ts_continuity=True
)


rdrs_ts = [
    produce_trading_ts(
        rdr, relevant_timeframes_d, take_every_nth=2, keep_ts_continuity=True
    )
    for rdr in tqdm(rdrs)
]
rhrs_ts = [
    produce_trading_ts(
        rhr, relevant_timeframes_h, take_every_nth=1, keep_ts_continuity=True
    )
    for rhr in tqdm(rhrs)
]
rtrs_ts = [
    produce_trading_ts(
        rtr, relevant_timeframes_t, take_every_nth=1, keep_ts_continuity=True
    )
    for rtr in tqdm(rtrs)
]



rdr_trading_ts = pd.concat(rdrs_ts).groupby(level=0).mean()
rhr_trading_ts = pd.concat(rhrs_ts).groupby(level=0).mean()
rtr_trading_ts = pd.concat(rtrs_ts).groupby(level=0).mean()

with open(os.path.join(paper1_univ.save_path_tables, "rdr_trading_ts.pkl"), "wb") as f:
    pickle.dump(rdr_trading_ts, f)

with open(os.path.join(paper1_univ.save_path_tables, "rhr_trading_ts.pkl"), "wb") as f:
    pickle.dump(rhr_trading_ts, f)

with open(os.path.join(paper1_univ.save_path_tables, "rtr_trading_ts.pkl"), "wb") as f:
    pickle.dump(rtr_trading_ts, f)

rdr_trading_ts = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rdr_trading_ts.pkl"))
rhr_trading_ts = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rhr_trading_ts.pkl"))
rtr_trading_ts = pd.read_pickle(os.path.join(paper1_univ.save_path_tables, "rtr_trading_ts.pkl"))

btcusd = pd.read_csv(
    "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/BTCUSDT.csv"
)
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
