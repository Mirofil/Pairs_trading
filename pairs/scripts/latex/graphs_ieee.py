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
from pairs.scripts.latex.helpers import *
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
# rdd_trading_ts = rdd_trading_ts.rename({'cumProfit':'Distance'}, axis=1)
# rdc_trading_ts = rdc_trading_ts.rename({'cumProfit':'Cointegration'}, axis=1)
# rdr_trading_ts = rdr_trading_ts.rename({'cumProfit':'Random'}, axis=1)

rdd_trading_ts["Distance"].plot(linewidth=0.5, color='tab:red', ax=ax)
rdc_trading_ts["Cointegration"].plot(linewidth=0.5, color='purple', ax=ax)
rdr_trading_ts["Random"].plot(linewidth=0.5, color="mediumblue", ax=ax)

plt.xlabel("Date")
plt.ylabel("Cumulative profit")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(paper1_univ.save_path_graphs, "btcprice.png"), dpi=300)

# %%
