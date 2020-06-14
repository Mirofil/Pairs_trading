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
from pairs.config import *


#%%
# DISTANCE EXAMPLE OF PAIR
signal = pd.read_pickle("test_dist.pkl")
model = signal.loc[idx["WAVESxXRP", trading[0] : trading[1]], :]
#%%
# DISTANCE EXAMPLE OF PAIR
plt.style.use("default")
fig, ax = plt.subplots(1, 1)
plt.sca(ax)
ax.grid(False)
ax2 = ax.twinx()
ax2.grid(False)
if model.index.nlevels == 2:
    model.index = model.index.droplevel(0)

horiz1 = pd.Series(index=model.index, data=2)
horiz2 = pd.Series(index=model.index, data=-2)
ax.set_ylabel("Normalized spread")
ax.set_ylim(-3, 3)
ax.tick_params(axis="x", labelsize=11)
line1 = model["normSpread"].plot(ax=ax, label="Normalized spread")
horiz1.plot(ax=ax, color="k", linestyle=":", label="Buy/sell threshold")
horiz2.plot(ax=ax, color="k", linestyle=":", label="")
color = "tab:red"
line2 = (
    model["cumProfit"]
    .fillna(method="ffill")
    .fillna(method="bfill")
    .plot(color=color, ax=ax2, label="Value of investment", linestyle="-.")
)
ax2.set_ylabel("Value of investment")
plt.grid(b=None)
ax.legend(loc="upper left")
ax2.legend()
plt.grid(b=None)
plt.tight_layout()
plt.savefig(save_path_graphs + "pairexample", dpi=300)


#%%
# BINANCE RISE TO FAME
# btcusd = pd.read_csv('C:\Bach\concatenated_price_data\BTCUSDT.CSV')
# btcusd=resample(btcusd, freq='1D')
# btcusd.to_pickle('btcprice.pkl')
btcusd = pd.read_pickle("btcprice.pkl")
# x1=prefilter(paths, start= datetime.date(*[2020,1,1]), cutoff=0)
# np.save('prefilteredhistory', x1)
x1 = np.load("prefilteredhistory.npy")
# y1=preprocess(x1[:,0], first_n=0, freq='1D')
# y1.to_pickle('preprocessedhistory.pkl')
y1 = pd.read_pickle("preprocessedhistory.pkl")
plt.style.use("default")
fig, ax = plt.subplots(1, 1)
y1.groupby(level=1)["Volume"].sum().plot(linewidth=0.5, color="k")
# plt.title('Volume of BTC traded')
plt.xlabel("Time")
plt.ylabel("BTC")
plt.tight_layout()
plt.legend()
plt.savefig(save_path_graphs + "binancehistory.png")
plt.show()

#%%
# GRAPH OF BTC PRICE
plt.style.use("default")
fig, ax = plt.subplots(1, 1)
btcusd = pd.read_pickle("btcprice.pkl")
btcusd["Close"].plot(linewidth=0.5, color="k", ax=ax)
plt.xlabel("Date")
plt.ylabel("BTC/USDT")
plt.legend()
plt.tight_layout()

plt.savefig(save_path_graphs + "btcprice.png")

#%%
# HISTOGRAMS AND KDE OF RETURNS
rdd = load_results("scenario1", "dist")
rhd = load_results("scenario3", "dist")
rdc = load_results("scenario1", "coint")
rhc = load_results("scenario3", "coint")
retdd = rdd["Profit"]
rethd = rhd["Profit"]
retdc = rdc["Profit"]
rethc = rhc["Profit"]
rets = {"rdd": retdd, "rhd": rethd, "rdc": retdc, "rhc": rethc}

#%%
# 2x2 PLOT OF KDES FOR APPENDIX
plt.style.use("default")
df = pd.read_pickle(
    "C:\\Users\\kawga\\Documents\\IES\Bach\\code\\Pairs-trading\\tables\\retstats.pkl"
)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
plt.xlim(-0.2, 0.2)
plt.sca(ax1)
plt.title("Daily")
ax1.set_ylabel("Distance")
rv = scipy.stats.norm(
    loc=df["Daily", "Distance"]["Mean"], scale=df["Daily", "Distance"]["Std"]
)
sns.kdeplot(retdd, linewidth=1, color="tab:blue")
sns.kdeplot(
    rv.rvs(size=100000), linewidth=1, label="Normal", linestyle="--", color="tab:red"
)

plt.sca(ax2)
plt.title("Hourly")
rv = scipy.stats.norm(
    loc=df["Hourly", "Distance"]["Mean"], scale=df["Hourly", "Distance"]["Std"]
)
plt.ylim(0, 55)
sns.kdeplot(rethd, linewidth=1)
sns.kdeplot(
    rv.rvs(size=100000), linewidth=1, label="Normal", linestyle="--", color="tab:red"
)

plt.sca(ax3)
rv = scipy.stats.norm(
    loc=df["Daily", "Cointegration"]["Mean"], scale=df["Daily", "Cointegration"]["Std"]
)
ax3.set_ylabel("Cointegration")
sns.kdeplot(retdc, linewidth=1)
sns.kdeplot(
    rv.rvs(size=100000), linewidth=1, label="Normal", linestyle="--", color="tab:red"
)

plt.sca(ax4)
rv = scipy.stats.norm(
    loc=df["Hourly", "Cointegration"]["Mean"],
    scale=df["Hourly", "Cointegration"]["Std"],
)
plt.ylim(0, 55)
sns.kdeplot(rethc, linewidth=1)
sns.kdeplot(
    rv.rvs(size=100000), linewidth=1, label="Normal", linestyle="--", color="tab:red"
)

plt.savefig(save_path_graphs + "retkdes", dpi=300)
#%%
def retkde(fig, ax, freq, method):
    fig.sca(ax)
    rv = scipy.stats.norm(loc=df[freq, method]["Mean"], scale=df[freq, method]["Std"])
    modifier1 = freq[0].lower()
    modifier2 = method[0].lower()
    sns.kdeplot(rets["ret" + modifier1 + modifier2], linewidth=1)
    sns.kdeplot(rv.rvs(size=100000), linewidth=1)


#%%

#%%
# QQ PLOT
plt.style.use("default")
df = pd.read_pickle(
    "C:\\Users\\kawga\\Documents\\IES\Bach\\code\\Pairs-trading\\tables\\retstats.pkl"
)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
plt.sca(ax1)
scipy.stats.probplot(
    retdd.dropna().values.astype("float32"), fit=True, plot=ax1, dist="norm"
)
plt.title(r"$\bf{Daily}$")
plt.xlabel("")
plt.ylabel(r"$\bf{Distance}$" + "\n" + r"$\mathrm{Ordered \: values}$", fontsize=11)
plt.sca(ax2)
scipy.stats.probplot(
    rethd.dropna().values.astype("float32"), fit=True, plot=ax2, dist="norm"
)
plt.title(r"$\bf{Hourly}$")
plt.ylabel("")
plt.xlabel("")
plt.sca(ax3)
scipy.stats.probplot(
    retdc.dropna().values.astype("float32"), fit=True, plot=ax3, dist="norm"
)
plt.title("")
plt.ylabel(r"$\bf{Cointegration}$" + "\n Ordered values")
plt.sca(ax4)
scipy.stats.probplot(
    rethc.dropna().values.astype("float32"), fit=True, plot=ax4, dist="norm"
)
plt.title("")
plt.ylabel("")

plt.tight_layout()
plt.savefig(save_path_graphs + "retqq", dpi=300)

#%%
# HOURLY DISTRIBUTION GRAPHS

rdd = load_results("scenario1", "dist")
rhd = load_results("scenario3", "dist")
rdc = load_results("scenario1", "coint")
rhc = load_results("scenario3", "coint")
retdd = rdd["Profit"]
rethd = rhd["Profit"]
retdc = rdc["Profit"]
rethc = rhc["Profit"]
hdisthd = hdist(rethd.astype("float32"))
hdisthc = hdist(rethc.astype("float32"))
hdisttable = pd.concat([hdisthd, hdisthc], keys=["Distance", "Cointegration"], axis=1)

#%%
meanplot = hdisttable[
    [
        ("Distance", "Returns distribution", "Mean"),
        ("Cointegration", "Returns distribution", "Mean"),
    ]
]
meanplotcorr = meanplot.corr()
meanplotcorr.index = meanplotcorr.index.droplevel([1, 2])
meanplotcorr.columns = meanplotcorr.columns.droplevel([1, 2])

plt.style.use("default")
fig, ax = plt.subplots(1, 1)
meanplot.columns = meanplot.columns.droplevel([1, 2])
meanplot.plot(ax=ax)
ax.properties()["children"][1].set_color("tab:red")
ax.properties()["children"][1].set_linestyle("--")
plt.ylabel("Average return")
plt.legend()
plt.tight_layout()
plt.savefig(save_path_graphs + "hdist", dpi=300)
#%%


#%%

