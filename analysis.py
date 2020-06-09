import datetime
import os
import pickle
import re
import shutil
from contextlib import contextmanager
from typing import *

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
import scipy
import statsmodels
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from config import data_path

def corrs(df):
    cols = ["Sharpe", "Sortino", "Calmar", "VaR"]
    arr = pd.DataFrame(columns=cols, index=cols)
    ps = pd.DataFrame(columns=cols, index=cols)
    df.replace([np.inf, -np.inf], np.nan)
    mask = (
        (df["Sharpe"].notnull())
        & (df["Sortino"].notnull())
        & (df["VaR"].notnull())
        & (df["Calmar"].notnull())
    )
    for i in range(len(cols)):
        for j in range(len(cols)):
            arr.loc[cols[i], cols[j]] = scipy.stats.spearmanr(
                df[cols[i]].loc[mask], df[cols[j]].loc[mask]
            )[0]
            ps.loc[cols[i], cols[j]] = rhoci(
                scipy.stats.spearmanr(df[cols[i]].loc[mask], df[cols[j]].loc[mask])[0],
                len(df[cols[i]].loc[mask]),
            )
    return (arr, ps)

def descriptive_stats(
    df,
    timeframe=5,
    freq="daily",
    riskfree=0.02,
    tradingdays=60,
    nonzero=False,
    trades_nonzero=False,
):
    """Input: one period of all pairs history, just one specific pair wont work
    Output: Summary statistics for every pair"""
    idx = pd.IndexSlice
    stats = pd.DataFrame(
        index=df.index.unique(level=0),
        columns=[
            "Mean",
            "Total profit",
            "Std",
            "Sharpe",
            "Number of trades",
            "Avg length of position",
            "Pct of winning trades",
            "Max drawdown",
        ],
    )
    desc = pd.DataFrame(
        index=["avg"],
        columns=[
            "Mean",
            "Total profit",
            "Std",
            "Sharpe",
            "Number of trades",
            "Avg length of position",
            "Pct of winning trades",
            "Max drawdown",
        ],
    )
    trad = infer_periods(df)["trading"]
    tradingdays = abs((trad[0][1] - trad[1][1]).days)
    annualizer = 365 / tradingdays
    monthlizer = 30 / tradingdays
    riskfree = riskfree / annualizer
    for name, group in df.groupby(level=0):
        stats.loc[name, "Mean"] = group["Profit"].mean()
        stats.loc[name, "Total profit"] = group["Profit"].sum()
        stats.loc[name, "Std"] = group["Profit"].std()
        stats.loc[name, "Number of trades"] = len(
            group[group["Signals"] == "Long"]
        ) + len(group[group["Signals"] == "Short"])
        # stats.loc[name, 'Roundtrip trades']=(len(group[group['Signals']=='sellLong'])+len(group[group['Signals']=='sellShort']))
        stats.loc[name, "Roundtrip trades"] = (
            len(group[group["Signals"] == "sellLong"])
            + len(group[group["Signals"] == "sellShort"])
        ) / max(1, stats.loc[name, "Number of trades"])
        # stats.loc[name, 'Avg length of position'] = ((len(group[group['Signals']=='keepLong'])/max(len(group[group['Signals']=='Long']),1))+len(group[group['Signals']=='keepShort'])/max(len(group[group['Signals']=='Short']),1))/max(1,stats.loc[name, 'Number of trades'])
        stats.loc[name, "Avg length of position"] = (
            (len(group[group["Signals"] == "keepLong"]))
            + len(group[group["Signals"] == "keepShort"])
        ) / max(1, stats.loc[name, "Number of trades"])
        stats.loc[name, "Max drawdown"] = abs(drawdown(group).min())
        neg_mask = group["Profit"] < 0
        stats.loc[name, "Downside Std"] = group.loc[neg_mask, "Profit"].std()
        stats.loc[name, "Sortino"] = (
            stats.loc[name, "Total profit"] - riskfree
        ) / stats.loc[name, "Downside Std"]
        stats.loc[name, "Sharpe"] = (
            stats.loc[name, "Total profit"] - riskfree
        ) / stats.loc[name, "Std"]
        stats.loc[name, "Monthly profit"] = (
            (stats.loc[name, "Total profit"] + 1) ** monthlizer
        ) - 1
        stats.loc[name, "Annual profit"] = (
            (stats.loc[name, "Total profit"] + 1) ** (annualizer)
        ) - 1
        if (pd.isna(group["Profit"].quantile(0.05)) == False) & (
            group["Profit"].quantile(0.05) != 0
        ):
            stats.loc[name, "VaR"] = -(
                stats.loc[name, "Total profit"] - riskfree
            ) / group["Profit"].quantile(0.05)
        else:
            stats.loc[name, "VaR"] = None
        stats.loc[name, "Calmar"] = (
            stats.loc[name, "Annual profit"] / stats.loc[name, "Max drawdown"]
        )
        last_valid = df.loc[
            idx[name, timeframe[0] : timeframe[1]], "cumProfit"
        ].last_valid_index()
        # if the pair never trades, then last_valid=None and it would fuck up indexing later
        if last_valid == None:
            last_valid = timeframe[1]
        # we have to make distinction here for the assignment to stats[CumProfit] to work
        # because sometimes we would assign None and sometimes a Series which would give error
        # the sum() is just to convert the Series to a scalar
        if isinstance(df.loc[idx[name, last_valid], "cumProfit"], pd.Series):
            stats.loc[name, "Cumulative profit"] = df.loc[
                idx[name, last_valid], "cumProfit"
            ].sum()
        else:
            stats.loc[name, "Cumulative profit"] = 1

        # picks the dates on which trades have ended
        mask2 = find_trades(df.loc[idx[name, :], :], timeframe)[1]
        stats.loc[name, "Pct of winning trades"] = (
            df.loc[idx[name, :], "cumProfit"][mask2] > 1
        ).sum() / max(stats.loc[name, "Number of trades"], 1)
        if nonzero == True:
            stats.loc[
                stats["Number of trades"] == 0,
                ["Mean", "Total profit", "Monthly profit", "Annual profit"],
            ] = None
        if trades_nonzero == True:
            stats.loc[
                stats["Number of trades"] == 0,
                ["Roundtrip trades", "Avg length of position"],
            ] = None
    return stats

def descriptive_frame(olddf):
    diag = [
        "Monthly profit",
        "Annual profit",
        "Total profit",
        "Std",
        "Sharpe",
        "Sortino",
        "VaR",
        "Calmar",
        "Number of trades",
        "Roundtrip trades",
        "Avg length of position",
        "Pct of winning trades",
        "Max drawdown",
        "Cumulative profit",
    ]
    idx = pd.IndexSlice
    # rebuilds the MultiIndex?
    temp = [[], []]
    for i in range(len(olddf.index.unique(level=0))):
        temp[0].append([i for x in range(len(olddf.loc[i].index.unique(level=0)))])
        temp[1].append([item for item in olddf.loc[i].index.unique(level=0).array])
    temp[0] = [item for sublist in temp[0] for item in sublist]
    temp[1] = [item for sublist in temp[1] for item in sublist]
    df = pd.DataFrame(index=temp, columns=diag)
    # print(df)
    for name, group in df.groupby(level=0):
        test_df = olddf.loc[name].index.unique(level=0)[0]
        stats = descriptive_stats(
            olddf.loc[name], infer_periods(olddf.loc[(name, test_df)])["trading"]
        )
        for col in df.loc[name].columns:
            df.loc[idx[name, :], col] = stats[col].values

    return df.astype("float32")

def summarize(df, index, mean=False):
    """ Summarizes the return distribution"""
    if mean == True:
        df = df.astype("float32").groupby(level=0).mean()
    res = pd.DataFrame(index=index, columns=[0])
    res.loc["Mean"] = df.mean()
    res.loc["Std"] = df.std()
    res.loc["Max"] = df.max()
    res.loc["Min"] = df.min()
    jb = statsmodels.stats.stattools.jarque_bera(df.dropna().values)
    res.loc["Jarque-Bera p-value"] = jb[1]
    res.loc["Kurtosis"] = jb[3]
    res.loc["Skewness"] = jb[2]
    count = df.count()
    res.loc["Positive"] = sum(df > 0) / count
    res.loc["t-stat"] = res.loc["Mean"] / res.loc["Std"] * (count) ** (1 / 2)
    return res

def aggregate(
    dfs,
    feasible,
    freqs=[60, 60, 10, 10],
    standard=True,
    returns_nonzero=False,
    trades_nonzero=False,
):
    temp = []
    for i in range(len(dfs)):
        df = dfs[i]
        numnom = len(df.index.get_level_values(level=1)) / (df.index[-1][0] + 1)
        numtr = len(df[df["Number of trades"] > 0].index.get_level_values(level=1)) / (
            df.index[-1][0] + 1
        )
        if returns_nonzero == True:
            df.loc[
                df["Number of trades"] == 0,
                ["Total profit", "Monthly profit", "Annual profit"],
            ] = None
        if trades_nonzero == True:
            df.loc[
                df["Number of trades"] == 0,
                ["Roundtrip trades", "Avg length of position"],
            ] = None
        mean = df.groupby(level=0).mean()
        mean["Trading period Sharpe"] = (
            mean["Total profit"] - (0.02 / (365 / freqs[i]))
        ) / mean["Std"]
        mean["Annualized Sharpe"] = mean["Trading period Sharpe"] * (
            (365 / freqs[i]) ** (1 / 2)
        )
        mean = mean.mean()
        mean["Annual profit"] = (1 + mean["Total profit"]) ** (365 / freqs[i]) - 1
        mean["Monthly profit"] = (1 + mean["Total profit"]) ** (30 / freqs[i]) - 1
        mean["Nominated pairs"] = numnom
        mean["Traded pairs"] = numtr
        mean["Traded pairs"] = mean["Traded pairs"] / mean["Nominated pairs"]
        temp.append(mean[feasible])
    concated = pd.concat(temp, axis=1)
    if standard == True:
        cols = pd.MultiIndex.from_product(
            [["Daily", "Hourly", "5-Minute"], ["Dist.", "Coint."]]
        )
        concated.columns = cols
    return concated

def rhoci(rho, n, conf=0.95):
    mean = np.arctanh(rho)
    std = 1 / ((n - 3) ** (1 / 2))
    norm = scipy.stats.norm(loc=mean, scale=std)
    ci = [mean - 1.96 * std, mean + 1.96 * std]
    trueci = [np.round(np.tanh(ci[0]), 2), np.round(np.tanh(ci[1]), 2)]
    return trueci


