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

from pairs.config import data_path


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


def infer_periods(single_backtest_df: pd.DataFrame, trading_delta = None):
    """Auto detects the Formation and Trading periods
    Works even with MultiIndexed since the periods are the same across all pairs"""
    trading_period_mask = ~(
        (single_backtest_df["Signals"] == "Formation")
        | (single_backtest_df["Signals"] == "pastFormation")
        | (single_backtest_df["Signals"] == "preFormation")
    )
    formation_mask = single_backtest_df["Signals"] == "Formation"
    # All pairs should have the same trading/formation periods so it does not matter which ones we pick
    example_pair = trading_period_mask.index.get_level_values(0)[0]
    formation_period = formation_mask.loc[(example_pair, slice(None))].loc[
        formation_mask.loc[(example_pair, slice(None))].values
    ]
    formation = (
        formation_period.index.get_level_values("Time")[0],
        formation_period.index.get_level_values("Time")[-1],
    )
    if trading_delta is None:
        trading_period = trading_period_mask.loc[(example_pair, slice(None))].loc[
            trading_period_mask.loc[(example_pair, slice(None))].values
        ]
        trading = (
            trading_period.index.get_level_values("Time")[0],
            trading_period.index.get_level_values("Time")[-1],
        )
    else:
        trading = (
            formation[1],
            formation[1] + trading_delta,
        )

    return {"formation": formation, "trading": trading}


def descriptive_stats(
    single_backtest_df: pd.DataFrame,
    trading_timeframe=None,
    freq: str = "daily",
    risk_free: int = 0.02,
    nonzero: bool = False,
    trades_nonzero: bool = False,
):
    """Input: one period of all pairs history, just one specific pair wont work
    Output: Summary statistics for every pair"""
    idx = pd.IndexSlice
    stats = pd.DataFrame(
        index=single_backtest_df.index.unique(level=0),
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
    if trading_timeframe is None:
        periods = infer_periods(single_backtest_df)
        trading_days = abs((periods["trading"][0] - periods["trading"][1]).days)
        trading_timeframe = periods["trading"]
    else:
        trading_days = abs((trading_timeframe[0] - trading_timeframe[1]).days)

    annualizer = 365 / trading_days
    monthlizer = 30 / trading_days
    risk_free = risk_free / annualizer
    for name, group in single_backtest_df.groupby(level=0):
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
            stats.loc[name, "Total profit"] - risk_free
        ) / stats.loc[name, "Downside Std"]
        stats.loc[name, "Sharpe"] = (
            stats.loc[name, "Total profit"] - risk_free
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
                stats.loc[name, "Total profit"] - risk_free
            ) / group["Profit"].quantile(0.05)
        else:
            stats.loc[name, "VaR"] = None
        stats.loc[name, "Calmar"] = (
            stats.loc[name, "Annual profit"] / stats.loc[name, "Max drawdown"]
        )
        last_valid = single_backtest_df.loc[
            idx[name, trading_timeframe[0] : trading_timeframe[1]], "cumProfit"
        ].last_valid_index()
        # if the pair never trades, then last_valid=None and it would fuck up indexing later
        if last_valid == None:
            last_valid = trading_timeframe[1]
        # we have to make distinction here for the assignment to stats[CumProfit] to work
        # because sometimes we would assign None and sometimes a Series which would give error
        # the sum() is just to convert the Series to a scalar
        if isinstance(
            single_backtest_df.loc[idx[name, last_valid], "cumProfit"], pd.Series
        ):
            stats.loc[name, "Cumulative profit"] = single_backtest_df.loc[
                idx[name, last_valid], "cumProfit"
            ].sum()
        else:
            stats.loc[name, "Cumulative profit"] = 1

        # picks the dates on which trades have ended
        mask2 = find_trades(single_backtest_df.loc[idx[name, :], :], trading_timeframe)[
            1
        ]
        stats.loc[name, "Pct of winning trades"] = (
            single_backtest_df.loc[idx[name, :], "cumProfit"][mask2] > 1
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


def descriptive_frame(olddf, show_progress_bar=False, trading_delta=None):
    # this should be a subset of the statistics from descriptive_stats I think
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
    # rebuilds the MultiIndex? Seems to go from BACKTEST_INDEX, PAIR, TIME to BACKTEST_INDEX, PAIR with the same columns
    # (or rather, with the diag on colums at the end which are quite close to the originals)
    temp = [[], []]
    for i in (olddf.index.unique(level=0)):
        temp[0].append([i for x in range(len(olddf.loc[i].index.unique(level=0)))])
        temp[1].append([item for item in olddf.loc[i].index.unique(level=0).array])
    temp[0] = [item for sublist in temp[0] for item in sublist]
    temp[1] = [item for sublist in temp[1] for item in sublist]
    df = pd.DataFrame(index=temp, columns=diag)

    # This is meant to be iteration over all the backtest indexes (0,1,..,N)
    for backtest_index, _ in tqdm(
        df.groupby(level=0), desc="Constructing descriptive frames over backtests", disable= not show_progress_bar
    ):
        backtest_index = int(backtest_index)
        trading_timeframe=infer_periods(olddf.loc[backtest_index], trading_delta=trading_delta)["trading"]

        stats = descriptive_stats(
            olddf.loc[backtest_index],
            trading_timeframe=trading_timeframe,
        )
        for col in df.loc[backtest_index].columns:
            df.loc[idx[backtest_index, :], col] = stats[col].values

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

def compute_period_length(specification:List):
    return specification[0]*30+specification[1]

def compute_period_length_in_days(freq:str):
    multiplier = None
    if freq =='1D':
        multiplier = 1
    elif freq == '1H':
        multiplier = 1/24
    elif freq == '5T':
        multiplier = 1/288

    assert multiplier is not None

    return multiplier

def compute_cols_from_freq(freqs:List[str], methods:List[str]):
    """ Useful for automatically populating the parameters in aggregate
    >>> compute_cols_from_freq(["1D"], ["dist"]) """
    results = []
    for freq in freqs:
        if freq[-1] == 'D':
            results.append('Daily')
        elif freq[-1] == 'H':
            results.append('Hourly')
        elif freq[-1] == 'T':
            results.append(f"{freq[:-2]}-Minute")
    return [results, methods]
    

def aggregate(
    descriptive_frames: List[pd.DataFrame],
    columns_to_pick:List[str]=None,
    trading_period_days:List[int]=[60, 60, 10, 10],
    multiindex_from_product_cols=[["Daily", "Hourly", "5-Minute"], ["Dist.", "Coint."]],
    returns_nonzero=True,
    trades_nonzero=True,
):
    assert len(trading_period_days) == len(descriptive_frames)
    assert len(multiindex_from_product_cols[0])*len(multiindex_from_product_cols[1]) == len(descriptive_frames)
    temp = []

    if columns_to_pick is None:
        columns_to_pick = [
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

    for i in range(len(descriptive_frames)):
        desc_frame = descriptive_frames[i]
        num_nominated = len(desc_frame.index.get_level_values(level=1)) / (
            desc_frame.index[-1][0] + 1
        )
        number_of_trades = len(
            desc_frame[desc_frame["Number of trades"] > 0].index.get_level_values(
                level=1
            )
        ) / (desc_frame.index[-1][0] + 1)
        if returns_nonzero == True:
            desc_frame.loc[
                desc_frame["Number of trades"] == 0,
                ["Total profit", "Monthly profit", "Annual profit"],
            ] = None
        if trades_nonzero == True:
            desc_frame.loc[
                desc_frame["Number of trades"] == 0,
                ["Roundtrip trades", "Avg length of position"],
            ] = None
        mean = desc_frame.groupby(level=0).mean()
        mean["Trading period Sharpe"] = (
            mean["Total profit"] - (0.02 / (365 / trading_period_days[i]))
        ) / mean["Std"]
        mean["Annualized Sharpe"] = mean["Trading period Sharpe"] * (
            (365 / trading_period_days[i]) ** (1 / 2)
        )
        mean = mean.mean()
        mean["Annual profit"] = (1 + mean["Total profit"]) ** (365 / trading_period_days[i]) - 1
        mean["Monthly profit"] = (1 + mean["Total profit"]) ** (30 / trading_period_days[i]) - 1
        mean["Nominated pairs"] = num_nominated
        mean["Traded pairs"] = number_of_trades
        mean["Traded pairs"] = mean["Traded pairs"] / mean["Nominated pairs"]
        temp.append(mean[columns_to_pick])
    concated = pd.concat(temp, axis=1)
    cols = pd.MultiIndex.from_product(
        multiindex_from_product_cols
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


def drawdown(df):
    """Calculates the maximum drawdown. Window is just meant to be bigger than examined period"""
    window = 25000
    roll_max = df["cumProfit"].rolling(window, min_periods=1).max()
    daily_drawdown = df["cumProfit"] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
    return max_daily_drawdown


def find_trades(df, timeframe=5):
    """ Identifies the periods where we actually trade the pairs"""
    idx = pd.IndexSlice
    starts = (df.loc[idx[:], "Signals"] == "Long") | (
        df.loc[idx[:], "Signals"] == "Short"
    )
    ends = (df.loc[idx[:], "Signals"] == "sellLong") | (
        df.loc[idx[:], "Signals"] == "sellShort"
    )
    if starts.sum() > ends.sum():
        ends = ends | (df.loc[idx[:], "Signals"] == "Sell")
    return (starts, ends)
