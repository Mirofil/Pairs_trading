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


def pick_range(df, start, end):
    """ Slices preprocessed index-wise to achieve y[start:end], taking into account the MultiIndex"""
    # There is a bug when the end is past the preprocessed length, one index of the level disappears?

    # if timeframe[1] in df.loc[name].index:
    #     end_of_formation = df.loc[name].index.get_loc(timeframe[1])
    # elif timeframe[1] - datetime.timedelta(days=1) in df.loc[name].index:
    #     end_of_formation = df.loc[name].index.get_loc(timeframe[1] - datetime.timedelta(days=1))
    # elif timeframe[1] - datetime.timedelta(days=2) in df.loc[name].index:
    #     end_of_formation = df.loc[name].index.get_loc(timeframe[1] - datetime.timedelta(days=2))

    past_start = df.index.levels[1] > pd.to_datetime(start)
    before_end = df.index.levels[1] <= pd.to_datetime(end)
    mask = (past_start) & (before_end)

    result = df.groupby(level=0).apply(lambda x: x.loc[mask])
    if result.index.names[0] == result.index.names[1]:
        result = result.droplevel(level=0)

    return result


def signals_numeric(olddf, copy=True):
    # TODO I THINK THIS IS NOT USED AND WORTHLESSS
    """ Prepares dummy variables so we can make a graph when the pair is open/close etc"""
    df = olddf.copy(deep=copy)
    for name, group in df.groupby(level=0):
        numeric = np.zeros((df.loc[name].shape[0]))
        numeric[
            (df.loc[name, "Signals"] == "Long")
            | (df.loc[name, "Signals"] == "keepLong")
        ] = 1
        numeric[
            (df.loc[name, "Signals"] == "Short")
            | (df.loc[name, "Signals"] == "keepShort")
        ] = -1
        df.loc[name, "Numeric"] = numeric
    return df


def signals_graph(df, pair, timeframe=None):
    # TODO I THINK THIS IS NOT USED AND WORTHLESSS
    if timeframe == None:
        df.loc[pair, "Numeric"].plot()
    else:
        sliced = pick_range(df, timeframe[0], timeframe[1]).loc[pair]
        sliced["Numeric"].plot()

    return 1


def sliced_norm(df, pair, column, timeframe):
    """ normalizes a dataframe by timeframe slice (afterwards, the mean overall
     is not actually 0 etc) """
    sliced = pick_range(df, timeframe[0], timeframe[1])
    diff = df.loc[pair[0], column] - df.loc[pair[1], column]
    mean = (sliced.loc[pair[0], column] - sliced.loc[pair[1], column]).mean()
    std = (sliced.loc[pair[0], column] - sliced.loc[pair[1], column]).std()
    return ((diff - mean) / std).values


def weights_from_signals(df, cost=0):
    """ Sets the initial weights on position open so they can be propagated"""
    df.loc[df["Signals"] == "Long", "1Weights"] = -df.loc[
        df["Signals"] == "Long", "SpreadBeta"
    ] * (1 + cost)
    df.loc[df["Signals"] == "Long", "2Weights"] = 1 * (1 - cost)
    df.loc[df["Signals"] == "Short", "1Weights"] = df.loc[
        df["Signals"] == "Short", "SpreadBeta"
    ] * (1 - cost)
    df.loc[df["Signals"] == "Short", "2Weights"] = -1 * (1 + cost)
    df.loc[
        (
            (df["Signals"] == "sellLong")
            | (df["Signals"] == "sellShort")
            | (df["Signals"] == "Sell")
        ),
        "1Weights",
    ] = 0
    df.loc[
        (
            (df["Signals"] == "sellLong")
            | (df["Signals"] == "sellShort")
            | (df["Signals"] == "Sell")
        ),
        "2Weights",
    ] = 0
    
def calculate_spreads(df, viable_pairs, timeframe, betas=None, show_progress_bar=True):
    """Picks out the viable pairs of the original df (which has all pairs)
    and adds to it the normPrice Spread among others, as well as initially
    defines Weights and Profit """
    idx = pd.IndexSlice
    spreads = []
    # the 1,1 betas are for Distance method and even though Cointegration would have different coeffs, I am not sure why it is here?
    if betas == None:
        betas = [np.array([1, 1]) for i in range(len(viable_pairs))]
    for pair, coefs in tqdm(
        zip(viable_pairs, betas),
        desc="Calculating spreads",
        disable=not show_progress_bar,
        total=len(viable_pairs),
    ):
        # labels will be IOTAADA rather that IOTABTCADABTC,
        # so we remove the last three characters
        first = re.sub(r"USDT$|USD$|BTC$", "", pair[0])
        second = re.sub(r"USDT$|USD$|BTC$", "", pair[1])
        composed = first + "x" + second
        multiindex = pd.MultiIndex.from_product(
            [[composed], df.loc[pair[0]].index], names=["Pair", "Time"]
        )
        newdf = pd.DataFrame(index=multiindex)
        newdf["1Weights"] = None
        newdf["2Weights"] = None
        newdf["Profit"] = 0
        newdf["normLogReturns"] = sliced_norm(df, pair, "logReturns", timeframe)
        newdf["1Price"] = df.loc[pair[0], "Price"].values
        newdf["2Price"] = df.loc[pair[1], "Price"].values
        newdf["Spread"] = (
            -coefs[1] * df.loc[pair[0], "Price"] + df.loc[pair[1], "Price"]
        ).values
        newdf["SpreadBeta"] = coefs[1]
        newdf["normSpread"] = (
            (
                newdf["Spread"]
                - newdf.loc[idx[composed, timeframe[0] : timeframe[1]], "Spread"].mean()
            )
            / newdf.loc[idx[composed, timeframe[0] : timeframe[1]], "Spread"].std()
        ).values
        # not sure what those lines do
        first = df.loc[pair[0]]
        first.columns = ["1" + x for x in first.columns]
        second = df.loc[pair[0]]
        second.columns = ["2" + x for x in second.columns]
        reindexed = (pd.concat([first, second], axis=1)).set_index(multiindex)

        spreads.append(newdf)
    return pd.concat(spreads)

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def propagate_weights(df, formation_timeframe: List):
    """Propagates weights according to price changes
    Timeframe should be Formation """
    for name, group in df.groupby(level=0):
        # The end of formation might fall on weekend and US stock data do not have any rows for non-trading days
        if formation_timeframe[1] in df.loc[name].index:
            end_of_formation = df.loc[name].index.get_loc(formation_timeframe[1])
        elif formation_timeframe[1] - datetime.timedelta(days=1) in df.loc[name].index:
            end_of_formation = df.loc[name].index.get_loc(
                formation_timeframe[1] - datetime.timedelta(days=1)
            )
        elif formation_timeframe[1] - datetime.timedelta(days=2) in df.loc[name].index:
            end_of_formation = df.loc[name].index.get_loc(
                formation_timeframe[1] - datetime.timedelta(days=2), method = 'backfill'
            )
        else:
            end_of_formation = df.loc[name].index.get_loc(
                formation_timeframe[1] - datetime.timedelta(days=2), method = 'nearest'
            )

        temp_weights1 = group["1Weights"].to_list()
        temp_weights2 = group["2Weights"].to_list()
        return1 = group["1Price"] - group["1Price"].shift(1)
        return2 = group["2Price"] - group["2Price"].shift(1)
        # print(end_of_formation, len(group.index), name)
        for i in range(end_of_formation + 1, len(group.index)):
            #I think shifting the index to god knows where might break this? Better put try to be safe
            try:
                if group.iloc[i]["Signals"] in ["keepLong", "keepShort"]:

                    temp_weights1[i] = temp_weights1[i - 1] * (1 + return1.iloc[i])
                    temp_weights2[i] = temp_weights2[i - 1] * (1 + return2.iloc[i])
            except:
                continue
        df.loc[name, "1Weights"] = temp_weights1
        df.loc[name, "2Weights"] = temp_weights2


def calculate_profit(df, cost=0):
    """Inplace calculates the profit per period as well as a cumulative profit
    Be careful to have the same cost as weights_from_signals
    This function counts the cost in Profit, while w_f_s does it for weights
    So its not double counting or anything"""
    idx = pd.IndexSlice
    mask = (df.loc[idx[:, "Signals"]] == "Long") | (
        df.loc[idx[:, "Signals"]] == "Short"
    )
    # used for cumProfit propagation
    mask2 = (
        (df.loc[idx[:, "Signals"]] == "Long")
        | (df.loc[idx[:, "Signals"]] == "Short")
        | (df.loc[idx[:, "Signals"]] == "keepShort")
        | (df.loc[idx[:, "Signals"]] == "keepLong")
        | (df.loc[idx[:, "Signals"]] == "Sell")
        | (df.loc[idx[:, "Signals"]] == "sellShort")
        | (df.loc[idx[:, "Signals"]] == "sellLong")
    )
    for name, group in df.groupby(level=0):
        returns1 = group["1Price"] - group["1Price"].shift(1).values
        returns2 = group["2Price"] - group["2Price"].shift(1).values
        temp = returns1 + returns2
        df.loc[name, "Profit"] = (
            df.loc[name, "1Weights"].shift(1) * returns1
            + df.loc[name, "2Weights"].shift(1) * returns2
        )
        df.loc[idx[name, mask], "Profit"] = -(
            df.loc[idx[name, mask], "1Weights"].abs() * cost
            + df.loc[idx[name, mask], "2Weights"].abs() * cost
        )
        df.loc[idx[name, mask2], "cumProfit"] = (
            df.loc[idx[name, mask2], "Profit"]
        ).cumsum() + 1


def signals_worker(
    multidf,
    start_date,
    end_date,
    trading_timeframe=5,
    formation=5,
    threshold=2,
    lag=0,
    stoploss=100,
    num_of_processes=1,
):

    # Without the copy, it was causing bugs since this there is in-place mutation - in particular, the simulation scheme where we share the dist/coint signals and have multiple parameters after that would cause problems
    multidf = multidf.copy(deep=True)

    #Stoploss should signify the number in excess of the threshold that causes stoploss!
    stoploss = threshold + stoploss
    idx = pd.IndexSlice
    for name, df in multidf.loc[
        pd.IndexSlice[:, trading_timeframe[0] : trading_timeframe[1]], :
    ].groupby(level=0):
        df["Signals"] = None
        # df.loc[mask,'Signals'] = True
        index = df.index
        # this is technicality because we truncate the DF to just trading period but
        # in the first few periods the signal generation needs to access prior values
        # which would be None so we just make them adhoc like this
        col = [None for x in range(lag + 2)]
        fill = "None"
        for i in range(len(df)):
            truei = i
            if i - lag < 0:
                col.append(fill)
                continue
            if (df.loc[index[i - lag], "normSpread"] > stoploss) & (
                col[i + lag + 1] in ["Short", "keepShort"]
            ):
                fill = "stopShortLoss"
                col.append(fill)
                fill = "None"
                continue
            if (df.loc[index[i - lag], "normSpread"] < (-stoploss)) & (
                col[i + lag + 1] in ["Long", "keepLong"]
            ):
                fill = "stopLongLoss"
                col.append(fill)
                fill = "None"
                continue
            if (
                (df.loc[index[i - lag], "normSpread"] >= threshold)
                & (df.loc[index[i - lag - 1], "normSpread"] < threshold)
                & (col[i + lag - 1] not in ["keepShort"])
                & (col[truei + lag + 1] not in ["Short", "keepShort"])
            ):
                fill = "Short"
                col.append(fill)
                fill = "keepShort"
                continue
            elif (
                (df.loc[index[i - lag], "normSpread"] <= 0)
                & (df.loc[index[i - lag - 1], "normSpread"] > 0)
                & (col[i + lag + 1] in ["Short", "keepShort"])
            ):
                fill = "sellShort"
                col.append(fill)
                fill = "None"
                continue
            elif (
                (
                    (df.loc[index[i - lag], "normSpread"] <= (-threshold))
                    & (df.loc[index[i - lag - 1], "normSpread"] > (-threshold))
                )
                & (col[i + lag - 1] not in ["keepLong"])
                & (col[truei + lag + 1] not in ["Long", "keepLong"])
            ):
                # print(i, col, name, col[truei+lag]!='Long', truei)
                fill = "Long"
                col.append(fill)
                fill = "keepLong"
                continue
            elif (
                (df.loc[index[i - lag], "normSpread"] >= 0)
                & (df.loc[index[i - lag - 1], "normSpread"] < 0)
                & (col[i + lag + 1] in ["Long", "keepLong"])
            ):
                fill = "sellLong"
                col.append(fill)
                fill = "None"
                continue
            col.append(fill)
        col = col[(lag + 2) : -1]
        col.append("Sell")
        # df['Signals'] = pd.Series(col[1:], index=df.index)
        multidf.loc[
            pd.IndexSlice[name, trading_timeframe[0] : trading_timeframe[1]], "Signals"
        ] = pd.Series(col, index=df.index)
        multidf.loc[
            pd.IndexSlice[name, trading_timeframe[1] : end_date], "Signals"
        ] = None
    multidf.loc[idx[:, trading_timeframe[1] : end_date], "Signals"] = None
    multidf.loc[idx[:, trading_timeframe[1] : end_date], "Signals"] = multidf.loc[
        idx[:, trading_timeframe[1] : end_date], "Signals"
    ].fillna(value="pastFormation")
    multidf.loc[idx[:, formation[0] : formation[1]], "Signals"] = multidf.loc[
        idx[:, formation[0] : formation[1]], "Signals"
    ].fillna(value="Formation")
    multidf.loc[idx[:, start_date : formation[0]], "Signals"] = multidf.loc[
        idx[:, start_date : formation[0]], "Signals"
    ].fillna(value="preFormation")
    # multidf['Signals'] = multidf['Signals'].fillna(value='Formation')
    return multidf


def signals(
    multidf,
    start_date,
    end_date,
    trading_timeframe=None,
    formation=None,
    threshold=2,
    lag=0,
    stoploss=100,
    num_of_processes=1,
):
    """ Fills in the Signals during timeframe period 
    Outside of the trading period, it fills Formation and pastTrading"""
    if num_of_processes == 1:

        return signals_worker(
            multidf,
            start_date=start_date,
            end_date=end_date,
            trading_timeframe=trading_timeframe,
            formation=formation,
            threshold=threshold,
            lag=lag,
        )
    if num_of_processes > 1:
        if len(multidf.index.unique(level=0)) < num_of_processes:
            num_of_processes = len(multidf.index.unique(level=0))
        pool = mp.Pool(num_of_processes)
        split = np.array_split(multidf.index.unique(level=0), num_of_processes)
        split = [multidf.loc[x] for x in split]
        # Im not sure what I was doing here to be honest..
        args_dict = {
            "start_date": start_date,
            "end_date": end_date,
            "trading": trading_timeframe,
            "formation": formation,
            "threshold": threshold,
            "lag": lag,
            "stoploss": stoploss,
            "num_of_processes": num_of_processes,
        }
        args = [
            args_dict["start_date"],
            args_dict["end_date"],
            args_dict["trading"],
            args_dict["formation"],
            args_dict["threshold"],
            args_dict["lag"],
            args_dict["stoploss"],
            args_dict["num_of_processes"],
        ]
        full_args = [[split[i], *args] for i in range(len(split))]
        results = pool.starmap(signals_worker, full_args)
        results = pd.concat(results)
        pool.close()
        pool.join()
        return results
