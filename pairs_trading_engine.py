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

from config import data_path, end_date, start_date

def pick_range(y, start, end):
    """ Slices preprocessed index-wise to achieve y[start:end], taking into account the MultiIndex"""
    past_start = y.index.levels[1] > pd.to_datetime(start)
    before_end = y.index.levels[1] <= pd.to_datetime(end)
    mask = (past_start) & (before_end)
    return y.groupby(level=0).apply(lambda x: x.loc[mask]).droplevel(level=0)

def signals_numeric(olddf, copy=True):
    #TODO I THINK THIS IS NOT USED AND WORTHLESSS
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
    #TODO I THINK THIS IS NOT USED AND WORTHLESSS
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

def propagate_weights(df, timeframe: List):
    """Propagates weights according to price changes
    Timeframe should be Formation """
    for name, group in df.groupby(level=0):
        end_of_formation = df.loc[name].index.get_loc(timeframe[1])
        temp_weights1 = group["1Weights"].to_list()
        temp_weights2 = group["2Weights"].to_list()
        return1 = group["1Price"] - group["1Price"].shift(1)
        return2 = group["2Price"] - group["2Price"].shift(1)
        # print(end_of_formation, len(group.index), name)
        for i in range(end_of_formation + 1, len(group.index)):
            if group.iloc[i]["Signals"] in ["keepLong", "keepShort"]:
                # print(temp_weights1[i-1], temp_weights2[i-1])
                # print(group.index[i])
                # df.loc[(name,group.index[i]),'1Weights']=df.loc[(name, group.index[i-1]), '1Weights']*1.1
                # not sure if the indexes are matched correctly here
                temp_weights1[i] = temp_weights1[i - 1] * (1 + return1.iloc[i])
                temp_weights2[i] = temp_weights2[i - 1] * (1 + return2.iloc[i])
        df.loc[name, "1Weights"] = temp_weights1
        df.loc[name, "2Weights"] = temp_weights2

# def propagate_weights2(df, timeframe):
#     idx = pd.IndexSlice
#     grouped = df.groupby(level=0)
#     for name, group in df.groupby(level=0):
#         end_of_formation = df.loc[name].index.get_loc(timeframe[1])
#         return1 = group["1Price"] - group["1Price"].shift(1)
#         return2 = group["2Price"] - group["2Price"].shift(1)
#         mask = (df["Signals"] == "keepLong") | (df["Signals"] == "keepShort")
#         # mask = (group['Signals']=='keepLong')|(group['Signals']=='keepShort')
#         cumreturn1 = (return1 + 1).loc[idx[mask]].cumprod()
#         cumreturn2 = (return2 + 1).cumprod()
#         # print(len(mask))
#         # print(df.loc[idx[name, mask], '1Weights'])
#         # print(cumreturn1)
#         # print(df.loc[idx[name, mask], '1Weights'])
#         df.loc[idx[name, mask], "1Weights"] = (
#             df.loc[idx[name, mask], "1Weights"].shift(1) * cumreturn1
#         )
#         # df.loc[idx[name,mask],'1Weights']=5

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
    timeframe=5,
    formation=5,
    threshold=2,
    lag=0,
    stoploss=100,
    num_of_processes=1,
):
    global end_date
    idx = pd.IndexSlice
    for name, df in multidf.loc[
        pd.IndexSlice[:, timeframe[0] : timeframe[1]], :
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
            pd.IndexSlice[name, timeframe[0] : timeframe[1]], "Signals"
        ] = pd.Series(col, index=df.index)
    multidf.loc[idx[:, timeframe[1] : end_date], "Signals"] = multidf.loc[
        idx[:, timeframe[1] : end_date], "Signals"
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
    timeframe=None,
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
            timeframe=timeframe,
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
            "trading": timeframe,
            "formation": formation,
            "threshold": threshold,
            "lag": lag,
            "stoploss": stoploss,
            "num_of_processes": num_of_processes,
        }
        args = [
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
