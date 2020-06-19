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

from pairs.config import data_path, start_date, end_date

def name_from_path(path: str):
    """ Goes from stuff like C:\Bach\concat_data\[pair].csv to [pair]"""
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def path_from_name(name: str, data_path=data_path):
    """ Goes from stuff like [pair] to C:\Bach\concat_data\[pair].csv"""
    path = os.path.join(data_path, name + ".csv")
    return name


# def prefilter(paths, start=start_date, end=end_date, cutoff=0.7):
#     """ Prefilters the time series so that we have only moderately old pairs (listed past start_date)
#     and uses a volume percentile cutoff. The output is in array (pair, its volume) """
#     idx = pd.IndexSlice
#     admissible = []
#     for i in tqdm(
#         range(len(paths)),
#         desc="Prefiltering pairs (based on volume and start/end of trading)",
#     ):
#         df = pd.read_csv(paths[i])
#         df.rename({"Opened": "Date"}, axis="columns", inplace=True, errors='ignore')
#         # filters out pairs that got listed past start_date
#         if (pd.to_datetime(df.iloc[0]["Date"]) < pd.to_datetime(start)) and (
#             pd.to_datetime(df.iloc[-1]["Date"]) > pd.to_datetime(end)
#         ):
#             # the Volume gets normalized to BTC before sorting
#             df = df.set_index("Date")
#             df = df.sort_index()
#             admissible.append(
#                 [
#                     paths[i],
#                     (
#                         df.loc[idx[str(start) : str(end)], "Volume"]
#                         * df.loc[idx[str(start) : str(end)], "Close"]
#                     ).sum(),
#                 ]
#             )
#     # sort by Volume and pick upper percentile
#     admissible.sort(key=lambda x: x[1])
#     admissible = admissible[int(np.round(len(admissible) * cutoff)) :]

#     #TODO NOTE!!! BEFORE THIS WOULD RETURN np.array(admissible), so that might be needed for backwards compatibility!
#     #I only go sort of halfway with the refactorization to make it easy to change back if needed..
#     result = pd.DataFrame(np.array(admissible), columns = ["0", "1"])
#     result.columns = [str(col) for col in result.columns]
#     return result


def resample(df, freq: str ="1D", start=start_date, fill: bool =True):
    """ Our original data is 1-min resolution, so we resample it to arbitrary frequency.
    Close prices get last values, Volume gets summed. 
    Only indexes past start_date are returned to have a common start for all series 
    (since they got listed at various dates)"""
    df.index = pd.to_datetime(df.Date)
    # Close prices get resampled with last values, whereas Volume gets summed
    if freq is not None:
        df["Close"] = df["Close"].resample(freq).last()
        df["Volume"] = df["Volume"] * df["Close"]
        df = df.resample(freq).agg({"Volume": np.sum})
    else:
        df["Volume"] = df["Volume"] * df["Close"]
    # log returns and normalization
    df["Close"] = df["Close"]
    if fill == True:
        df["Close"] = df["Close"].fillna(method="ffill")
    df["logClose"] = np.log(df["Close"])
    df["logReturns"] = (
        df["logClose"] - df["logClose"].shift(1)
    ).values
    df["Price"] = df["logReturns"].cumsum()
    return df[df.index > pd.to_datetime(start)]


# def preprocess(paths, freq:str ="1D", end=end_date, first_n: int=0, start=start_date):
#     """Finishes the preprocessing based on prefiltered paths. We filter out pairs that got delisted early
#     (they need to go at least as far as end_date). Then all the eligible time series for pairs formation analysis
#     are concated into one big DF with a multiIndex (pair, time)."""

#     paths = paths[first_n:]
#     preprocessed = []
#     for i in tqdm(range(len(paths)), desc="Preprocessing files"):
#         raw_coin = pd.read_csv(paths[i])
#         # The new Binance_fetcher API downloads Date as Opened instead..
#         raw_coin.rename({"Opened": "Date"}, axis="columns", inplace=True)
#         raw_coin = raw_coin.sort_index()
#         raw_coin = resample(raw_coin, freq, start=start)
#         raw_coin = raw_coin.sort_index()
#         # truncates the time series to a slightly earlier end date
#         # because the last period is inhomogeneous due to pulling from API
#         if raw_coin.index[-1] > pd.to_datetime(end):
#             newdf = raw_coin[raw_coin.index < pd.to_datetime(end)]
#             multiindex = pd.MultiIndex.from_product(
#                 [[name_from_path(paths[i])], list(newdf.index.values)],
#                 names=["Pair", "Time"],
#             )
#             preprocessed.append(newdf.set_index(multiindex))
#     # concat.groupby(level=0)['Price']=concat.groupby(level=0)['Price'].shift(0)-concat.groupby(level=0)['Price'][0]
#     # this step has to be done here even though it thematically fits end of prefilter since its not fully truncated by date and we would have to at least subtract the first row but whatever
#     # concat.groupby(level=0).apply(lambda x: x['Price']=x['logReturns'].cumsum())
#     return pd.concat(preprocessed)


def latexsave(df, file, params=[]):
    with open(file + ".tex", "w") as tf:
        tf.write(df.to_latex(*params, escape=False))

def load_results(name, methods, base="results"):
    path = os.path.join(base, name)
    files = os.listdir(path)
    dfs = []
    # for file in files:
    for i in tqdm(range(len(files)), desc='Loading saved results'):
        for file in files:
            rg = r"^" + str(len(dfs)) + "[a-z]"
            if methods in file and re.match(rg, file):

                df = pd.read_pickle(os.path.join(path, file))
                dfs.append(df)
                continue
    return pd.concat(dfs, keys=range(len(dfs)))




def find_same(r1, r2):
    """Finds overlap of pairs across methods """
    percentages = []
    for i in range(len(r1.index.unique(level=0))):
        same = r1.loc[(i), :].index.unique(level=0)[
            r1.loc[(i), :]
            .index.unique(level=0)
            .isin(r2.loc[(i), :].index.unique(level=0))
        ]
        percentage = len(same) / len(r1.loc[(i), :].index.unique(level=0))
        percentages.append(percentage)
    return pd.Series(percentages).mean()


def stoploss_results(
    newbase,
    methods=["dist"],
    freqs=["daily"],
    thresh=["1", "2", "3"],
    stoploss=["2", "3", "4", "5", "6"],
):
    res = {}
    global save
    for f in os.listdir(save):
        if (
            ("dist" in methods)
            and ("scenarios" in f)
            and (f[-2] in thresh)
            and (f[-1] in stoploss)
            and (f[-3] in [x[0] for x in freqs])
        ):
            res[f] = load_results(f, "dist", newbase)
        if (
            ("coint" in methods)
            and ("scenarios" in f)
            and (f[-2] in thresh)
            and (f[-1] in stoploss)
            and (f[-3] in [x[0] for x in freqs])
        ):
            res[f] = load_results(f, "coint", newbase)

    return res


def stoploss_preprocess(res, savename, savepath):
    des = {k: descriptive_frame(v) for k, v in res.items()}
    with open(savepath + savename + ".pkl", "wb") as handle:
        pickle.dump(des, handle, protocol=2)
    pass


def stoploss_streamed(
    savename,
    savepath,
    methods=["dist"],
    freqs=["daily"],
    thresh=["1", "2", "3"],
    stoploss=["2", "3", "4", "5", "6"],
):
    des = {}
    save = "C:\\Bach\\results\\"
    for f in os.listdir(save):
        if (
            ("dist" in methods)
            and ("scenarios" in f)
            and (f[-2] in thresh)
            and (f[-1] in stoploss)
            and (f[-3] in [x[0] for x in freqs])
        ):
            res = load_results(f, "dist")
            des[f] = res
            del res
        if (
            ("coint" in methods)
            and ("scenarios" in f)
            and (f[-2] in thresh)
            and (f[-1] in stoploss)
            and (f[-3] in [x[0] for x in freqs])
        ):
            res = load_results(f, "coint")
            des[f] = descriptive_frame(res)
            del res
    with open(savepath + savename + ".pkl", "wb") as handle:
        pickle.dump(des, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return des


def filter_nonsense(df):
    for col in df.columns.get_level_values(level=1):

        df.loc[(df.index.get_level_values(level=1) <= col), ("Threshold", col)] = "None"
    return df


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
