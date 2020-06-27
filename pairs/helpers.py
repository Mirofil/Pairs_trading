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


def remake_into_lists(*args):
    result=[]
    for arg in args:
        if not isinstance(arg, list):
            result.append([arg])
        else:
            result.append(arg)
    return result

def name_from_path(path: str):
    """ Goes from stuff like C:\Bach\concat_data\[pair].csv to [pair]"""
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def path_from_name(name: str, data_path=data_path):
    """ Goes from stuff like [pair] to C:\Bach\concat_data\[pair].csv"""
    path = os.path.join(data_path, name + ".csv")
    return name

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
