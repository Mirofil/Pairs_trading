#%%
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import itertools
import timeit
import multiprocess as mp
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from pairs.stattools_alt import coint
import re
try:
    import CyURT as urt
except:
    pass


def find_integrated_fast(df, confidence=0.05, trend=b"c", regression=False, show_progress_bar=True):
    """Uses ADF test to decide I(1) as in the first step of AEG test. 
    Takes the data from preprocess and filters out stationary series,
    returning data in the same format. 
    DF Test has unit root as null"""
    pairs = df.index.unique(0)
    integrated = []
    df["logClose"] = np.log(df["Close"].values)
    for pair in tqdm(pairs, desc='Finding integrated time series', disable= not show_progress_bar):
        df.loc[pair, "logReturns"] = (
            np.log(df.loc[pair, "Close"]) - np.log(df.loc[pair, "Close"].shift(1))
        ).values
        pvalue = urt.ADF_d(
            df.loc[pair, "logClose"].fillna(method="ffill").values,
            trend=trend,
            regression=regression
        )
        pvalue.show()
        pvalue = pvalue.pval
        if pvalue >= confidence:
            integrated.append(pair)

    return df.loc[integrated]

def find_integrated(df, confidence=0.05, regression="c", num_of_processes=1, show_progress_bar = True):
    """Uses ADF test to decide I(1) as in the first step of AEG test. 
    Takes the data from preprocess and filters out stationary series,
    returning data in the same format. 
    DF Test has unit root as null"""
    if num_of_processes > 1:
        pairs = df.index.unique(0)
        integrated = []
        split = np.array_split(pairs, num_of_processes)
        pool = mp.Pool(num_of_processes)

        def worker(pairs):
            integrated = []
            import statsmodels.tsa.stattools as ts
            import numpy as np

            for pair in pairs:
                df.loc[pair, "logReturns"] = (
                    np.log(df.loc[pair, "Close"])
                    - np.log(df.loc[pair, "Close"].shift(1))
                ).values
                df.loc[pair, "logClose"] = np.log(df.loc[pair, "Close"].values)
                pvalue = ts.adfuller(
                    df.loc[pair, "logClose"].fillna(method="ffill").values,
                    regression=regression,
                )[1]
                if pvalue >= confidence:
                    integrated.append(pair)
            return integrated

        result = pool.map(worker, split)
        pool.close()
        pool.join()
        flat_result = [item for sublist in result for item in sublist]
        return df.loc[flat_result]
    else:
        pairs = df.index.unique(0)
        integrated = []
        for pair in tqdm(pairs, desc='Finding integrated time series', disable = not show_progress_bar):
            df.loc[pair, "logClose"] = np.log(df.loc[pair, "Close"].values)
            df.loc[pair, "logReturns"] = (
                df.loc[pair, "logClose"] - df.loc[pair, "logClose"].shift(1)
            ).values
            pvalue = ts.adfuller(
                df.loc[pair, "logClose"].fillna(method="ffill").values,
                regression=regression,
            )[1]
            if pvalue >= confidence:
                integrated.append(pair)
        return df.loc[integrated]

def cointegration_mixed(df_integrated_pairs, viable_pairs, desired_num=20, confidence=0.05, show_progress_bar=True):
    """Computationally efficient cointegration method by mixing dist method with coint
    Args:
        df_integrated_pairs ([type]): DF where the stocks are all integrated
        viable_pairs ([type]): Should come from dist method
    """
    integrated_pairs = df_integrated_pairs.index.unique(0)
    cointegrated = []

    for pair in tqdm(viable_pairs, desc ='Finding cointegrations across pairs', disable= not show_progress_bar):
        if pair[0] not in integrated_pairs or pair[1] not in integrated_pairs:
            continue

        x = df_integrated_pairs.loc[pair[0], "logClose"].fillna(method="ffill").values
        x = x.reshape((x.shape[0], 1))
        y = df_integrated_pairs.loc[pair[1], "logClose"].fillna(method="ffill").values
        y = y.reshape((y.shape[0], 1))
        if ts.coint(x, y)[1] <= confidence:
            model = sm.OLS(y, sm.add_constant(x))
            results = model.fit()
            # the model is like "second(logClose) - coef*first(logClose) = mean(logClose)+epsilon" in the pair
            cointegrated.append([pair, results.params])
        
        if len(cointegrated) >= desired_num:
            break
    
    return cointegrated


def cointegration_fast(df, confidence=0.05, show_progress_bar = True):
    pairs = df.index.unique(0)
    cointegrated = []
    df["logClose"] = df["logClose"].fillna(method="ffill")
    df["logClose"].fillna(method="ffill").values
    for pair in tqdm(itertools.combinations(pairs, 2), total=len(pairs)*(len(pairs)-1)/2, desc ='Finding cointegrations across pairs', disable= not show_progress_bar):
        x = df.loc[pair[0], "logClose"].values
        y = df.loc[pair[1], "logClose"].values
        if coint(x, y)[1] <= confidence:
            # model = sm.OLS(y, sm.add_constant(x))
            # results = model.fit()
            fit = urt.OLS_d(y, sm.add_constant(x), True)
            # the model is like "second(logClose) - coef*first(logClose) = mean(logClose)+epsilon" in the pair
            cointegrated.append([pair, fit.param])
    return cointegrated

def cointegration(df, confidence=0.05, num_of_processes=1, show_progress_bar = True):
    if num_of_processes > 1:
        pairs = df.index.unique(0)
        cointegrated = []
        split = np.array_split(list(itertools.combinations(pairs, 2)), 3)
        pool = mp.Pool(num_of_processes)

        def worker(pairs, confidence=0.05):
            nonlocal df
            import statsmodels.api as sm
            import statsmodels.tsa.stattools as ts

            cointegrated = []
            for pair in pairs:
                x = df.loc[pair[0], "logClose"].fillna(method="ffill").values
                x = x.reshape((x.shape[0], 1))
                y = df.loc[pair[1], "logClose"].fillna(method="ffill").values
                y = y.reshape((y.shape[0], 1))
                if ts.coint(x, y)[1] <= confidence:
                    model = sm.OLS(y, x)
                    results = model.fit()
                    # the model is like "second(logClose) - coef*first(logClose) = mean(logClose)+epsilon" in the pair
                    cointegrated.append([pair, results.params])
            return cointegrated

        result = pool.map(worker, split)
        pool.close()
        pool.join()
        flat_result = [item for sublist in result for item in sublist]
        flat_result = [[tuple(item[0]), item[1]] for item in flat_result]
        return flat_result
    else:
        pairs = df.index.unique(0)
        cointegrated = []
        for pair in tqdm(itertools.combinations(pairs, 2), total=len(pairs)*(len(pairs)-1)/2, desc ='Finding cointegrations across pairs', disable= not show_progress_bar):
            x = df.loc[pair[0], "logClose"].fillna(method="ffill").values
            x = x.reshape((x.shape[0], 1))
            y = df.loc[pair[1], "logClose"].fillna(method="ffill").values
            y = y.reshape((y.shape[0], 1))
            if ts.coint(x, y)[1] <= confidence:
                model = sm.OLS(y, sm.add_constant(x))
                results = model.fit()
                # the model is like "second(logClose) - coef*first(logClose) = mean(logClose)+epsilon" in the pair
                cointegrated.append([pair, results.params])
    return cointegrated
