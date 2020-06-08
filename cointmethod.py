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
from helpers import *

#%%
def find_integrated(df, confidence=0.05, regression="c", num_of_processes=1):
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
        for pair in pairs:
            df.loc[pair, "logReturns"] = (
                np.log(df.loc[pair, "Close"]) - np.log(df.loc[pair, "Close"].shift(1))
            ).values
            df.loc[pair, "logClose"] = np.log(df.loc[pair, "Close"].values)
            pvalue = ts.adfuller(
                df.loc[pair, "logClose"].fillna(method="ffill").values,
                regression=regression,
            )[1]
            if pvalue >= confidence:
                integrated.append(pair)
        return df.loc[integrated]


def cointegration(df, confidence=0.05, num_of_processes=1):
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
                    model = sm.OLS(y, sm.add_constant(x))
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
        for pair in itertools.combinations(pairs, 2):
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


def coint_spread(df, viable_pairs, timeframe, betas=1):
    """Picks out the viable pairs of the original df (which has all pairs)
    and adds to it the normPrice Spread among others, as well as initially
    defines Weights and Profit """
    idx = pd.IndexSlice
    spreads = []
    if betas == 1:
        betas = [np.array([1, 1]) for i in range(len(viable_pairs))]
    for pair, coefs in zip(viable_pairs, betas):
        # labels will be IOTAADA rather that IOTABTCADABTC,
        # so we remove the last three characters
        first = pair[0][:-3]
        second = pair[1][:-3]
        composed = first + "x" + second
        multiindex = pd.MultiIndex.from_product(
            [[composed], df.loc[pair[0]].index], names=["Pair", "Time"]
        )
        newdf = pd.DataFrame(index=multiindex)
        newdf["1Weights"] = None
        newdf["2Weights"] = None
        newdf["Profit"] = 0
        # newdf['normLogReturns']= sliced_norm (df, pair, 'logReturns', timeframe)
        newdf["1Price"] = df.loc[pair[0], "Price"].values
        newdf["2Price"] = df.loc[pair[1], "Price"].values
        newdf["1logClose"] = df.loc[pair[0], "logClose"].values
        newdf["2logClose"] = df.loc[pair[1], "logClose"].values
        newdf["Spread"] = newdf["2logClose"] - newdf["1logClose"] * coefs[1]
        newdf["SpreadBeta"] = coefs[1]
        newdf["normSpread"] = (
            (
                newdf["Spread"]
                - newdf.loc[idx[composed, timeframe[0] : timeframe[1]], "Spread"].mean()
            )
            / newdf.loc[idx[composed, timeframe[0] : timeframe[1]], "Spread"].std()
        ).values
        # pick_range(newdf, *timeframe)['Spread'].mean()
        # not sure what those lines do
        first = df.loc[pair[0]]
        first.columns = ["1" + x for x in first.columns]
        second = df.loc[pair[0]]
        second.columns = ["2" + x for x in second.columns]
        reindexed = (pd.concat([first, second], axis=1)).set_index(multiindex)

        # normPriceOld = reindexed.normPrice
        # reindexed.loc[:,'normPrice'] = (reindexed.loc[:,'normPrice']-reindexed.loc[:,'normPrice'].mean())/reindexed.loc[:,'normPrice'].std()
        # possible deletion of useless columns to save memory..
        # but maybe should be done earlier? Only normPrice
        # should be relevant since its the spread at this point
        # reindexed.drop(['Volume', 'Close', 'Returns'], axis = 1)
        # reindexed['normPriceOld'] = normPriceOld
        spreads.append(newdf)
    return pd.concat(spreads)


#%%
