import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from pairs.pairs_trading_engine import sliced_norm
from tqdm import tqdm
import sklearn.metrics
import re

# new = []
# x=newdf['normPrice'].reset_index(level=1)
# for stock in newdf['normPrice'].index.unique(0):
#     new.append(pd.DataFrame(pd.Series(x.loc[stock, 'normPrice'].values, index = x.loc[stock, 'Time'].values)).T)
# interim = pd.concat(new)
# interim.index = newdf['normPrice'].index.unique(0)
# np.power(sklearn.metrics.pairwise_distances(interim.drop(interim.columns[[0]], axis=1)), 2)

def distance(df: pd.DataFrame, num:int =5, method='modern'):
    """
    Args:
        df (pd.DataFrame): Df is expected to be a Multi-Indexed dataframe (result of helpers/preprocess)
        num (int, optional): How many shortest-distance pairs to take. Defaults to 5.

    Returns:
        Distances (pairwise distance matrix) as third items
        then some trash?
    """
    newdf = df.copy()
    # df has a MultiIndex of the form PAIR-DATE
    pairs = newdf.index.unique(0)
    dim = len(pairs)
    # gonna construct N*N matrix of pairwise distances
    for pair in tqdm(pairs, desc = 'Calculating price statistics across pairs'):
        newdf.loc[pair, "logReturns"] = (
            np.log(newdf.loc[pair, "Close"]) - np.log(newdf.loc[pair, "Close"].shift(1))
        ).values
        newdf.loc[pair, "normReturns"] = (
            (newdf.loc[pair, "logReturns"] - newdf.loc[pair, "logReturns"].mean())
            / newdf.loc[pair, "logReturns"].std()
        ).values
        newdf.loc[pair, "normPrice"] = newdf.loc[pair, "normReturns"].cumsum().values
    
    newdf = newdf.astype(np.float32)
    if method == 'oldschool':
        distances = np.zeros((dim, dim), dtype=np.float32)

        # the distances matrix will be symmetric (think of covariance matrix)
        # pairwise SSD calculation
        for i in tqdm(range(dim), desc= 'Going across X axis of distance matrix'):
            for j in range(i, dim):
                distances[i, j] = np.sum(
                    np.power(
                        newdf.loc[pairs[i], "normPrice"] - newdf.loc[pairs[j], "normPrice"],
                        2
                    )
                )
                distances[j, i] = distances[i, j]
    elif method == 'modern':
        new = []
        x=newdf['normPrice'].reset_index(level=1)
        for stock in newdf['normPrice'].index.unique(0):
            new.append(pd.DataFrame(pd.Series(x.loc[stock, 'normPrice'].values, index = x.loc[stock, 'Time'].values)).T)
        interim = pd.concat(new)
        interim.index = newdf['normPrice'].index.unique(0)
        distances = np.power(sklearn.metrics.pairwise_distances(interim.drop(interim.columns[[0]], axis=1)), 2)

    # we use the distance matrix as upper triangular to avoid duplicates in sorting
    triang = np.triu(distances)
    sorted_array = np.argsort(triang, axis=None)
    original_index = np.unravel_index(sorted_array, distances.shape)
    # index of first nonzero element in the sorted upper triangular array
    nonzero_index = np.nonzero(triang[original_index])[0][0]
    # we will offset from this to unravel the smallest positive SSD indexes
    top_indexes = np.unravel_index(
        sorted_array[nonzero_index : nonzero_index + num], distances.shape
    )
    # a different form of the indexes in top_indexes - returns list of coordinate pairs that describe
    # a single pair rather than two arrays where X and Y coordinates of a single pair are split among those
    zipped = np.array(list(zip(top_indexes[0], top_indexes[1])))
    viable_pairs = [(pairs[x[0]], pairs[x[1]]) for x in zipped]
    return OrderedDict({'distances':distances, 'top_indexes':top_indexes, 'viable_pairs': viable_pairs, 'zipped': zipped, 'newdf':newdf})


def distance_spread(df, viable_pairs, timeframe, betas=None):
    """Picks out the viable pairs of the original df (which has all pairs)
    and adds to it the normPrice Spread among others, as well as initially
    defines Weights and Profit """
    idx = pd.IndexSlice
    spreads = []
    if betas == None:
        betas = [np.array([1, 1]) for i in range(len(viable_pairs))]
    for pair, coefs in zip(viable_pairs, betas):
        # labels will be IOTAADA rather that IOTABTCADABTC,
        # so we remove the last three characters
        first = re.sub(r'USDT$|USD$|BTC$','', pair[0])
        second = re.sub(r'USDT$|USD$|BTC$','', pair[1])
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

