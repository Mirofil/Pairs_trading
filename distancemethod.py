import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from config import *
from collections import namedtuple
from helpers import *

#%%
def distance(df, num=5):
    """ Df is expected to be a Multi-Indexed dataframe (result of helpers/preprocess)
    It returns: Distances (pairwise distance matrix),  """
    newdf = df.copy()
    # df has a MultiIndex of the form PAIR-DATE
    pairs = newdf.index.unique(0)
    dim = len(pairs)
    # gonna construct N*N matrix of pairwise distances
    distances = np.zeros((dim, dim))
    for pair in pairs:
        newdf.loc[pair, "logReturns"] = (
            np.log(newdf.loc[pair, "Close"]) - np.log(newdf.loc[pair, "Close"].shift(1))
        ).values
        newdf.loc[pair, "normReturns"] = (
            (newdf.loc[pair, "logReturns"] - newdf.loc[pair, "logReturns"].mean())
            / newdf.loc[pair, "logReturns"].std()
        ).values
        newdf.loc[pair, "normPrice"] = newdf.loc[pair, "normReturns"].cumsum().values
    # the distances matrix will be symmetric (think of covariance matrix)
    # pairwise SSD calculation
    for i in range(distances.shape[0]):
        for j in range(i, distances.shape[1]):
            distances[i, j] = np.sum(
                np.power(
                    newdf.loc[pairs[i], "normPrice"] - newdf.loc[pairs[j], "normPrice"],
                    2,
                )
            )
            distances[j, i] = distances[i, j]
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
    return (distances, top_indexes, viable_pairs, zipped, newdf)


def distance_spread(df, viable_pairs, timeframe, betas=1):
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

        # normPriceOld = reindexed.normPrice
        # reindexed.loc[:,'normPrice'] = (reindexed.loc[:,'normPrice']-reindexed.loc[:,'normPrice'].mean())/reindexed.loc[:,'normPrice'].std()
        # possible deletion of useless columns to save memory..
        # but maybe should be done earlier? Only normPrice
        # should be relevant since its the spread at this point
        # reindexed.drop(['Volume', 'Close', 'Returns'], axis = 1)
        # reindexed['normPriceOld'] = normPriceOld
        spreads.append(newdf)
    return pd.concat(spreads)


def adjust_weights(olddf, copy=True):
    """Manages the evolution of initial weights (through price changes)
    The percentage change is approximated by different in Price 
    (which is on log scale)"""
    df = olddf.copy(deep=copy)
    price_diff = (
        (
            1
            + (
                df.groupby(level=0)["1Price"].shift(0)
                - df.groupby(level=0)["1Price"].shift(1)
            )
        )
        .groupby(level=0)
        .cumprod()
    )
    df["1Weights"] = df["1Weights"].groupby(level=0).shift(1) * price_diff.groupby(
        level=0
    ).shift(0)
    price_diff2 = (
        (
            1
            + (
                df.groupby(level=0)["2Price"].shift(0)
                - df.groupby(level=0)["2Price"].shift(1)
            )
        )
        .groupby(level=0)
        .cumprod()
    )
    df["2Weights"] = df["2Weights"].groupby(level=0).shift(1) * price_diff2.groupby(
        level=0
    ).shift(0)
    return df


def distance_propagate_weights(df, timeframe, debug=False):
    for name, group in df.groupby(level=0):
        end_of_formation = df.loc[name].index.get_loc(timeframe[1])
        temp_weights1 = group["1Weights"].to_list()
        temp_weights2 = group["2Weights"].to_list()
        return1 = group["1Price"] - group["1Price"].shift(1)
        return2 = group["2Price"] - group["2Price"].shift(1)
        if debug == True:
            print(end_of_formation, len(group.index), name)
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


def distance_profit(df, beta=1.5):
    """ Calculates the per-period profits
    Returns a Multi-Indexed one column DF with Profit"""
    # there used to be df as first arg in the func definition but its probably worthless?
    # first = ((df.groupby(level=0)['1Price'].shift(0)-df.groupby(level=0)['1Price'].shift(1))*df.groupby(level=0)['1Price'].apply(np.sign)).groupby(level=0).cumsum()
    # second = ((df.groupby(level=0)['2Price'].shift(0)-df.groupby(level=0)['2Price'].shift(1))*df.groupby(level=0)['2Price'].apply(np.sign)).groupby(level=0).cumsum()*beta
    pure_1return = df.groupby(level=0)["1Price"].shift(0) - df.groupby(level=0)[
        "1Price"
    ].shift(1)
    # Might have to drop the apply(sign) if I use signed weights
    signed_1return = pure_1return * df.groupby(level=0)["Spread"].apply(np.sign)
    weighted_1return = signed_1return * (df.groupby(level=0)["1Weights"].apply(np.abs))
    pure_2return = df.groupby(level=0)["2Price"].shift(0) - df.groupby(level=0)[
        "2Price"
    ].shift(1)
    signed_2return = pure_2return * (df.groupby(level=0)["Spread"].apply(np.sign) * -1)
    weighted_2return = signed_2return * (df.groupby(level=0)["2Weights"].apply(np.abs))

    return weighted_1return + weighted_2return
