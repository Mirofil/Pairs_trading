import datetime
import os
import re
from collections import OrderedDict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm

from pairs.pairs_trading_engine import sliced_norm

def distance(df: pd.DataFrame, num: int = 5, method="modern", show_progress_bar=True):
    """
    Args:
        df (pd.DataFrame): Df is expected to be a Multi-Indexed dataframe (result of helpers/preprocess)
        num (int, optional): How many shortest-distance pairs to take. Defaults to 5.

    Returns:
        Distances (pairwise distance matrix) as third items
        then some trash?
    """
    df = df.copy()
    # df has a MultiIndex of the form PAIR-DATE
    pairs = df.index.unique(0)
    dim = len(pairs)
    # gonna construct N*N matrix of pairwise distances
    for pair in tqdm(
        pairs,
        desc="Calculating price statistics across pairs",
        disable=not show_progress_bar,
    ):
        df.loc[pair, "logReturns"] = (
            np.log(df.loc[pair, "Close"]) - np.log(df.loc[pair, "Close"].shift(1))
        ).values
        df.loc[pair, "normReturns"] = (
            (df.loc[pair, "logReturns"] - df.loc[pair, "logReturns"].mean())
            / df.loc[pair, "logReturns"].std()
        ).values
        df.loc[pair, "normPrice"] = df.loc[pair, "normReturns"].cumsum().values

    df = df.astype(np.float32)
    if method == "oldschool":
        distances = np.zeros((dim, dim), dtype=np.float32)

        # the distances matrix will be symmetric (think of covariance matrix)
        # pairwise SSD calculation
        for i in tqdm(range(dim), desc="Going across X axis of distance matrix"):
            for j in range(i, dim):
                distances[i, j] = np.sum(
                    np.power(
                        df.loc[pairs[i], "normPrice"] - df.loc[pairs[j], "normPrice"], 2
                    )
                )
                distances[j, i] = distances[i, j]
    elif method == "modern":
        new = []
        x = df["normPrice"].reset_index(level=1)
        for stock in df["normPrice"].index.unique(0):
            new.append(
                pd.DataFrame(
                    pd.Series(
                        x.loc[stock, "normPrice"].values,
                        index=x.loc[stock, "Time"].values,
                    )
                ).T
            )
        interim = pd.concat(new)
        interim.index = df["normPrice"].index.unique(0)
        distances = np.power(
            sklearn.metrics.pairwise_distances(
                interim.drop(interim.columns[[0]], axis=1)
            ),
            2,
        )

    # the SKLEARN fast method seems to have some rounding errors, so the diagonal of the distance matrix might not be always 0
    eps = 0.0000001
    distances[np.abs(distances) < eps] = 0
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
    return OrderedDict(
        {
            "distances": distances,
            "top_indexes": top_indexes,
            "viable_pairs": viable_pairs,
            "zipped": zipped,
            "newdf": df,
        }
    )
