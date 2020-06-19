import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

from pairs.config import (
    NUMOFPROCESSES,
    data_path,
    end_date,
    save,
    start_date,
    version,
    TradingUniverse,
)
from pairs.helpers import name_from_path, resample
from abc import ABC

class Dataset(ABC):
    def __init__(self):
        pass

    def preprocess(self, freq=None, start_date=None, end_date=None, first_n: int = 0, show_progress_bar = False):
        """Finishes the preprocessing based on prefiltered paths. We filter out pairs that got delisted early
        (they need to go at least as far as end_date). Then all the eligible time series for pairs formation analysis
        are concated into one big DF with a multiIndex (pair, time).
        Params:
            first_n: Useful for smoketests; avoids taking the first n items"""
        prefiltered_paths = self.prefiltered_paths
        freq = self.config["freq"]
        if end_date is None:
            end_date = self.config["end_date"]
        if start_date is None:
            start_date = self.config["start_date"]

        prefiltered_paths = prefiltered_paths[first_n:]["0"]
        preprocessed = []
        for i in tqdm(range(len(prefiltered_paths)), desc="Preprocessing files", disable = not show_progress_bar):
            stock_price = pd.read_csv(prefiltered_paths[i])
            stock_price.rename({"Opened": "Date"}, axis="columns", inplace=True, errors='ignore')
            stock_price = stock_price.sort_index()
            stock_price = resample(stock_price, freq=None, start=start_date)
            stock_price = stock_price.sort_index()
            # truncates the time series to a slightly earlier end date
            # because the last period is inhomogeneous due to pulling from API
            if stock_price.index[-1] > pd.to_datetime(end_date):
                newdf = stock_price[stock_price.index < pd.to_datetime(end_date)]
                multiindex = pd.MultiIndex.from_product(
                    [[name_from_path(prefiltered_paths[i])], list(newdf.index.values)],
                    names=["Pair", "Time"],
                )
                newdf = newdf.drop(labels="Date", axis=1, errors='ignore')
                preprocessed.append(newdf.set_index(multiindex))
        all_time_series = pd.concat(preprocessed)

        self.preprocessed_paths = all_time_series
        return all_time_series


    def prefilter(self, paths=None, start_date=None, end_date=None, drop_any_na=True, show_progress_bar=False):
        """ Prefilters the time series so that we have only moderately old pairs (listed past start_date_date)
        and uses a volume percentile cutoff. The output is in array (pair, its volume) """
        if paths is None:
            paths = self.paths

        if start_date is None:
            start_date = self.config["start_date"]

        if end_date is None:
            end_date = self.config["end_date"]

        if isinstance(start_date, list):
            start_date = datetime.date(*start_date)
        if isinstance(end_date, list):
            end_date = datetime.date(*end_date)

        volume_cutoff = self.config["volume_cutoff"]

        idx = pd.IndexSlice
        admissible = []
        for i in tqdm(
            range(len(paths)),
            desc="Prefiltering pairs (based on volume and start_date/end_date of trading)",
            disable = not show_progress_bar
        ):

            df = pd.read_csv(paths[i])
            if drop_any_na is True:
                if len(df.dropna()) < len(df):
                    continue
            df.rename({"Opened": "Date"}, axis="columns", inplace=True, errors='ignore')

            # filters out pairs that got listed past start_date_date
            if (pd.to_datetime(df.iloc[0]["Date"]) < pd.to_datetime(start_date)) and (
                pd.to_datetime(df.iloc[-1]["Date"]) > pd.to_datetime(end_date)
            ):
                # the Volume gets normalized to BTC before sorting
                df = df.set_index("Date")
                df = df.sort_index()
                admissible.append(
                    [
                        paths[i],
                        (
                            df.loc[idx[str(start_date) : str(end_date)], "Volume"]
                            * df.loc[idx[str(start_date) : str(end_date)], "Close"]
                        ).sum(),
                    ]
                )
        # sort by Volume and pick upper percentile
        admissible.sort(key=lambda x: x[1])
        admissible = admissible[
            int(np.round(len(admissible) * volume_cutoff[0])) : int(
                np.round(len(admissible) * volume_cutoff[1])
            )
        ]

        result = np.array(admissible)

        result = pd.DataFrame(np.array(admissible), columns=["0", "1"])
        result.columns = [str(col) for col in result.columns]

        self.prefiltered_paths = result
        return result