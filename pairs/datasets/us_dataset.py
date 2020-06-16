import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from pairs.config import (NUMOFPROCESSES, data_path, end_date, save,
                          start_date, version, TradingUniverse)
from pairs.helpers import name_from_path, resample


class USDataset:
    def __init__(self, config = TradingUniverse()):
        files = os.listdir(config["data_path"])
        self.paths = [os.path.join(config["data_path"], x) for x in files]
        self.config = config
        super().__init__()

    def prefilter(self):
        """ Prefilters the time series so that we have only moderately old pairs (listed past start_date)
        and uses a volume percentile cutoff. The output is in array (pair, its volume) """
        paths = self.paths
        start = self.config["start_date"]
        end = self.config["end_date"]
        volume_cutoff = self.config["volume_cutoff"]
        
        idx = pd.IndexSlice
        admissible = []
        for i in tqdm(
            range(len(paths)),
            desc="Prefiltering pairs (based on volume and start/end of trading)",
        ):
            df = pd.read_csv(paths[i])
            # filters out pairs that got listed past start_date
            if (pd.to_datetime(df.iloc[0]["Date"]) < pd.to_datetime(start)) and (
                pd.to_datetime(df.iloc[-1]["Date"]) > pd.to_datetime(end)
            ):
                # the Volume gets normalized to BTC before sorting
                df = df.set_index("Date")
                df = df.sort_index()
                admissible.append(
                    [
                        paths[i],
                        (
                            df.loc[idx[str(start) : str(end)], "Volume"]
                        ).sum(),
                    ]
                )
        # sort by Volume and pick upper percentile
        admissible.sort(key=lambda x: x[1])
        admissible = admissible[int(np.round(len(admissible) * volume_cutoff)) :]

        result = np.array(admissible)
        self.prefiltered_paths = result

        return result
 
    def preprocess(self, first_n: int=0):
        """Finishes the preprocessing based on prefiltered paths. We filter out pairs that got delisted early
        (they need to go at least as far as end_date). Then all the eligible time series for pairs formation analysis
        are concated into one big DF with a multiIndex (pair, time).
        Params:
            first_n: Useful for smoketests; avoids taking the first n items"""
        prefiltered_paths = self.prefiltered_paths
        freq = self.config["freq"]
        end_date = self.config["end_date"]
        start_date = self.config["start_date"]

        prefiltered_paths = prefiltered_paths[first_n:][:,0]
        preprocessed = []
        for i in tqdm(range(len(prefiltered_paths)), desc="Preprocessing files"):
            stock_price = pd.read_csv(prefiltered_paths[i])
            stock_price = stock_price.sort_index()
            stock_price = resample(stock_price, freq, start=start_date)
            stock_price = stock_price.sort_index()
            # truncates the time series to a slightly earlier end date
            # because the last period is inhomogeneous due to pulling from API
            if stock_price.index[-1] > pd.to_datetime(end_date):
                newdf = stock_price[stock_price.index < pd.to_datetime(end_date)]
                multiindex = pd.MultiIndex.from_product(
                    [[name_from_path(prefiltered_paths[i])], list(newdf.index.values)],
                    names=["Pair", "Time"],
                )
                preprocessed.append(newdf.set_index(multiindex))
        all_time_series = pd.concat(preprocessed)
        self.preprocessed_paths = all_time_series
        return all_time_series
