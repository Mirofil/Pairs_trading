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

    @staticmethod
    def prefilter(paths, start, end=end_date, volume_cutoff=0.7):
        """ Prefilters the time series so that we have only moderately old pairs (listed past start_date)
        and uses a volume percentile cutoff. The output is in array (pair, its volume) """
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
        return np.array(admissible)

    @staticmethod 
    def preprocess(paths, freq:str ="1D", end=end_date, first_n: int=0, start=start_date):
        """Finishes the preprocessing based on prefiltered paths. We filter out pairs that got delisted early
        (they need to go at least as far as end_date). Then all the eligible time series for pairs formation analysis
        are concated into one big DF with a multiIndex (pair, time).
        Params:
            first_n: Useful for smoketests; avoids taking the first n items"""

        paths = paths[first_n:]
        preprocessed = []
        for i in tqdm(range(len(paths)), desc="Preprocessing files"):
            stock_price = pd.read_csv(paths[i])
            stock_price = stock_price.sort_index()
            stock_price = resample(stock_price, freq, start=start)
            stock_price = stock_price.sort_index()
            # truncates the time series to a slightly earlier end date
            # because the last period is inhomogeneous due to pulling from API
            if stock_price.index[-1] > pd.to_datetime(end):
                newdf = stock_price[stock_price.index < pd.to_datetime(end)]
                multiindex = pd.MultiIndex.from_product(
                    [[name_from_path(paths[i])], list(newdf.index.values)],
                    names=["Pair", "Time"],
                )
                preprocessed.append(newdf.set_index(multiindex))
        all_time_series = pd.concat(preprocessed)
        return all_time_series
