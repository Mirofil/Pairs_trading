import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from pairs.config import (TradingUniverse)
from pairs.helpers import name_from_path, resample


class CryptoDataset:
    def __init__(self, config = TradingUniverse()):
        files = os.listdir(config["data_path"])
        self.paths = [os.path.join(config["data_path"], x) for x in files if x not in ['BTCUSDT.csv', 'ETHUSDT.csv', 'CLOAKBTC.csv']]
        self.config = config

    
    def prefilter(self, paths=None, start=None, end=None, volume_cutoff=None):
        """ Prefilters the time series so that we have only moderately old pairs (listed past start_date)
        and uses a volume percentile cutoff. The output is in array (pair, its volume) """
        if paths is None:
            paths = self.paths

        if start is None:
            start = self.config["start_date"]
        
        if end is None:
            end = self.config["end_date"]
        
        if volume_cutoff is None:
            volume_cutoff = self.config["volume_cutoff"]

        idx = pd.IndexSlice
        admissible = []
        for i in tqdm(
            range(len(paths)),
            desc="Prefiltering pairs (based on volume and start/end of trading)",
        ):
            df = pd.read_csv(paths[i])
            df.rename({"Opened": "Date"}, axis="columns", inplace=True, errors='ignore')
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
                            * df.loc[idx[str(start) : str(end)], "Close"]
                        ).sum(),
                    ]
                )
        # sort by Volume and pick upper percentile
        admissible.sort(key=lambda x: x[1])
        admissible = admissible[int(np.round(len(admissible) * volume_cutoff)) :]
        return np.array(admissible)
    
    def preprocess(self, paths=None, freq:str = None, end=None, first_n: int=None, start=None):
        """Finishes the preprocessing based on prefiltered paths. We filter out pairs that got delisted early
        (they need to go at least as far as end_date). Then all the eligible time series for pairs formation analysis
        are concated into one big DF with a multiIndex (pair, time)."""
        if paths is None:
            paths = self.paths

        if start is None:
            start = self.config["start_date"]
        
        if end is None:
            end = self.config["end_date"]
        
        if freq is None:
            freq = self.config["freq"]

        
        if first_n is not None:
            paths = paths[:first_n]
        preprocessed = []
        for i in tqdm(range(len(paths)), desc="Preprocessing files"):
            raw_coin = pd.read_csv(paths[i])
            # The new Binance_fetcher API downloads Date as Opened instead..
            raw_coin.rename({"Opened": "Date"}, axis="columns", inplace=True)
            raw_coin = raw_coin.sort_index()
            raw_coin = resample(raw_coin, freq, start=start)
            raw_coin = raw_coin.sort_index()
            # truncates the time series to a slightly earlier end date
            # because the last period is inhomogeneous due to pulling from API
            if raw_coin.index[-1] > pd.to_datetime(end):
                newdf = raw_coin[raw_coin.index < pd.to_datetime(end)]
                multiindex = pd.MultiIndex.from_product(
                    [[name_from_path(paths[i])], list(newdf.index.values)],
                    names=["Pair", "Time"],
                )
                preprocessed.append(newdf.set_index(multiindex))
        all_time_series = pd.concat(preprocessed)
        # concat.groupby(level=0)['Price']=concat.groupby(level=0)['Price'].shift(0)-concat.groupby(level=0)['Price'][0]
        # this step has to be done here even though it thematically fits end of prefilter since its not fully truncated by date and we would have to at least subtract the first row but whatever
        # concat.groupby(level=0).apply(lambda x: x['Price']=x['logReturns'].cumsum())
        return all_time_series
