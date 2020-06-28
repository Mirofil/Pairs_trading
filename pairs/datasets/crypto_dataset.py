import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

from pairs.config import TradingUniverse
from pairs.datasets.base import Dataset

class CryptoDataset (Dataset):
    def __init__(self, config = TradingUniverse()):
        super().__init__()
        files = os.listdir(config["data_path"])
        self.paths = [os.path.join(config["data_path"], x) for x in files if x not in ['BTCUSDT.csv', 'ETHUSDT.csv', 'CLOAKBTC.csv']]
        self.config = config
