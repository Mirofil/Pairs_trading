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
from pairs.datasets.base import Dataset


class USDataset(Dataset):
    def __init__(self, config=TradingUniverse()):
        files = os.listdir(config["data_path"])
        self.paths = [os.path.join(config["data_path"], x) for x in files]
        self.config = config
        super().__init__()