import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

from pairs.datasets.base import Dataset


class USDataset(Dataset):
    def __init__(self, config=None):
        files = os.listdir(config["data_path"])
        self.paths = [os.path.join(config["data_path"], x) for x in files]
        self.config = config
        super().__init__()