import datetime
import glob
import os
import pickle
import timeit
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from dateutil.relativedelta import relativedelta
from p_tqdm import p_map
from ray import tune
from tqdm import tqdm

from pairs.analysis import (aggregate, descriptive_frame, descriptive_stats,
                            infer_periods)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import beautify, standardize_results
from pairs.helpers import *
from pairs.helpers import latexsave
from pairs.scripts.latex.helpers import *
from pairs.pairs_trading_engine import pick_range, backtests_up_to_date
