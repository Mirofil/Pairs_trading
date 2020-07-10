import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import timeit
import datetime
from dateutil.relativedelta import relativedelta
from pairs.distancemethod import *
from pairs.helpers import *
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.analysis import (
    descriptive_frame,
    descriptive_stats,
    infer_periods,
    aggregate,
)
from pairs.formatting import standardize_results
import glob
from tqdm import tqdm
import pickle
from pairs.formatting import standardize_results, beautify
from pairs.helpers import latexsave
from pairs.scripts.latex.helpers import *
from pairs.formatting import beautify, standardize_results


