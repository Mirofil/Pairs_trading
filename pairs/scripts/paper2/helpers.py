import datetime
import glob
import os
import timeit
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from tqdm import tqdm

from pairs.analysis import (aggregate, descriptive_frame, descriptive_stats,
                            drawdown, infer_periods)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import standardize_results, beautify
from pairs.helpers import *


def nya_stats(nya: pd.DataFrame, feasible, riskfree=0.02):
    """
    Args:
        btc ([type]): BTC time series with cumProfit as cumulative profit. Should be DAILY frequency!
    """
    profit = nya.iloc[-1]["cumProfit"] - nya.iloc[0]["cumProfit"]
    num_of_trading_days = (nya.index[-1] - nya.index[0]).days
    num_of_trading_months = num_of_trading_days / 30
    monthly_profit = (1 + profit) ** (1 / num_of_trading_months)-1
    max_drawdown = abs(drawdown(nya).min())
    annualized_sd = profit ** ((num_of_trading_days / 360) ** 1 / 2)
    annualized_sharpe = (
        (1 + profit) ** (1 / (num_of_trading_days / 360)) - 1 - riskfree
    ) / annualized_sd

    result= pd.DataFrame(
        [
            monthly_profit,
            monthly_profit ** 12,
            monthly_profit,
            annualized_sharpe,
            None,
            None,
            None,
            None,
            None,
            max_drawdown,
            None,
            None,
        ],
        index=feasible,
    )
    result.columns = pd.MultiIndex.from_product([['Market'], ['NYA']])
    result = beautify(result)
    result = result.drop(["Annual profit", "Trading period profit", "Trading period Sharpe"])
    result = result.rename({"Number of trades":"Monthly number of trades", "Avg length of position":"Length of position (days)"})
    result = result.replace('nan', 'None').replace('nan\%', 'None')

    return result
