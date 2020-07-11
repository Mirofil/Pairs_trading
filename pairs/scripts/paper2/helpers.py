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


def ts_stats(ts: pd.DataFrame, feasible=None, riskfree=0.02):
    """
    Args:
        btc ([type]): BTC time series with cumProfit as cumulative profit. Should be DAILY frequency!
    """
    if feasible is None:
        feasible = [
            "Monthly profit",
            "Annual profit",
            "Total profit",
            "Annualized Sharpe",
            "Trading period Sharpe",
            "Number of trades",
            "Roundtrip trades",
            "Avg length of position",
            "Pct of winning trades",
            "Max drawdown",
            "Nominated pairs",
            "Traded pairs",
        ]
    profit = ts.iloc[-1]["cumProfit"] - ts.iloc[0]["cumProfit"]
    num_of_trading_days = (ts.index[-1] - ts.index[0]).days
    # num_of_trading_days = len(ts.index)
    num_of_trading_months = num_of_trading_days / 30
    monthly_profit = (1 + profit) ** (1 / num_of_trading_months)-1
    max_drawdown = abs(drawdown(ts).min())
    annualized_sd = ts['cumProfit'].std() ** ((num_of_trading_days / 360) ** 1 / 2)
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

def nya_stats(start_date:str, end_date:str, nya_path:pd.DataFrame = '/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/hist/NYA.csv'):
    nya = pd.read_csv(nya_path)
    nya = nya.set_index('Date')
    nya.index = pd.to_datetime(nya.index)
    if type(end_date) is str:
        end_date = pd.to_datetime(end_date)
    if type(start_date) is str:
        start_date = pd.to_datetime(start_date)

    nya = nya.loc[start_date:end_date]
    nya['Close'] = nya['Close']/nya["Close"].iloc[0]
    nya["cumProfit"] = nya["Close"]

    return ts_stats(nya)