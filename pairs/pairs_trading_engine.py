import datetime
import os
import pickle
import re
import shutil
from contextlib import contextmanager
from typing import *

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
import scipy
import statsmodels
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from pairs.config import data_path
from pairs.analysis import infer_periods
from dateutil.relativedelta import relativedelta
from functools import partial
from p_tqdm import p_map
from joblib import Parallel, delayed

def find_original_ids(analysis:pd.DataFrame):
    """Finds row indexes which have the parent_id equal to the row index, thus picking out only the actually ran simulations rather than artifically added txcost scenarios etc. """
    original_ids = []
    for idx in analysis.index.get_level_values(0).unique(0):
        if analysis.loc[idx, "parent_id"] == idx:
            original_ids.append(idx)
    return original_ids

def change_txcost_in_backtests(backtests:pd.DataFrame, old_txcost, new_txcost, workers=int(os.environ.get("cpu", len(os.sched_getaffinity(0))))):
    pd.set_option('mode.chained_assignment', None)

    # backtests = backtests.copy(deep=True)
    worker = partial(change_txcost_in_backtest, old_txcost=old_txcost, new_txcost=new_txcost)
    shards = np.array_split(backtests, workers)

    result = Parallel(n_jobs=workers, verbose=1)(delayed(worker)(backtests.loc[backtest_idx]) for backtest_idx in backtests.index.get_level_values(0).unique(0))
    return pd.concat(result, keys=range(len(backtests.index.get_level_values(0).unique(0))))

def change_txcost_in_backtest(backtest:pd.DataFrame, old_txcost:float, new_txcost:float, copy=True):
    """Generates new synthetic backtets with changed txcost

    Args:
        backtest (pd.DataFrame): [description]
        old_txcost (float): [description]
        new_txcost (float): [description]
        copy (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if copy is True:
        backtest=backtest.copy(deep=True)
    trade_signals = ['Short', 'Long', 'Sell', 'sellShort', 'sellLong']
    for pair in backtest.index.get_level_values('Pair').unique('Pair'):
        mask = backtest.loc[pair, "Signals"].isin(trade_signals)
        spread_beta_for_pair = backtest.loc[pair, "SpreadBeta"].iloc[0]
        backtest.loc[pair].loc[mask, "Profit"] = backtest.loc[pair].loc[mask, "Profit"] + 1 * old_txcost + spread_beta_for_pair * old_txcost
        backtest.loc[pair].loc[mask, "Profit"] = backtest.loc[pair].loc[mask, "Profit"] - 1 * new_txcost - spread_beta_for_pair * old_txcost
        backtest.loc[pair, "cumProfit"] = (backtest.loc[pair, "Profit"].cumsum()+1).values
    return backtest

def calculate_new_experiments_txcost(analysis:pd.DataFrame, new_txcosts:List[float], original_only = True, add_backtests=True):
    """Adds new rows to the Analysis DF by changing txcost inside the backtest DF (which is a fairly easy manipulation of the Profit calculation)"""
    if original_only is True:
        admissible_ids = find_original_ids(analysis)
    else:
        admissible_ids = analysis.index.values
    
    new_rows = []
    for new_txcost in new_txcosts:
        for admissible_id in tqdm(admissible_ids, desc='Going over admissible ids'):
            generated = analysis.loc[admissible_id].copy(deep=True)
            if add_backtests is True:
                generated["backtests"] = change_txcost_in_backtests(generated["backtests"], old_txcost=generated["txcost"], new_txcost=new_txcost)
            new_rows.append(generated)
    
    return new_rows

def pick_range(df: pd.DataFrame, start=None, end=None):
    """ Slices preprocessed index-wise to achieve y[start:end], taking into account the MultiIndex
    DF should have index of the shape TICKER-TIME"""

    new_df = []
    for ticker in df.index.unique(0):
        interim = df.loc[ticker]
        new_df.append(
            interim.loc[
                (interim.index > pd.to_datetime(start))
                & (interim.index <= pd.to_datetime(end))
            ]
        )
    df = pd.concat(new_df, keys=df.index.unique(0))
    result = df
    # The old way of doing this - should be qul
    # result = df.groupby(level=0).apply(lambda x: x.loc[x.index.levels[1] > pd.to_datetime(start) & x.index.levels[1] <= pd.to_datetime(end)])
    if result.index.names[0] == result.index.names[1]:
        result = result.droplevel(level=0)

    return result


def guess_ids_in_period(
    start_date,
    end_date,
    desired_start_date,
    desired_end_date,
    formation_delta,
    trading_delta,
    jump_delta,
):
    """I though this would with speed of backtests_up_to_date but apparently not """
    (start_date, end_date, desired_start_date, desired_end_date,) = [
        pd.to_datetime(x) if type(x) is str else x
        for x in [start_date, end_date, desired_start_date, desired_end_date,]
    ]
    formation_delta, trading_delta, jump_delta = [
        relativedelta(months=x[0], days=x[1], hours=x[2]) if type(x) is list else x
        for x in [formation_delta, trading_delta, jump_delta]
    ]

    whole_period_delta = formation_delta + jump_delta
    whole_period_delta_months = (
        whole_period_delta.years * 12 + whole_period_delta.months
    )
    start_idx = None
    end_idx = None
    for i in range(1000000):
        if start_date + jump_delta * i >= desired_start_date and start_idx is None:
            start_idx = i

        if (
            start_date + whole_period_delta + jump_delta * i > desired_end_date
            and end_idx is None
        ):
            end_idx = i
            break

    return (start_idx, end_idx)



def backtests_up_to_date(
    backtests: pd.DataFrame,
    min_formation_period_start=None,
    max_trading_period_end: str = None,
    print_chosen_periods=False,
):
    if type(max_trading_period_end) is str:
        max_trading_period_end = pd.to_datetime(max_trading_period_end)
    if type(min_formation_period_start) is str:
        min_formation_period_start = pd.to_datetime(min_formation_period_start)

    if min_formation_period_start is None:
        min_formation_period_start = backtests.iloc[0]
    backtests_trimmed = []
    backtests_trimmed_idxs = []

    # for experiment_idx in tqdm(analysis.index, desc='Going through backtests'):
    #     row = analysis.loc[experiment_idx]
    #     for backtests in row["backtests"]:
    #         start_id, end_id = guess_ids_in_period(
    #             start_date=row["start_date"],
    #             end_date=row["end_date"],
    #             desired_start_date=min_formation_period_start,
    #             desired_end_date=max_trading_period_end,
    #             formation_delta=row["pairs_deltas/formation_delta"],
    #             trading_delta=row["pairs_deltas/training_delta"],
    #             jump_delta=row["config/jump"],
    #         )
    #         backtests_trimmed.append(row["backtests"].loc[range(start_id, end_id)])

    for backtest_idx in backtests.index.get_level_values(0).unique(0):
        periods = infer_periods(backtests.loc[backtest_idx])
        if (
            pd.to_datetime(periods["trading"][1]) < max_trading_period_end
            and pd.to_datetime(periods["formation"][0]) >= min_formation_period_start
        ):
            backtests_trimmed.append(
                pick_range(
                    backtests.loc[backtest_idx],
                    start=min_formation_period_start,
                    end=max_trading_period_end,
                )
            )
            backtests_trimmed_idxs.append(backtest_idx)

        if pd.to_datetime(periods["trading"][1]) > max_trading_period_end:
            break

    return pd.concat(backtests_trimmed, keys=backtests_trimmed_idxs)


def signals_numeric(olddf, copy=True):
    # TODO I THINK THIS IS NOT USED AND WORTHLESSS
    """ Prepares dummy variables so we can make a graph when the pair is open/close etc"""
    df = olddf.copy(deep=copy)
    for name, group in df.groupby(level=0):
        numeric = np.zeros((df.loc[name].shape[0]))
        numeric[
            (df.loc[name, "Signals"] == "Long")
            | (df.loc[name, "Signals"] == "keepLong")
        ] = 1
        numeric[
            (df.loc[name, "Signals"] == "Short")
            | (df.loc[name, "Signals"] == "keepShort")
        ] = -1
        df.loc[name, "Numeric"] = numeric
    return df


def signals_graph(df, pair, timeframe=None):
    # TODO I THINK THIS IS NOT USED AND WORTHLESSS
    if timeframe == None:
        df.loc[pair, "Numeric"].plot()
    else:
        sliced = pick_range(df, timeframe[0], timeframe[1]).loc[pair]
        sliced["Numeric"].plot()

    return 1


def sliced_norm(df, pair, column, timeframe):
    """ normalizes a dataframe by timeframe slice (afterwards, the mean overall
     is not actually 0 etc) """
    sliced = pick_range(df, timeframe[0], timeframe[1])
    diff = df.loc[pair[0], column] - df.loc[pair[1], column]
    mean = (sliced.loc[pair[0], column] - sliced.loc[pair[1], column]).mean()
    std = (sliced.loc[pair[0], column] - sliced.loc[pair[1], column]).std()
    result = ((diff - mean) / std).values
    result = result[~np.isnan(result)]
    return result


def weights_from_signals(df, cost=0):
    """ Sets the initial weights on position open so they can be propagated"""
    df.loc[df["Signals"] == "Long", "1Weights"] = -df.loc[
        df["Signals"] == "Long", "SpreadBeta"
    ] * (1 + cost)
    df.loc[df["Signals"] == "Long", "2Weights"] = 1 * (1 - cost)
    df.loc[df["Signals"] == "Short", "1Weights"] = df.loc[
        df["Signals"] == "Short", "SpreadBeta"
    ] * (1 - cost)
    df.loc[df["Signals"] == "Short", "2Weights"] = -1 * (1 + cost)
    df.loc[
        (
            (df["Signals"] == "sellLong")
            | (df["Signals"] == "sellShort")
            | (df["Signals"] == "Sell")
        ),
        "1Weights",
    ] = 0
    df.loc[
        (
            (df["Signals"] == "sellLong")
            | (df["Signals"] == "sellShort")
            | (df["Signals"] == "Sell")
        ),
        "2Weights",
    ] = 0


def resample(df, freq: str = "1D", start=None, fill: bool = True):
    """ Our original data is 1-min resolution, so we resample it to arbitrary frequency.
    Close prices get last values, Volume gets summed. 
    Only indexes past start_date are returned to have a common start for all series 
    (since they got listed at various dates)"""
    df.index = pd.to_datetime(df.Date)
    # Close prices get resampled with last values, whereas Volume gets summed
    if freq is not None:
        df["Close"] = df["Close"].resample(freq).last()
        df["Volume"] = df["Volume"] * df["Close"]
        df = df.resample(freq).agg({"Volume": np.sum})
    else:
        df["Volume"] = df["Volume"] * df["Close"]
    # log returns and normalization
    df["Close"] = df["Close"]
    if fill == True:
        df["Close"] = df["Close"].fillna(method="ffill")
    df["logClose"] = np.log(df["Close"])
    df["logReturns"] = (df["logClose"] - df["logClose"].shift(1)).values
    df["Price"] = df["logReturns"].cumsum()
    return df[df.index > pd.to_datetime(start)]


def calculate_spreads(df, viable_pairs, timeframe, betas=None, show_progress_bar=True):
    """Picks out the viable pairs of the original df (which has all pairs)
    and adds to it the normPrice Spread among others, as well as initially
    defines Weights and Profit """
    idx = pd.IndexSlice
    spreads = []
    # the 1,1 betas are for Distance method and even though Cointegration would have different coeffs, I am not sure why it is here?
    if betas == None:
        betas = [np.array([1, 1]) for i in range(len(viable_pairs))]
    for pair, coefs in tqdm(
        zip(viable_pairs, betas),
        desc="Calculating spreads",
        disable=not show_progress_bar,
        total=len(viable_pairs),
    ):
        # labels will be IOTAADA rather that IOTABTCADABTC,
        # so we remove the last three characters
        first = re.sub(r"USDT$|USD$|BTC$", "", pair[0])
        second = re.sub(r"USDT$|USD$|BTC$", "", pair[1])
        composed = first + "x" + second
        pairs_index_intersection = df.loc[pair[0]].index.intersection(
            df.loc[pair[1]].index
        )

        multiindex = pd.MultiIndex.from_product(
            [[composed], pairs_index_intersection], names=["Pair", "Time"]
        )
        newdf = pd.DataFrame(index=multiindex)
        newdf["1Weights"] = None
        newdf["2Weights"] = None
        newdf["Profit"] = 0
        sliced_norm_logreturns = sliced_norm(df, pair, "logReturns", timeframe)

        newdf["normLogReturns"] = sliced_norm_logreturns
        newdf["1Price"] = df.loc[(pair[0], pairs_index_intersection), "Price"].values
        newdf["2Price"] = df.loc[(pair[1], pairs_index_intersection), "Price"].values

        spreads_interim = (
            -coefs[1] * df.loc[(pair[0], pairs_index_intersection), "Price"]
            + df.loc[(pair[1], pairs_index_intersection), "Price"].values
        )
        newdf["Spread"] = spreads_interim.values
        newdf["SpreadBeta"] = coefs[1]
        newdf["normSpread"] = (
            (
                newdf["Spread"]
                - newdf.loc[idx[composed, timeframe[0] : timeframe[1]], "Spread"].mean()
            )
            / newdf.loc[idx[composed, timeframe[0] : timeframe[1]], "Spread"].std()
        ).values

        spreads.append(newdf)
    if len(spreads) == 0:
        return pd.DataFrame(spreads)
    else:
        return pd.concat(spreads)


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def propagate_weights(df, formation_timeframe: List):
    """Propagates weights according to price changes
    Timeframe should be Formation """
    for name, group in df.groupby(level=0):
        # The end of formation might fall on weekend and US stock data do not have any rows for non-trading days
        if formation_timeframe[1] in df.loc[name].index:
            end_of_formation = df.loc[name].index.get_loc(formation_timeframe[1])
        elif formation_timeframe[1] - datetime.timedelta(days=1) in df.loc[name].index:
            end_of_formation = df.loc[name].index.get_loc(
                formation_timeframe[1] - datetime.timedelta(days=1)
            )
        elif formation_timeframe[1] - datetime.timedelta(days=2) in df.loc[name].index:
            end_of_formation = df.loc[name].index.get_loc(
                formation_timeframe[1] - datetime.timedelta(days=2), method="backfill"
            )
        else:
            end_of_formation = df.loc[name].index.get_loc(
                formation_timeframe[1] - datetime.timedelta(days=2), method="nearest"
            )

        temp_weights1 = group["1Weights"].to_list()
        temp_weights2 = group["2Weights"].to_list()
        return1 = group["1Price"] - group["1Price"].shift(1)
        return2 = group["2Price"] - group["2Price"].shift(1)
        for i in range(end_of_formation + 1, len(group.index)):
            # I think shifting the index to god knows where might break this? Better put try to be safe
            try:
                if group.iloc[i]["Signals"] in ["keepLong", "keepShort"]:

                    temp_weights1[i] = temp_weights1[i - 1] * (1 + return1.iloc[i])
                    temp_weights2[i] = temp_weights2[i - 1] * (1 + return2.iloc[i])
            except:
                continue
        df.loc[name, "1Weights"] = temp_weights1
        df.loc[name, "2Weights"] = temp_weights2


def calculate_profit(df, cost=0):
    """Inplace calculates the profit per period as well as a cumulative profit
    Be careful to have the same cost as weights_from_signals
    This function counts the cost in Profit, while w_f_s does it for weights
    So its not double counting or anything"""
    idx = pd.IndexSlice
    mask = (df.loc[idx[:, "Signals"]] == "Long") | (
        df.loc[idx[:, "Signals"]] == "Short"
    )
    # used for cumProfit propagation
    mask2 = (
        (df.loc[idx[:, "Signals"]] == "Long")
        | (df.loc[idx[:, "Signals"]] == "Short")
        | (df.loc[idx[:, "Signals"]] == "keepShort")
        | (df.loc[idx[:, "Signals"]] == "keepLong")
        | (df.loc[idx[:, "Signals"]] == "Sell")
        | (df.loc[idx[:, "Signals"]] == "sellShort")
        | (df.loc[idx[:, "Signals"]] == "sellLong")
    )
    for name, group in df.groupby(level=0):
        returns1 = group["1Price"] - group["1Price"].shift(1).values
        returns2 = group["2Price"] - group["2Price"].shift(1).values
        df.loc[name, "Profit"] = (
            df.loc[name, "1Weights"].shift(1) * returns1
            + df.loc[name, "2Weights"].shift(1) * returns2
        )
        df.loc[idx[name, mask], "Profit"] = -(
            df.loc[idx[name, mask], "1Weights"].abs() * cost
            + df.loc[idx[name, mask], "2Weights"].abs() * cost
        )
        df.loc[idx[name, mask2], "cumProfit"] = (
            df.loc[idx[name, mask2], "Profit"]
        ).cumsum() + 1


def signals_worker(
    multidf,
    start_date,
    end_date,
    trading_timeframe=5,
    formation=5,
    threshold=2,
    lag=0,
    stoploss=100,
    num_of_processes=1,
):

    # Without the copy, it was causing bugs since this there is in-place mutation - in particular, the simulation scheme where we share the dist/coint signals and have multiple parameters after that would cause problems
    multidf = multidf.copy(deep=True)

    # Stoploss should signify the number in excess of the threshold that causes stoploss!
    stoploss = threshold + stoploss
    idx = pd.IndexSlice
    for name, df in multidf.loc[
        pd.IndexSlice[:, trading_timeframe[0] : trading_timeframe[1]], :
    ].groupby(level=0):
        df["Signals"] = None
        # df.loc[mask,'Signals'] = True
        index = df.index
        # this is technicality because we truncate the DF to just trading period but
        # in the first few periods the signal generation needs to access prior values
        # which would be None so we just make them adhoc like this
        col = [None for x in range(lag + 2)]
        fill = "None"
        for i in range(len(df)):
            truei = i
            if i - lag < 0:
                col.append(fill)
                continue
            if (df.loc[index[i - lag], "normSpread"] > stoploss) & (
                col[i + lag + 1] in ["Short", "keepShort"]
            ):
                fill = "stopShortLoss"
                col.append(fill)
                fill = "None"
                continue
            if (df.loc[index[i - lag], "normSpread"] < (-stoploss)) & (
                col[i + lag + 1] in ["Long", "keepLong"]
            ):
                fill = "stopLongLoss"
                col.append(fill)
                fill = "None"
                continue
            if (
                (df.loc[index[i - lag], "normSpread"] >= threshold)
                & (df.loc[index[i - lag - 1], "normSpread"] < threshold)
                & (col[i + lag - 1] not in ["keepShort"])
                & (col[truei + lag + 1] not in ["Short", "keepShort"])
            ):
                fill = "Short"
                col.append(fill)
                fill = "keepShort"
                continue
            elif (
                (df.loc[index[i - lag], "normSpread"] <= 0)
                & (df.loc[index[i - lag - 1], "normSpread"] > 0)
                & (col[i + lag + 1] in ["Short", "keepShort"])
            ):
                fill = "sellShort"
                col.append(fill)
                fill = "None"
                continue
            elif (
                (
                    (df.loc[index[i - lag], "normSpread"] <= (-threshold))
                    & (df.loc[index[i - lag - 1], "normSpread"] > (-threshold))
                )
                & (col[i + lag - 1] not in ["keepLong"])
                & (col[truei + lag + 1] not in ["Long", "keepLong"])
            ):
                fill = "Long"
                col.append(fill)
                fill = "keepLong"
                continue
            elif (
                (df.loc[index[i - lag], "normSpread"] >= 0)
                & (df.loc[index[i - lag - 1], "normSpread"] < 0)
                & (col[i + lag + 1] in ["Long", "keepLong"])
            ):
                fill = "sellLong"
                col.append(fill)
                fill = "None"
                continue
            col.append(fill)
        col = col[(lag + 2) : -1]
        col.append("Sell")
        # df['Signals'] = pd.Series(col[1:], index=df.index)
        multidf.loc[
            pd.IndexSlice[name, trading_timeframe[0] : trading_timeframe[1]], "Signals"
        ] = pd.Series(col, index=df.index)
        multidf.loc[
            pd.IndexSlice[name, trading_timeframe[1] : end_date], "Signals"
        ] = None
    multidf.loc[idx[:, trading_timeframe[1] : end_date], "Signals"] = None
    multidf.loc[idx[:, trading_timeframe[1] : end_date], "Signals"] = multidf.loc[
        idx[:, trading_timeframe[1] : end_date], "Signals"
    ].fillna(value="pastFormation")
    multidf.loc[idx[:, formation[0] : formation[1]], "Signals"] = multidf.loc[
        idx[:, formation[0] : formation[1]], "Signals"
    ].fillna(value="Formation")
    multidf.loc[idx[:, start_date : formation[0]], "Signals"] = multidf.loc[
        idx[:, start_date : formation[0]], "Signals"
    ].fillna(value="preFormation")
    # multidf['Signals'] = multidf['Signals'].fillna(value='Formation')
    return multidf


def signals(
    multidf,
    start_date,
    end_date,
    trading_timeframe=None,
    formation=None,
    threshold=2,
    lag=0,
    stoploss=100,
    num_of_processes=1,
):
    """ Fills in the Signals during timeframe period 
    Outside of the trading period, it fills Formation and pastTrading"""
    if num_of_processes == 1:

        return signals_worker(
            multidf,
            start_date=start_date,
            end_date=end_date,
            trading_timeframe=trading_timeframe,
            formation=formation,
            threshold=threshold,
            lag=lag,
        )
    if num_of_processes > 1:
        if len(multidf.index.unique(level=0)) < num_of_processes:
            num_of_processes = len(multidf.index.unique(level=0))
        pool = mp.Pool(num_of_processes)
        split = np.array_split(multidf.index.unique(level=0), num_of_processes)
        split = [multidf.loc[x] for x in split]
        # Im not sure what I was doing here to be honest..
        args_dict = {
            "start_date": start_date,
            "end_date": end_date,
            "trading": trading_timeframe,
            "formation": formation,
            "threshold": threshold,
            "lag": lag,
            "stoploss": stoploss,
            "num_of_processes": num_of_processes,
        }
        args = [
            args_dict["start_date"],
            args_dict["end_date"],
            args_dict["trading"],
            args_dict["formation"],
            args_dict["threshold"],
            args_dict["lag"],
            args_dict["stoploss"],
            args_dict["num_of_processes"],
        ]
        full_args = [[split[i], *args] for i in range(len(split))]
        results = pool.starmap(signals_worker, full_args)
        results = pd.concat(results)
        pool.close()
        pool.join()
        return results
