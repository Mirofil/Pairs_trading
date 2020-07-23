import numpy as np
import pandas as pd
from typing import *

def beautify(df, overlap=False):
    formats = {
        "Monthly profit": "{:.2%}",
        "Annual profit": "{:.2%}",
        "Total profit": "{:.2%}",
        "Sharpe": "{:.2}",
        "Roundtrip trades": "{:.2%}",
        "Number of trades": "{:.3}",
        "Avg length of position": "{:3.1f}",
        "Pct of winning trades": "{:.2%}",
        "Max drawdown": "{:.2%}",
        "Cumulative profit": "{:.2%}",
        "Mean": "{:1.5f}",
        "Std": "{:.3}",
        "Max": "{:.3}",
        "Min": "{:.3}",
        "Jarque-Bera p-value": "{:.3}",
        "Skewness": "{:.3}",
        "Kurtosis": "{:.3}",
        "Positive": "{:.2%}",
        "t-stat": "{:.2}",
        "Sortino": "{:.2}",
        "Calmar": "{:.2}",
        "VaR": "{:.2}",
        "\% of identical pairs": "{:.2%}",
        "Trading p. Sharpe": "{:.2}",
        "Annualized Sharpe": "{:.2}",
        "Trading period Sharpe": "{:.2}",
        "Monthly number of trades": "{:.3}",
        "Length of position (days)": "{:3.1f}",
        "Monthly profit (committed)": "{:.2%}",
        "Nominated pairs": "{:.3}",
        "Traded pairs": "{:.2%}",
        "Standard deviation":"{:.2}",
        "Corr. to market": "{:.2}"
    }
    if overlap == False:
        df = df.astype("float32")
        for row in df.index:
            if row in formats.keys():
                df.loc[row] = df.loc[row].map(formats[row].format)
                df.loc[row] = df.loc[row].apply(lambda x: x.replace("%", "\%"))
        df = df.rename(index={"Total profit": "Trading period profit"})
    else:
        df = df.astype("float32")
        for i in range(len(df.index)):
            if df.index[i] in formats.keys():
                df.iloc[i] = df.iloc[i].map(formats[df.index[i]].format)
                df.iloc[i] = df.iloc[i].apply(lambda x: x.replace("%", "\%"))
        df = df.rename(index={"Total profit": "Trading period profit"})
    return df


def rhoci(rho, n, conf=0.95):
    mean = np.arctanh(rho)
    std = 1 / ((n - 3) ** (1 / 2))
    norm = scipy.stats.norm(loc=mean, scale=std)
    ci = [mean - 1.96 * std, mean + 1.96 * std]
    trueci = [np.round(np.tanh(ci[0]), 2), np.round(np.tanh(ci[1]), 2)]
    return trueci


def hdist(df):
    grouper = df.index.get_level_values("Time").hour
    res = df.groupby(by=grouper).agg(["mean", "count", "std"])
    res["t-stat"] = res["mean"] / res["std"] * res["count"].pow(1 / 2)
    res.columns = res.columns.str.capitalize()
    res.columns = pd.MultiIndex.from_product([["Returns distribution"], res.columns])
    res.index.rename("Hour", inplace=True)
    return res

def stoploss_table(
    dict,
    stem,
    freq,
    feasible=["Monthly profit", "Annualized Sharpe"],
    standard=False,
    thresh=[1, 2, 3],
    stoploss=[2, 3, 4, 5, 6],
):
    rows = []
    for i in range(len(stoploss)):
        temp = []
        for j in range(len(thresh)):
            temp.append(
                aggregate(
                    [dict[stem + str(thresh[j]) + str(stoploss[i])]],
                    feasible,
                    standard=standard,
                    freqs=freq,
                    trades_nonzero=True,
                    returns_nonzero=True,
                )
            )
        rows.append(temp)
    return rows

def produce_stoploss_table(des, prefix, freqs):
    df = stoploss_table(des, prefix, freqs)
    df = pd.concat(list(map(lambda x: pd.concat(x, axis=1), df)))
    df = beautify(df, overlap=True)
    cols = pd.MultiIndex.from_product([["Threshold"], [1, 2, 3]])
    index = pd.MultiIndex.from_arrays(
        [
            [2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            [
                "Monthly profit",
                "Annualized Sharpe",
                "Monthly profit",
                "Annualized Sharpe",
                "Monthly profit",
                "Annualized Sharpe",
                "Monthly profit",
                "Annualized Sharpe",
                "Monthly profit",
                "Annualized Sharpe",
            ],
        ]
    )
    df.columns = cols
    df.index = index
    df = pd.concat([df], axis=0, keys=["Stop-loss"])
    df = df.round(3)
    # gets rid of entries where threshold < stoploss
    df = filter_nonsense(df)
    return df

def guess_frequency_from_col(aggregated:pd.DataFrame):
    freqs = []
    for col in aggregated.columns:
        if 'Daily' in col:
            freqs.append(1)
        elif 'Hourly' in col:
            freqs.append(1/24)
        elif '5-Minute' in col:
            freqs.append(1/288)

    return freqs

def standardize_results(
    aggregated, poslen=None, numtrades:List[float]=[1 / 2, 1 / 2, 3, 3], drop=True
):
    """This standardizes some metrics that rely on time ratios to days or months

    Args:
        df ([type]): Should be the outcome of aggregate, so one of the final steps
        poslen (list, optional): Used for normalizing the trading period. Should be the length of one time period in days. Defaults to [1, 1, 1 / 24, 1 / 24].
        numtrades (list, optional): Converts the number of trades to monthly. Assumes linear scaling. Defaults to [1 / 2, 1 / 2, 3, 3].
        drop (bool, optional): Drops some garbage columns that are undesirable in the final output. Defaults to True.

    Returns:
        [type]: [description]
    """
    if poslen is None:
        poslen = guess_frequency_from_col(aggregated=aggregated)

    aggregated.loc["Avg length of position"] = (
        aggregated.loc["Avg length of position"].astype("float32") * poslen
    )
    aggregated.loc["Number of trades"] = (
        aggregated.loc["Number of trades"].astype("float32") * numtrades
    )
    aggregated = aggregated.rename(
        {
            "Avg length of position": "Length of position (days)",
            "Number of trades": "Monthly number of trades",
        }
    )
    if drop == True:
        aggregated = aggregated.drop(
            [
                "Trading period profit",
                "Trading period Sharpe",
                "Annual profit",
                "Total profit",
            ],
            errors="ignore",
        )
    return aggregated
