
from pairs.analysis import descriptive_stats, descriptive_frame, summarize, aggregate
from pairs.helpers import load_results
from pairs.formatting import beautify, standardize_results
from pairs.config import paper1_results
import pandas as pd
import os

def test_results_table():
    newbase = paper1_results
    rdd = load_results("scenario1", "dist", newbase)
    rhd = load_results("scenario3", "dist", newbase)
    rdc = load_results("scenario1", "coint", newbase)
    rhc = load_results("scenario3", "coint", newbase)
    ddd = descriptive_frame(rdd)
    dhd = descriptive_frame(rhd)
    ddc = descriptive_frame(rdc)
    dhc = descriptive_frame(rhc)

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
    agg = aggregate(
        [ddd, ddc, dhd, dhc], columns_to_pick=feasible, multiindex_from_product_cols=[["Daily", "Hourly"], ["Dist.", "Coint."]], trades_nonzero=True, returns_nonzero=True
    )
    agg = standardize_results(agg)
    agg = beautify(agg)

    agg.index = agg.index.to_flat_index()
    agg.columns = [str(column) for column in agg.columns]

    parent_dir = os.path.dirname(__file__)
    assert agg.equals(pd.read_parquet(os.path.join(parent_dir, "results_table_reference.parquet")))
