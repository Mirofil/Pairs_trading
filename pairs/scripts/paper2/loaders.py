import datetime
import glob
import os
import pickle
import timeit
from typing import *
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from dateutil.relativedelta import relativedelta
from p_tqdm import p_map
from ray import tune
from tqdm import tqdm
from joblib import Parallel, delayed

from pairs.analysis import (
    aggregate,
    descriptive_frame,
    descriptive_stats,
    infer_periods,
    compute_cols_from_freq,
    compute_period_length_in_days,
)
from pairs.cointmethod import *
from pairs.config import paper1_univ
from pairs.distancemethod import *
from pairs.formatting import beautify, standardize_results
from pairs.helpers import *
from pairs.helpers import latexsave
from pairs.scripts.latex.helpers import *
from pairs.pairs_trading_engine import (
    pick_range,
    backtests_up_to_date,
    calculate_new_experiments_txcost,
    find_original_ids,
)
from pairs.scripts.latex.loaders import join_results_by_id
from pairs.scripts.paper2.helpers import ts_stats, nya_stats
from pairs.scripts.paper2.subperiods import Subperiod


def load_experiment(
    subperiod: Subperiod,
    experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
    ids: List[int] = None,
    trimmed_backtests: pd.DataFrame = None,
    descs: pd.DataFrame = None,
    new_txcosts: List[float] = None,
    workers=int(os.environ.get("cpu", len(os.sched_getaffinity(0)))),
):
    if ids is not None:
        analysis = ray.tune.Analysis(experiment_dir=experiment_dir).dataframe().loc[ids]
    else:
        analysis = ray.tune.Analysis(experiment_dir=experiment_dir).dataframe()
    analysis["pairs_deltas/formation_delta"] = analysis[
        "pairs_deltas/formation_delta"
    ].apply(lambda x: eval(x))
    analysis["pairs_deltas/training_delta"] = analysis[
        "pairs_deltas/training_delta"
    ].apply(lambda x: eval(x))
    analysis["parent_id"] = analysis.index.values

    analysis = join_results_by_id(analysis, ids=ids)

    if trimmed_backtests is None:
        worker = partial(
            backtests_up_to_date,
            min_formation_period_start=subperiod.start_date,
            max_trading_period_end=subperiod.end_date,
        )
        trimmed_backtests = Parallel(n_jobs=workers, verbose=1)(
            delayed(worker)(backtests)
            for backtests in tqdm(analysis["backtests"], desc="Trimming backtests")
        )

        analysis["backtests"] = trimmed_backtests

    if new_txcosts is not None:
        new_rows = calculate_new_experiments_txcost(analysis, new_txcosts)
        analysis = analysis.append(new_rows)

    if type(descs) is str and os.path.isfile(descs):
        descs = pd.read_parquet(descs)
    elif descs is None:
        # descs = p_map(descriptive_frame, trimmed_backtests, num_cpus=workers)
        descs = Parallel(n_jobs=workers, verbose=1)(
            delayed(descriptive_frame)(backtests)
            for backtests in tqdm(analysis["backtests"], desc="Desc frames")
        )
        descs = pd.DataFrame(pd.Series(descs, index=analysis.index, name="descs"))
        descs = pd.DataFrame(pd.concat(
            [desc.iloc[0] for desc in descs["descs"]], keys=range(len(descs))
        ), columns = ['descs'])
        descs.to_parquet(
            os.path.join(experiment_dir, f"{subperiod.name}_descs.parquet")
        )

    descs_interim = []
    for experiment_idx in descs.index.get_level_values(0).unique(0):
        descs_interim.append(descs.loc[experiment_idx])
    analysis["descs"] = descs_interim

    analysis["parent_id"] = analysis.index.values

    return analysis
