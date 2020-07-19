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
from loguru import logger

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
from pairs.scripts.latex.loaders import join_backtests_by_id
from pairs.scripts.paper2.helpers import ts_stats, nya_stats
from pairs.scripts.paper2.subperiods import Subperiod
from pairs.analysis import find_scenario


def load_analysis_dataframe(
    experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
    ids: List[int] = None,
    exclude_params: List[Dict] = None,
):
    if ids is not None:
        analysis = ray.tune.Analysis(experiment_dir=experiment_dir).dataframe().loc[ids]
    else:
        analysis = ray.tune.Analysis(experiment_dir=experiment_dir).dataframe()
    if exclude_params is not None:
        for forbidden_params in exclude_params:
            analysis = find_scenario(analysis, forbidden_params, negate=True)
    analysis["pairs_deltas/formation_delta"] = analysis[
        "pairs_deltas/formation_delta"
    ].apply(lambda x: eval(x))
    analysis["pairs_deltas/training_delta"] = analysis[
        "pairs_deltas/training_delta"
    ].apply(lambda x: eval(x))
    analysis["parent_id"] = analysis.index.values
    return analysis

def load_descs_from_parquet(fpath):
    descs = pd.read_parquet(fpath)
    reshaped_descs= []
    for run_idx in descs.index.get_level_values(0).unique(0):
        candidate = descs.loc[run_idx]
        candidate.index = pd.MultiIndex.from_product([[0], candidate.index])
        reshaped_descs.append(candidate)
    descs = pd.DataFrame(pd.Series(reshaped_descs, index=range(len(reshaped_descs))), columns=["descs"])
    return descs

def load_experiment(
    subperiod: Subperiod,
    ids: Optional[List[int]] = None,
    experiment_dir: str = "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
    new_txcosts: List[float] = None,
    exclude_params: List[Dict] = [
        {
            "config/pairs_deltas": {
                "formation_delta": [10, 0, 0],
                "training_delta": [20, 0, 0],
            },
        },
        {
            "config/pairs_deltas": {
                "formation_delta": [1, 0, 0],
                "training_delta": [6, 0, 0],
            }
        },
    ]
):
    analysis = load_analysis_dataframe(
        experiment_dir=experiment_dir, ids=ids, exclude_params=exclude_params
    )
    if new_txcosts is not None:
        new_rows = calculate_new_experiments_txcost(
            analysis, new_txcosts, add_backtests=False
        )
        analysis = analysis.append(new_rows)
        analysis = analysis.reset_index()

    analysis["descs"]=load_descs_from_parquet(os.path.join(experiment_dir, f"{subperiod.name}_descs.parquet"))

    return analysis

def process_experiment(
    subperiod: Subperiod,
    experiment_dir="/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/",
    ids: List[int] = None,
    trimmed_backtests: pd.DataFrame = None,
    descs: pd.DataFrame = None,
    new_txcosts: List[float] = None,
    workers=int(os.environ.get("cpu", len(os.sched_getaffinity(0)))),
    exclude_params: List[Dict] = [
        {
            "config/pairs_deltas": {
                "formation_delta": [10, 0, 0],
                "training_delta": [20, 0, 0],
            },
        },
        {
            "config/pairs_deltas": {
                "formation_delta": [1, 0, 0],
                "training_delta": [6, 0, 0],
            }
        },
    ],
):
    """Loading and processing the experiment results from Ray. Saves some files for later reuse also.
    I generated wrong configs for some of additional distance scenario experiments, and I think I wont be doing them for coint anyways.. 
        so I will just ignore it at this point, which is what the default value of exclude_params should do

    Args:
        subperiod (Subperiod): The subperiod we are analyzing, ie. nineties, dotcom crash etc
        experiment_dir (str, optional): Ray_results dir. Defaults to "/mnt/shared/dev/code_knowbot/miroslav/test/Pairs_trading2/ray_results/simulate_dist_retries_nomlflow/".
        ids (List[int], optional): Only take those ids. Defaults to None.
        trimmed_backtests (pd.DataFrame, optional): If trimmed_backtests are supplied, they are used instead of computed. Defaults to None.
        descs (pd.DataFrame, optional): If descs are supplied, they are used instead of computed. Defaults to None.
        new_txcosts (List[float], optional): Generates new Txcost scenarios based on those in a cartesian product way. Defaults to None.
        workers ([type], optional): Workers for Joblib. Defaults to int(os.environ.get("cpu", len(os.sched_getaffinity(0)))).
        exclude_params (List[Dict], optional): Exclude scenarios with those param values. Defaults to [ { "config/pairs_deltas": { "formation_delta": [10, 0, 0], "training_delta": [20, 0, 0], }, }, { "config/pairs_deltas": { "formation_delta": [1, 0, 0], "training_delta": [6, 0, 0], } }, ].

    Returns:
        [type]: [description]
    """
    analysis = load_analysis_dataframe(
        experiment_dir=experiment_dir, ids=ids, exclude_params=exclude_params
    )

    analysis = join_backtests_by_id(analysis, ids=ids)

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

        # for idx, backtests in enumerate(analysis["backtests"]):
        #     results = []
        #     print(idx)
        #     results.append(
        #         backtests_up_to_date(
        #             backtests,
        #             min_formation_period_start=subperiod.start_date,
        #             max_trading_period_end=subperiod.end_date,
        #         )
        #     )

        analysis["backtests"] = trimmed_backtests

    if new_txcosts is not None:
        new_rows = calculate_new_experiments_txcost(
            analysis, new_txcosts, add_backtests=True
        )
        analysis = analysis.append(new_rows)
        analysis = analysis.reset_index()

    if type(descs) is str and os.path.isfile(descs):
        descs = pd.read_parquet(descs)
    elif descs is None:
        descs = Parallel(n_jobs=workers, verbose=1)(
            delayed(descriptive_frame)(backtests)
            for backtests in tqdm(analysis["backtests"], desc="Desc frames")
        )
        descs = pd.DataFrame(pd.Series(descs, index=analysis.index, name="descs"))
        descs = pd.DataFrame(
            pd.concat(
                [desc.iloc[0] for desc in descs["descs"]], keys=range(len(descs))
            ),
            columns=["descs"],
        )

        logger.info(
            f"Saving subperiod {subperiod.name} at {os.path.join(experiment_dir, subperiod.name + '_descs.parquet')}"
        )
        descs.to_parquet(
            os.path.join(experiment_dir, f"{subperiod.name}_descs.parquet")
        )

    descs_interim = []
    for experiment_idx in descs.index.get_level_values(0).unique(0):
        descs_interim.append(descs.loc[experiment_idx])
    analysis["descs"] = descs_interim

    # #NOTE this will make the newly added different-txcost-scenarios have the proper parent id because they were a copy of one of the original rows, and thus have their ID as index?
    # analysis["parent_id"] = analysis.index.values

    return analysis
