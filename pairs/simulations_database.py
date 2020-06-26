import datetime
from pairs.config import start_date, end_date, data_path, save, paper1_data_path, paper1_results
from pairs.config import TradingUniverse
from pairs.datasets.us_dataset import USDataset
import itertools
import time
import mlflow
from pairs.helpers import remake_into_lists
import ray
from ray import tune

def generate_scenario(
    freq="1D",
    lag=1,
    txcost=0.003,
    pairs_deltas = {"training_delta": [2,0,0], "formation_delta":[4,0,0]},
    jump=[1, 0, 0],
    method="dist",
    dist_num=20,
    confidence=0.05,
    threshold=2,
    stoploss=100,
    redo_prefiltered=False,
    redo_preprocessed=False,
    truncate=True,
    trading_univ: TradingUniverse = TradingUniverse(),
    dataset = None
):
    # mlflow.set_tracking_uri(trading_univ["tracking_uri"])
    # lag, stoploss, threshold, txcost = remake_into_lists(lag, stoploss, threshold, txcost)
    # step2_arg_product = list(itertools.product(lag,threshold, stoploss, txcost))
    run_ids = []
    # for _ in step2_arg_product:
    #     run = mlflow.start_run(experiment_id= mlflow.get_experiment_by_name(trading_univ["name"]).experiment_id)
    #     run_ids.append(run.info.run_id)
    #     mlflow.end_run()
    print(pairs_deltas)
    to_return = {
        "freq": freq,
        "lag": lag,
        "txcost": txcost,
        'jump':[1,0,0],
        'method': method,
        'dist_num':dist_num,
        "pairs_deltas":pairs_deltas,
        "confidence" : confidence,
        'threshold':threshold,
        'stoploss':stoploss,
        'redo_prefiltered':redo_prefiltered,
        'redo_preprocessed':redo_preprocessed,
        'truncate':True,
        'data_path':trading_univ["data_path"],
        'save_path_results':trading_univ["save_path_results"],
        'show_progress_bar':trading_univ["show_progress_bar"],
        'saving_method':trading_univ["saving_method"],
        'start_date':trading_univ["start_date"],
        'end_date':trading_univ["end_date"],
        'name':trading_univ["name"],
        "volume_cutoff": trading_univ["volume_cutoff"],
        "tracking_uri": trading_univ["tracking_uri"],
        "dataset":dataset,
        "trading_univ":trading_univ,
        "run_ids":run_ids
    }
    # if 'grid_search' in pairs_deltas.keys():
    #     to_return["pairs_deltas"] = 
    return to_return

# generate_scenario(
#             freq="1D",
#             lag=tune.grid_search([0,1]),
#             txcost=[0, 0.003, 0.005],
#             pairs_deltas = tune.grid_search([{"training_delta":[3,0,0], "formation_delta":[6,0,0]},
#                 {"training_delta":[6,0,0], "formation_delta":[12,0,0]},
#                 {"training_delta":[1,0,0], "formation_delta":[2,0,0]}]),
#             jump=[1, 0, 0],
#             method="dist",
#             dist_num=tune.grid_search([5, 10, 20, 40, 70, 100]),
#             threshold=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
#             stoploss=[0, 1, 2],
#             redo_prefiltered=True,
#             redo_preprocessed=True,
#             truncate=True,
#             trading_univ=univ,
#             dataset=USDataset(config=univ),
#         )
# Order sensitive!
#################DAILY
starting_date = start_date
ending_date = end_date
scenario1 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [2, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario1",
}
scenario2 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [2, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario2",
}
# changed cutoff
scenario1_1 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [2, 0, 0],
    "cutoff": 0.0,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario1_1",
}
scenario2_1 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [2, 0, 0],
    "cutoff": 0.0,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario2_2",
}
# COINT version
scenario1_coint = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [2, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario1",
}
scenario2_coint = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [2, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario2",
}
scenario1_1 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [2, 0, 0],
    "cutoff": 0.0,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario1_1",
}
scenario2_1 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [2, 0, 0],
    "cutoff": 0.0,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario2_2",
}

# NOLAG
scenario1_nolag = {
    "freq": "1D",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [2, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario1_nolag",
}
scenario2_nolag = {
    "freq": "1D",
    "lag": 0,
    "txcost": 0.000,
    "training_delta": [2, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [4, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario2_nolag",
}

################HOURLY
scenario3 = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 10, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario3",
}
scenario4 = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 10, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario4",
}
# changed cutoff
scenario3_1 = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 10, 0],
    "cutoff": 0.0,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario3_1",
}
scenario4_1 = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 10, 0],
    "cutoff": 0.0,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario4_1",
}

# NOLAG
scenario3_nolag = {
    "freq": "1H",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [0, 10, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario3_nolag",
}
scenario4_nolag = {
    "freq": "1H",
    "lag": 0,
    "txcost": 0.000,
    "training_delta": [0, 10, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario4_nolag",
}
# COINT version
scenario3_coint = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 10, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario3",
}
scenario4_coint = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 10, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 20, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 10, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario4",
}
scenario3_1_coint = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 15, 0],
    "cutoff": 0.0,
    "formation_delta": [1, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario3_1",
}
scenario4_1_coint = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 15, 0],
    "cutoff": 0.0,
    "formation_delta": [1, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario4_1",
}

# VARIOUS DELTA SCHEMES
scenario3_nolag1 = {
    "freq": "1H",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [0, 15, 0],
    "cutoff": 0.7,
    "formation_delta": [1, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 15, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario3_nolag1",
}
scenario1_nolag1 = {
    "freq": "1D",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [3, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [6, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario1_nolag1",
}
scenario11 = {
    "freq": "1D",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [3, 0, 0],
    "cutoff": 0.7,
    "formation_delta": [6, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [1, 0, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario11",
}
scenario31 = {
    "freq": "1H",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 15, 0],
    "cutoff": 0.7,
    "formation_delta": [1, 0, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 15, 0],
    "methods": ["dist", "coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario31",
}

#############MINUTE
scenario5 = {
    "freq": "1T",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 1, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 2, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 2, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario5",
}
scenario6 = {
    "freq": "1T",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 1, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 2, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 2, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario6",
}
scenario5_coint = {
    "freq": "1T",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 1, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 2, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 2, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario5",
}
scenario6_coint = {
    "freq": "1T",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 1, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 2, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 2, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario6",
}

scenario5_coint_nolag = {
    "freq": "1T",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [0, 1, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 2, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 2, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario5_nolag",
}
scenario5_nolag = {
    "freq": "1T",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [0, 1, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 2, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 2, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario5_nolag",
}


#############5MINUTE
scenario7 = {
    "freq": "5T",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 3, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 6, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 6, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario7",
}
scenario8 = {
    "freq": "5T",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 3, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 6, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 6, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario8",
}
scenario7_coint = {
    "freq": "5T",
    "lag": 1,
    "txcost": 0.003,
    "training_delta": [0, 3, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 6, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 6, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario7",
}
scenario8_coint = {
    "freq": "5T",
    "lag": 1,
    "txcost": 0.000,
    "training_delta": [0, 3, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 6, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 6, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario8",
}

scenario7_coint_nolag = {
    "freq": "5T",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [0, 3, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 6, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 6, 0],
    "methods": ["coint"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario7_nolag",
}
scenario7_nolag = {
    "freq": "5T",
    "lag": 0,
    "txcost": 0.003,
    "training_delta": [0, 3, 0],
    "cutoff": 0.7,
    "formation_delta": [0, 6, 0],
    "start": starting_date,
    "end": ending_date,
    "jump": [0, 6, 0],
    "methods": ["dist"],
    "dist_num": 20,
    "threshold": 2,
    "stoploss": 100,
    "name": "scenario7_nolag",
}
