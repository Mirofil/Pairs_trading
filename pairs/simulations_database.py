import datetime
from pairs.config import start_date, end_date, data_path, save, paper1_data_path, paper1_results


def generate_scenario(
    freq="1D",
    lag=1,
    txcost=0.003,
    training_delta=[2, 0, 0],
    volume_cutoff=0.7,
    formation_delta=[4, 0, 0],
    start=start_date,
    end=end_date,
    jump=[1, 0, 0],
    method="dist",
    dist_num=20,
    threshold=2,
    stoploss=100,
    redo_prefiltered=False,
    redo_preprocessed=False,
    truncate=True,
    name="scenarioX",
    data_path = paper1_data_path,
    save=paper1_results,
    show_progress_bar=False
):
    return {
        "freq": freq,
        "lag": lag,
        "txcost": txcost,
        "training_delta": training_delta,
        "volume_cutoff": volume_cutoff,
        "training_delta": [2, 0, 0],
        'formation_delta': [4,0,0],
        'start':start_date,
        'end':end_date,
        'jump':[1,0,0],
        'method':'dist',
        'dist_num':dist_num,
        'threshold':threshold,
        'stoploss':stoploss,
        'redo_prefiltered':redo_prefiltered,
        'redo_preprocessed':redo_preprocessed,
        'truncate':True,
        'name':name,
        'data_path':data_path,
        'save':save,
        'show_progress_bar':show_progress_bar
    }


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
