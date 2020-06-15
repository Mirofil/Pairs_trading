import datetime
import os


class TradingUniverse:
    def __init__(
        self,
        start_date=datetime.date(*[2018, 1, 1]),
        end_date=datetime.date(*[2019, 9, 1]),
        volume_cutoff=0.7,
        root_folder="NEW",
        save_path_graphs=None,
        save_path_tables=None,
        freq='1D'
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.volume_cutoff = volume_cutoff
        self.root_folder = root_folder
        if save_path_graphs is None:
            save_path_graphs = root_folder + "graphs" + os.sep
        if save_path_tables is None:
            save_path_tables = root_folder + "tables" + os.sep

        self.save_path_graphs = save_path_graphs
        self.save_path_tables = save_path_tables

        self.save_path_results = root_folder + "results" + os.sep

        self.freq = freq

    def __getitem__(self, key):
        return getattr(self, key)


start_date = datetime.date(*[2018, 1, 1])
end_date = datetime.date(*[2019, 9, 1])
volume_cutoff = 0.7
version = "NEW"
desc = [
    "Mean",
    "Total profit",
    "Std",
    "Sharpe",
    "Sortino",
    "VaR",
    "Calmar",
    "Number of trades",
    "Roundtrip trades",
    "Avg length of position",
    "Pct of winning trades",
    "Max drawdown",
    "Cumulative profit",
]
data_path = version + r"concatenated_price_data" + os.sep
save = version + r"results" + os.sep
save_path_graphs = version + r"graphs" + os.sep
save_path_tables = version + r"tables" + os.sep
NUMOFPROCESSES = 1
paper1_data_path = "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWconcatenated_price_data"
paper1_results = (
    "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWresults"
)

standard_result_metrics_from_desc_stats = [
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

config = {}
config["start_date"] = start_date
config["end_date"] = end_date
config["volume_cutoff"] = volume_cutoff
config["version"] = version
config["desc"] = desc
config["data_path"] = data_path
config["save"] = save
config["save_path_graphs"] = save_path_graphs
config["save_path_tables"] = save_path_tables
config["NUMOFPROCESSES"] = NUMOFPROCESSES
