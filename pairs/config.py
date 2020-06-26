import datetime
import os


class TradingUniverse:
    def __init__(
        self,
        start_date=[2018, 1, 1],
        end_date=[2019, 9, 1],
        volume_cutoff=[0.1,1],
        root_folder=None,
        save_path_graphs=None,
        save_path_tables=None,
        save_path_results=None,
        freq="1D",
        data_path=None,
        name="Whatever name",
        saving_method = 'parquet',
        show_progress_bar=False,
        tracking_uri = "http://0.0.0.0:5000"
    ):
        self.start_date = datetime.date(*start_date)
        self.end_date = datetime.date(*end_date)
        self.volume_cutoff = volume_cutoff
        self.root_folder = root_folder
        if save_path_graphs is None and root_folder is not None:
            self.save_path_graphs = os.path.join(root_folder, "graphs")
        else:
            self.save_path_graphs = None

        if save_path_tables is None and root_folder is not None:
            self.save_path_tables = os.path.join(root_folder, "tables")
        else:
            self.save_path_tables = None

        if save_path_results is None and root_folder is not None:
            self.save_path_results = os.path.join(root_folder, "results")
        else:
            self.save_path_results = None

        if data_path is None and root_folder is not None:
            self.data_path = os.path.join(root_folder, "concatenated_price_data")
        else:
            self.data_path = data_path

        self.freq = freq

        self.name = name
        self.saving_method = saving_method
        self.show_progress_bar = show_progress_bar
        self.tracking_uri = tracking_uri

    def __getitem__(self, key):
        return getattr(self, key)


paper1_univ = TradingUniverse(
    start_date=[2018, 1, 1],
    end_date=[2019, 9, 1],
    volume_cutoff=0.7,
    root_folder="paper1",
    data_path="/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWconcatenated_price_data",
    save_path_results = "/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWresults",
    save_path_graphs="/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWgraphs",
    save_path_tables="/Users/miro/Documents/Projects/bachelor/Pairs_trading_new/paper1/NEWtables",
)


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
