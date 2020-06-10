from helpers import name_from_path
from config import paper1_data_path


def test_name_from_path():
    assert name_from_path(os.path.join(paper1_data_path, "DGDBTC.csv")) == "DGDBTC.csv"
    assert (
        name_from_path(os.path.join(paper1_data_path, "STORMBTC.csv")) == "STORMBTC.csv"
    )

