from pairs.helpers import name_from_path
from pairs.config import paper1_data_path
import os


def test_name_from_path():
    assert name_from_path(os.path.join(paper1_data_path, "DGDBTC.csv")) == "DGDBTC"
    assert (
        name_from_path(os.path.join(paper1_data_path, "STORMBTC.csv")) == "STORMBTC"
    )

def test_filter_by_period():
    parent_dir = os.path.dirname(__file__)
    reference = pd.read_parquet(os.path.join(parent_dir, "dist_signal_reference.parquet"))