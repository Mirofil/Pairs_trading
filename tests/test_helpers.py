from pairs.helpers import name_from_path
from pairs.config import paper1_data_path
import os


def test_name_from_path():
    assert name_from_path(os.path.join(paper1_data_path, "DGDBTC.csv")) == "DGDBTC"
    assert (
        name_from_path(os.path.join(paper1_data_path, "STORMBTC.csv")) == "STORMBTC"
    )

