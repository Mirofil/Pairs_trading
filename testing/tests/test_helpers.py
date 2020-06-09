def test_name_from_path():
    assert name_from_path("paper1/NEWconcatenated_price_data/DGDBTC.csv") == 'DGDBTC.csv'