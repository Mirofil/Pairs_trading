from pairs.analysis import compute_cols_from_freq

def test_compute_cols():
    assert compute_cols_from_freq(["1D"], ["dist"]) == [['Daily'], ['dist']]