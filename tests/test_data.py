from mlops_lab.data import load_data, split_data


def test_load_data_shapes_match():
    X, y = load_data()
    assert len(X) == len(y)
    assert X.shape[1] > 0


def test_split_is_deterministic_and_stratified():
    X, y = load_data()
    a = split_data(X, y, seed=1)
    b = split_data(X, y, seed=1)
    assert a[0].index.equals(b[0].index)
    # class balance is preserved within 2 percentage points
    assert abs(a[2].mean() - y.mean()) < 0.02
