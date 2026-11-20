import pytest

from mlops_lab.config import TrainConfig
from mlops_lab.data import load_data
from mlops_lab.predict import load_model, predict
from mlops_lab.train import train


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    train(TrainConfig(output_dir=out))
    return load_model(out / "model.joblib")


def test_predict_returns_one_row_per_input(model):
    X, _ = load_data()
    result = predict(model, X.head(5))

    assert len(result) == 5
    assert set(result["prediction"]) <= {0, 1}
    assert result["probability"].between(0, 1).all()


def test_predict_ignores_extra_columns_and_column_order(model):
    X, _ = load_data()
    shuffled = X.head(3)[list(reversed(X.columns))].assign(extra=1)

    assert predict(model, shuffled)["prediction"].tolist() == (
        predict(model, X.head(3))["prediction"].tolist()
    )


def test_predict_rejects_missing_columns(model):
    X, _ = load_data()
    with pytest.raises(ValueError, match="Missing 1 required column"):
        predict(model, X.head(3).drop(columns=["mean radius"]))
