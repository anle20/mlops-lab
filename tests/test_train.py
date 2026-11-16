import json

import joblib

from mlops_lab.train import train


def test_train_writes_artifacts_and_meets_quality_floor(tmp_path):
    metrics = train(out_dir=tmp_path)

    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    # sanity floor: a broken pipeline would fall far below this
    assert metrics["accuracy"] > 0.9


def test_saved_model_predicts(tmp_path):
    train(out_dir=tmp_path)
    model = joblib.load(tmp_path / "model.joblib")

    from mlops_lab.data import load_data

    X, _ = load_data()
    pred = model.predict(X.head(5))
    assert len(pred) == 5
