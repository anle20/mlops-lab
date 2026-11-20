import json

import joblib

from mlops_lab.config import DataConfig, ModelConfig, TrainConfig
from mlops_lab.train import train


def test_train_writes_artifacts_and_meets_quality_floor(tmp_path):
    metrics = train(TrainConfig(output_dir=tmp_path))

    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    # sanity floor: a broken pipeline would fall far below this
    assert metrics["accuracy"] > 0.9


def test_config_changes_the_trained_model(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    train(TrainConfig(model=ModelConfig(c=1.0), output_dir=a))
    train(TrainConfig(model=ModelConfig(c=0.001), output_dir=b))

    coef_a = joblib.load(a / "model.joblib").named_steps["clf"].coef_
    coef_b = joblib.load(b / "model.joblib").named_steps["clf"].coef_
    # stronger regularization (smaller C) must shrink the weights
    assert abs(coef_b).sum() < abs(coef_a).sum()


def test_same_config_gives_same_metrics(tmp_path):
    cfg = TrainConfig(data=DataConfig(seed=7), output_dir=tmp_path)
    assert train(cfg) == train(cfg)
