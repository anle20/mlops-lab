"""Train a baseline model and save the model plus its metrics.

Run with: make train   (or: python -m mlops_lab.train --config configs/train.yaml)
"""

import argparse
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_lab.config import TrainConfig, load_config
from mlops_lab.data import load_data, split_data


def build_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


def train(cfg: TrainConfig | None = None) -> dict[str, float]:
    cfg = cfg or TrainConfig()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg.data.test_size, seed=cfg.data.seed
    )

    model = build_pipeline(cfg.model.c).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.output_dir / "model.joblib")
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
