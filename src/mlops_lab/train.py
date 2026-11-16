"""Train a baseline model and save the model plus its metrics.

Run with: make train
"""

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_lab.data import load_data, split_data

ARTIFACT_DIR = Path("artifacts")


def build_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


def train(c: float = 1.0, seed: int = 42, out_dir: Path = ARTIFACT_DIR) -> dict[str, float]:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)

    model = build_pipeline(c).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
