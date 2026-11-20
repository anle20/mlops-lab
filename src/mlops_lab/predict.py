"""Load a trained model and predict on new data.

Run with: python -m mlops_lab.predict --input examples/sample.csv

`load_model` and `predict` are what the FastAPI service (week 6) will call, so
the serving code does not need its own copy of this logic.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def validate_columns(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return df restricted to (and ordered as) the columns the model was trained on."""
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required column(s): {missing}")
    return df[expected]


def predict(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    X = validate_columns(model, df)
    return pd.DataFrame(
        {
            "prediction": model.predict(X),
            "probability": model.predict_proba(X)[:, 1],
        },
        index=X.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on a CSV file.")
    parser.add_argument("--input", required=True, help="CSV with the model's feature columns")
    parser.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args()

    result = predict(load_model(args.model), pd.read_csv(args.input))
    result.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
