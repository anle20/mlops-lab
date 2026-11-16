"""Data loading and splitting.

Uses a dataset bundled with scikit-learn so it runs offline. Swap `load_data`
for your course project (e.g. Loan Approval) later: keep the same return shape
(a DataFrame of features and a Series of labels) and nothing else changes.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    ds = load_breast_cancer(as_frame=True)
    return ds.data, ds.target


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
