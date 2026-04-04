"""Dataset loading with consistent train/val/test splits."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from agentune.core.models import DatasetSplit

DATASETS = {
    "breast_cancer": {"loader": load_breast_cancer, "metric": "accuracy", "direction": "maximize"},
    "california_housing": {"loader": fetch_california_housing, "metric": "rmse", "direction": "minimize"},
    "digits": {"loader": load_digits, "metric": "accuracy", "direction": "maximize"},
    "covertype": {"loader": "_load_covertype", "metric": "accuracy", "direction": "maximize"},
    "credit_g": {"loader": "_load_credit_g", "metric": "accuracy", "direction": "maximize"},
    "phoneme": {"loader": "_load_phoneme", "metric": "accuracy", "direction": "maximize"},
}


def _load_covertype() -> tuple[np.ndarray, np.ndarray]:
    """Covertype: 7-class forest cover, subsampled to 20k for speed."""
    from sklearn.datasets import fetch_covtype

    X, y = fetch_covtype(return_X_y=True)
    # Remap labels from 1-7 to 0-6 for XGBoost
    y = y - 1
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=20_000, replace=False)
    return X[idx], y[idx]


def _load_credit_g() -> tuple[np.ndarray, np.ndarray]:
    """German Credit: 1000 rows, imbalanced binary, mixed types."""
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("credit-g", version=1, return_X_y=True, as_frame=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # Encode categoricals and fill NaNs
    import pandas as pd

    for col in X.columns:
        if X[col].dtype == "category" or X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X_arr = X.to_numpy(dtype=np.float64)
    # Fill any remaining NaNs with column median
    for col_idx in range(X_arr.shape[1]):
        mask = np.isnan(X_arr[:, col_idx])
        if mask.any():
            X_arr[mask, col_idx] = np.nanmedian(X_arr[:, col_idx])
    return X_arr, y_enc


def _load_phoneme() -> tuple[np.ndarray, np.ndarray]:
    """Phoneme: noisy speech classification, 5404 rows."""
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("phoneme", version=1, return_X_y=True, as_frame=False)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


_CUSTOM_LOADERS = {
    "_load_covertype": _load_covertype,
    "_load_credit_g": _load_credit_g,
    "_load_phoneme": _load_phoneme,
}


def load_dataset(name: str, seed: int = 42) -> tuple[DatasetSplit, dict]:
    """Load a dataset with consistent splits. Returns (split, metadata)."""
    info = DATASETS[name]
    loader = info["loader"]

    if isinstance(loader, str):
        X, y = _CUSTOM_LOADERS[loader]()
    else:
        X, y = loader(return_X_y=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    split = DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)
    return split, {"metric": info["metric"], "direction": info["direction"]}
