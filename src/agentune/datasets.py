"""Dataset loading with consistent train/val/test splits."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
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
    "store_sales": {"loader": "_load_ts", "metric": "rmse", "direction": "minimize", "temporal": True, "file": "store_sales.parquet"},
    "rossmann": {"loader": "_load_ts", "metric": "rmse", "direction": "minimize", "temporal": True, "file": "rossmann.parquet"},
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


def _get_data_path(filename: str) -> str:
    """Resolve data file path relative to the package root."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "data", filename),
        os.path.join("data", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Dataset file '{filename}' not found. Run: uv run python scripts/prepare_ts_datasets.py"
    )


def _make_mlforecast():
    """Create a configured MLForecast instance for feature engineering."""
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean

    return MLForecast(
        models=[],
        freq="D",
        lags=[1, 7, 14, 28],
        lag_transforms={
            1: [RollingMean(window_size=7), RollingMean(window_size=14)],
            7: [RollingMean(window_size=28)],
        },
        date_features=["dayofweek", "month", "day"],
    )


# Longest lag/window used — rows before each split boundary needed for warm-up
_WARMUP_DAYS = 28


def _preprocess_split(fcst, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess a single split's DataFrame and return (X, y).

    The DataFrame should include warm-up rows before the actual split period
    so lag/rolling features can be computed. Only rows from the actual split
    period (after warm-up) are returned.
    """
    prep = fcst.preprocess(df, static_features=[], dropna=True)
    feature_cols = [c for c in prep.columns if c not in ("unique_id", "ds", "y")]
    X = prep[feature_cols].to_numpy(dtype=np.float64)
    y = prep["y"].to_numpy(dtype=np.float64)
    return X, y


def _load_ts_dataset(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split temporally first, then preprocess each split independently.

    Each split gets warm-up rows from the preceding period so lag/rolling
    features are computed without data leakage.
    """
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Find global date boundaries for 60/20/20 split
    all_dates = df["ds"].sort_values().unique()
    n_dates = len(all_dates)
    train_cutoff = all_dates[int(n_dates * 0.6)]
    val_cutoff = all_dates[int(n_dates * 0.8)]
    warmup = pd.Timedelta(days=_WARMUP_DAYS)

    # Split raw data — each split includes warm-up rows from prior period
    train_raw = df[df["ds"] < train_cutoff]
    val_raw = df[(df["ds"] >= train_cutoff - warmup) & (df["ds"] < val_cutoff)]
    test_raw = df[df["ds"] >= val_cutoff - warmup]

    fcst = _make_mlforecast()

    # Preprocess each split independently — mlforecast only sees that split's data
    X_train, y_train = _preprocess_split(fcst, train_raw)
    X_val_full, y_val_full = _preprocess_split(fcst, val_raw)
    X_test_full, y_test_full = _preprocess_split(fcst, test_raw)

    # Trim warm-up rows from val and test (keep only rows >= cutoff)
    # After preprocessing + dropna, we need to re-derive which rows are in the
    # actual split period. The warm-up rows are the first ones; we trim based on
    # the expected count from train preprocessing.
    val_prep = fcst.preprocess(val_raw, static_features=[], dropna=True)
    val_in_period = val_prep["ds"] >= train_cutoff
    X_val = X_val_full[val_in_period.values]
    y_val = y_val_full[val_in_period.values]

    test_prep = fcst.preprocess(test_raw, static_features=[], dropna=True)
    test_in_period = test_prep["ds"] >= val_cutoff
    X_test = X_test_full[test_in_period.values]
    y_test = y_test_full[test_in_period.values]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_custom_dataset(
    path: str,
    target: str = "target",
    seed: int = 42,
) -> DatasetSplit:
    """Load a custom CSV or parquet file as a dataset.

    Args:
        path: Path to CSV or parquet file.
        target: Name of the target column.
        seed: Random seed for train/val/test split (60/20/20).

    Returns:
        DatasetSplit with 60/20/20 split.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {list(df.columns)}"
        )

    y = df[target].to_numpy()
    X = df.drop(columns=[target])

    # Encode string/categorical columns
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_arr = X.to_numpy(dtype=np.float64)

    # Fill NaNs with column median
    for col_idx in range(X_arr.shape[1]):
        mask = np.isnan(X_arr[:, col_idx])
        if mask.any():
            X_arr[mask, col_idx] = np.nanmedian(X_arr[:, col_idx])

    X_train, X_temp, y_train, y_temp = train_test_split(X_arr, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)


def _parse_custom_descriptor(name: str) -> tuple[str, dict]:
    """Parse 'path.csv:target=col:metric=rmse:direction=minimize' into (path, options)."""
    parts = name.split(":")
    path = parts[0]
    options = {}
    for part in parts[1:]:
        if "=" in part:
            key, val = part.split("=", 1)
            options[key] = val
    return path, options


def _is_file_path(name: str) -> bool:
    """Check if name looks like a file path (has extension or path separator)."""
    base = name.split(":")[0]  # strip descriptor suffix
    return "." in base and ("/" in base or os.path.exists(base))


def load_dataset(name: str, seed: int = 42) -> tuple[DatasetSplit, dict]:
    """Load a dataset with consistent splits. Returns (split, metadata).

    name can be:
      - A built-in name: 'breast_cancer', 'california_housing', etc.
      - A file path: '/path/to/data.csv' or '/path/to/data.parquet'
      - A descriptor: '/path/to/data.csv:target=label:metric=accuracy:direction=maximize'
    """
    if _is_file_path(name):
        path, options = _parse_custom_descriptor(name)
        target = options.get("target", "target")
        split = load_custom_dataset(path, target=target, seed=seed)
        return split, {
            "metric": options.get("metric"),
            "direction": options.get("direction"),
        }

    info = DATASETS.get(name)
    if info is None:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}")

    # Time-series datasets: split first, then preprocess (no data leakage)
    if info.get("temporal"):
        path = _get_data_path(info["file"])
        df = pd.read_parquet(path)
        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = pd.to_datetime(df["ds"])
        X_train, y_train, X_val, y_val, X_test, y_test = _load_ts_dataset(df)
    else:
        loader = info["loader"]
        if isinstance(loader, str):
            X, y = _CUSTOM_LOADERS[loader]()
        else:
            X, y = loader(return_X_y=True)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    split = DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)
    return split, {"metric": info["metric"], "direction": info["direction"]}
