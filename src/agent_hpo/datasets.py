"""Dataset loading with consistent train/val/test splits."""

from __future__ import annotations

from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split

from agent_hpo.core.models import DatasetSplit

DATASETS = {
    "breast_cancer": {"loader": load_breast_cancer, "metric": "accuracy", "direction": "maximize"},
    "california_housing": {"loader": fetch_california_housing, "metric": "rmse", "direction": "minimize"},
    "digits": {"loader": load_digits, "metric": "accuracy", "direction": "maximize"},
}


def load_dataset(name: str, seed: int = 42) -> tuple[DatasetSplit, dict]:
    """Load a dataset with consistent splits. Returns (split, metadata)."""
    info = DATASETS[name]
    X, y = info["loader"](return_X_y=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    split = DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)
    return split, {"metric": info["metric"], "direction": info["direction"]}
