"""Tests for custom dataset loading."""

import numpy as np
import pandas as pd
import pytest

from agentune.datasets import load_custom_dataset, load_dataset
from agentune.core.models import DatasetSplit


class TestLoadCustomDataset:
    def test_loads_csv_with_target_column(self, tmp_path):
        df = pd.DataFrame({
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
            "label": np.random.randint(0, 2, 100),
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        split = load_custom_dataset(str(path), target="label", seed=42)
        assert isinstance(split, DatasetSplit)
        assert split.X_train.shape[1] == 2
        assert len(split.y_train) + len(split.y_val) + len(split.y_test) == 100

    def test_loads_parquet(self, tmp_path):
        df = pd.DataFrame({
            "x1": np.random.randn(80),
            "x2": np.random.randn(80),
            "y": np.random.randn(80),
        })
        path = tmp_path / "data.parquet"
        df.to_parquet(path, index=False)
        split = load_custom_dataset(str(path), target="y")
        assert split.X_train.shape[1] == 2

    def test_raises_on_missing_target(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Target column 'label' not found"):
            load_custom_dataset(str(path), target="label")

    def test_handles_nan_values(self, tmp_path):
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0, np.nan, 5.0] * 20,
            "target": list(range(100)),
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        split = load_custom_dataset(str(path), target="target")
        assert not np.isnan(split.X_train).any()

    def test_encodes_categorical_columns(self, tmp_path):
        df = pd.DataFrame({
            "color": ["red", "blue", "green"] * 40,
            "size": ["S", "M", "L"] * 40,
            "target": np.random.randint(0, 2, 120),
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        split = load_custom_dataset(str(path), target="target")
        assert split.X_train.dtype == np.float64

    def test_split_ratios_are_60_20_20(self, tmp_path):
        df = pd.DataFrame({
            "x": np.random.randn(1000),
            "target": np.random.randint(0, 2, 1000),
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        split = load_custom_dataset(str(path), target="target")
        total = len(split.y_train) + len(split.y_val) + len(split.y_test)
        assert total == 1000
        assert abs(len(split.y_train) / 1000 - 0.6) < 0.02
        assert abs(len(split.y_val) / 1000 - 0.2) < 0.02


class TestLoadDatasetWithCustomPath:
    def test_file_path_loads_custom_dataset(self, tmp_path):
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        path = tmp_path / "custom.csv"
        df.to_csv(path, index=False)
        split, meta = load_dataset(str(path), seed=42)
        assert isinstance(split, DatasetSplit)
        assert meta["metric"] is None
        assert meta["direction"] is None

    def test_file_path_with_descriptor_metadata(self, tmp_path):
        df = pd.DataFrame({
            "feat": np.random.randn(100),
            "label": np.random.randint(0, 2, 100),
        })
        path = tmp_path / "custom.csv"
        df.to_csv(path, index=False)
        descriptor = f"{path}:target=label:metric=accuracy:direction=maximize"
        split, meta = load_dataset(descriptor, seed=42)
        assert isinstance(split, DatasetSplit)
        assert meta["metric"] == "accuracy"
        assert meta["direction"] == "maximize"

    def test_builtin_dataset_still_works(self):
        split, meta = load_dataset("breast_cancer", seed=42)
        assert isinstance(split, DatasetSplit)
        assert meta["metric"] == "accuracy"

    def test_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")
