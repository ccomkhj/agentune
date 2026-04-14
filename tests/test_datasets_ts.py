import pytest
import numpy as np
from agentune.datasets import load_dataset, DATASETS


class TestStoreSalesDataset:
    def test_registered_in_datasets(self):
        assert "store_sales" in DATASETS

    def test_load_returns_split_and_metadata(self):
        split, meta = load_dataset("store_sales")
        assert meta["metric"] == "rmse"
        assert meta["direction"] == "minimize"

    def test_split_shapes_consistent(self):
        split, _ = load_dataset("store_sales")
        assert split.X_train.shape[0] == split.y_train.shape[0]
        assert split.X_val.shape[0] == split.y_val.shape[0]
        assert split.X_test.shape[0] == split.y_test.shape[0]
        assert split.X_train.shape[1] == split.X_val.shape[1] == split.X_test.shape[1]

    def test_temporal_split_ratio(self):
        split, _ = load_dataset("store_sales")
        total = split.X_train.shape[0] + split.X_val.shape[0] + split.X_test.shape[0]
        train_ratio = split.X_train.shape[0] / total
        assert 0.5 < train_ratio < 0.7

    def test_no_nans_in_features(self):
        split, _ = load_dataset("store_sales")
        assert not np.isnan(split.X_train).any()
        assert not np.isnan(split.X_val).any()
        assert not np.isnan(split.X_test).any()


class TestRossmannDataset:
    def test_registered_in_datasets(self):
        assert "rossmann" in DATASETS

    def test_load_returns_split_and_metadata(self):
        split, meta = load_dataset("rossmann")
        assert meta["metric"] == "rmse"
        assert meta["direction"] == "minimize"

    def test_split_shapes_consistent(self):
        split, _ = load_dataset("rossmann")
        assert split.X_train.shape[0] == split.y_train.shape[0]
        assert split.X_val.shape[0] == split.y_val.shape[0]
        assert split.X_test.shape[0] == split.y_test.shape[0]
        assert split.X_train.shape[1] == split.X_val.shape[1] == split.X_test.shape[1]

    def test_has_more_features_than_store_sales(self):
        """Rossmann has exogenous features (DayOfWeek, Promo, SchoolHoliday)."""
        ss, _ = load_dataset("store_sales")
        ross, _ = load_dataset("rossmann")
        assert ross.X_train.shape[1] > ss.X_train.shape[1]

    def test_no_nans_in_features(self):
        split, _ = load_dataset("rossmann")
        assert not np.isnan(split.X_train).any()
        assert not np.isnan(split.X_val).any()
        assert not np.isnan(split.X_test).any()
