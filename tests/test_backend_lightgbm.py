import pytest
import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from agentune.backends.lightgbm import LightGBMBackend
from agentune.core.models import ParamSpec, DatasetSplit


@pytest.fixture
def dataset():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)


class TestParamCatalog:
    def test_available_params_is_superset_of_default(self):
        backend = LightGBMBackend()
        available_names = {p.name for p in backend.available_params()}
        default_names = {p.name for p in backend.default_search_space()}
        assert default_names.issubset(available_names)

    def test_available_params_includes_extended_params(self):
        backend = LightGBMBackend()
        available_names = {p.name for p in backend.available_params()}
        assert "min_split_gain" in available_names
        assert "max_bin" in available_names
        assert "scale_pos_weight" in available_names

    def test_default_search_space_is_smaller_than_catalog(self):
        backend = LightGBMBackend()
        assert len(backend.default_search_space()) < len(backend.available_params())


class TestLightGBMBackend:
    def test_default_search_space(self):
        backend = LightGBMBackend()
        space = backend.default_search_space()
        assert len(space) > 0
        names = {p.name for p in space}
        assert "num_leaves" in names
        assert "learning_rate" in names

    def test_create_objective_returns_callable(self, dataset):
        backend = LightGBMBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        assert callable(objective)

    def test_objective_runs_and_returns_float(self, dataset):
        backend = LightGBMBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2, show_progress_bar=False)
        assert len(study.trials) == 2
        assert all(t.value is not None for t in study.trials)

    def test_objective_logs_train_metric(self, dataset):
        backend = LightGBMBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        assert "train_accuracy" in study.trials[0].user_attrs

    def test_evaluate_test_returns_score_in_range(self, dataset):
        backend = LightGBMBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        best_params = study.best_trial.params
        score = backend.evaluate_test(dataset, "accuracy", best_params)
        assert 0.0 <= score <= 1.0

    def test_tuning_guide_has_required_fields(self):
        backend = LightGBMBackend()
        guide = backend.tuning_guide()
        assert guide.backend_name == "lightgbm"
        assert len(guide.overview) > 0
        assert len(guide.params) > 0
        assert len(guide.diagnostics) > 0
        assert len(guide.tuning_order) > 0

    def test_tuning_guide_to_dict_roundtrip(self):
        backend = LightGBMBackend()
        guide = backend.tuning_guide()
        d = guide.to_dict()
        assert isinstance(d, dict)
        assert d["backend"] == "lightgbm"
        assert "params" in d
        assert "diagnostics" in d
        assert "tuning_order" in d
        # Verify params roundtrip
        param_names = {p["name"] for p in d["params"]}
        assert "num_leaves" in param_names
        assert "learning_rate" in param_names
