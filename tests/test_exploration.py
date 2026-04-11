import pytest
from agentune.core.models import ParamSpec
from agentune.exploration import select_exploration_params


class FakeBackend:
    """Minimal backend for testing param selection."""
    def available_params(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
            ParamSpec(name="n_estimators", type="int", low=50, high=500),
            ParamSpec(name="max_depth", type="int", low=1, high=15),
            ParamSpec(name="min_child_weight", type="float", low=1.0, high=10.0),
            ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ParamSpec(name="colsample_bytree", type="float", low=0.5, high=1.0),
            ParamSpec(name="gamma", type="float", low=0.0, high=5.0),
            ParamSpec(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="max_leaves", type="int", low=0, high=256),
            ParamSpec(name="max_bin", type="int", low=32, high=1024),
        ]


class TestSelectExplorationParams:
    def test_returns_9_params(self):
        backend = FakeBackend()
        result = select_exploration_params(backend, [])
        assert len(result) == 9

    def test_always_includes_learning_rate(self):
        backend = FakeBackend()
        result = select_exploration_params(backend, [])
        names = {p.name for p in result}
        assert "learning_rate" in names

    def test_prioritizes_untried_params(self):
        backend = FakeBackend()
        rounds = [
            {
                "search_space": [
                    {"name": "learning_rate"}, {"name": "n_estimators"},
                    {"name": "max_depth"}, {"name": "min_child_weight"},
                    {"name": "subsample"}, {"name": "colsample_bytree"},
                    {"name": "gamma"}, {"name": "reg_alpha"}, {"name": "reg_lambda"},
                ],
                "reset_number": 0,
            },
        ]
        result = select_exploration_params(backend, rounds)
        names = {p.name for p in result}
        assert "max_leaves" in names
        assert "max_bin" in names

    def test_returns_param_specs_from_catalog(self):
        backend = FakeBackend()
        result = select_exploration_params(backend, [])
        lr = next(p for p in result if p.name == "learning_rate")
        assert lr.low == 0.001
        assert lr.high == 1.0
        assert lr.log is True

    def test_small_catalog_returns_all(self):
        class SmallBackend:
            def available_params(self):
                return [
                    ParamSpec(name="learning_rate", type="float", low=0.01, high=1.0, log=True),
                    ParamSpec(name="n_estimators", type="int", low=50, high=500),
                    ParamSpec(name="max_depth", type="int", low=1, high=15),
                ]
        result = select_exploration_params(SmallBackend(), [])
        assert len(result) == 3
