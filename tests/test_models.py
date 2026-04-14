import pytest
from agentune.core.models import (
    ParamSpec,
    CampaignConfig,
    StopConditions,
    ImprovementCriteria,
    RoundSummary,
    ActionProposal,
    DatasetSplit,
)
import numpy as np


class TestParamSpec:
    def test_float_param(self):
        p = ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True)
        assert p.name == "learning_rate"
        assert p.type == "float"
        assert p.log is True

    def test_int_param(self):
        p = ParamSpec(name="max_depth", type="int", low=1, high=15)
        assert p.type == "int"
        assert p.low == 1

    def test_categorical_param(self):
        p = ParamSpec(name="booster", type="categorical", choices=["gbtree", "dart"])
        assert p.choices == ["gbtree", "dart"]

    def test_float_param_requires_low_high(self):
        p = ParamSpec(name="lr", type="float", low=None, high=1.0)
        with pytest.raises(ValueError):
            p.validate()

    def test_categorical_requires_choices(self):
        p = ParamSpec(name="b", type="categorical", choices=None)
        with pytest.raises(ValueError):
            p.validate()


class TestImprovementCriteria:
    def test_strict_better_maximize(self):
        ic = ImprovementCriteria(mode="strict_better", threshold=0.0)
        assert ic.is_improvement(0.91, 0.90, "maximize") is True
        assert ic.is_improvement(0.90, 0.90, "maximize") is False

    def test_strict_better_minimize(self):
        ic = ImprovementCriteria(mode="strict_better", threshold=0.0)
        assert ic.is_improvement(0.89, 0.90, "minimize") is True
        assert ic.is_improvement(0.90, 0.90, "minimize") is False

    def test_min_absolute_delta(self):
        ic = ImprovementCriteria(mode="min_absolute_delta", threshold=0.01)
        assert ic.is_improvement(0.92, 0.90, "maximize") is True
        assert ic.is_improvement(0.905, 0.90, "maximize") is False

    def test_min_relative_delta(self):
        ic = ImprovementCriteria(mode="min_relative_delta", threshold=0.05)
        # 5% of 0.90 = 0.045, so 0.95 (delta 0.05) passes
        assert ic.is_improvement(0.95, 0.90, "maximize") is True
        assert ic.is_improvement(0.94, 0.90, "maximize") is False

    def test_min_relative_delta_zero_prev(self):
        """Falls back to absolute delta when prev_best is 0."""
        ic = ImprovementCriteria(mode="min_relative_delta", threshold=0.05)
        assert ic.is_improvement(0.06, 0.0, "maximize") is True
        assert ic.is_improvement(0.04, 0.0, "maximize") is False


class TestStopConditions:
    def test_serialization_roundtrip(self):
        sc = StopConditions(
            max_rounds=10,
            max_total_trials=500,
            max_wall_time_seconds=3600.0,
            patience_rounds=3,
            target_score=0.95,
        )
        d = sc.to_dict()
        sc2 = StopConditions.from_dict(d)
        assert sc == sc2

    def test_optional_fields_none(self):
        sc = StopConditions(
            max_rounds=None,
            max_total_trials=None,
            max_wall_time_seconds=None,
            patience_rounds=3,
            target_score=None,
        )
        assert sc.max_rounds is None


class TestActionProposal:
    def test_continue_valid(self):
        ap = ActionProposal(
            action="continue",
            justification="Score improved by 0.02 in round 3",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[3],
        )
        ap.validate()

    def test_narrow_requires_search_space(self):
        ap = ActionProposal(
            action="narrow_search",
            justification="Focus on high-importance params",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[3],
        )
        with pytest.raises(ValueError, match="proposed_search_space"):
            ap.validate()

    def test_increase_budget_requires_budget(self):
        ap = ActionProposal(
            action="increase_budget",
            justification="Need more trials",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[3],
        )
        with pytest.raises(ValueError, match="proposed_budget"):
            ap.validate()

    def test_empty_references_rejected(self):
        ap = ActionProposal(
            action="continue",
            justification="keep going",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[],
        )
        with pytest.raises(ValueError, match="reference_round_ids"):
            ap.validate()


class TestCampaignConfigMode:
    def test_default_mode_is_standard(self):
        config = CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[ParamSpec(name="max_depth", type="int", low=1, high=15)],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(patience_rounds=3),
            trials_per_round=50,
            dataset="breast_cancer",
        )
        assert config.mode == "standard"

    def test_mode_can_be_set_to_strong_exploration(self):
        config = CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[ParamSpec(name="max_depth", type="int", low=1, high=15)],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(patience_rounds=3),
            trials_per_round=50,
            dataset="breast_cancer",
            mode="strong-exploration",
        )
        assert config.mode == "strong-exploration"
