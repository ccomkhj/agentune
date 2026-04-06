"""Unit tests for AgentReasoner._choose_action() decision paths.

All tests are pure Python — no database or external services required.
"""

from __future__ import annotations

import pytest

from agentune.agent import AgentReasoner
from agentune.core.models import RoundSummary


# ---------------------------------------------------------------------------
# XGBoost default 9-param search space
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE = [
    {"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": True},
    {"name": "max_depth", "type": "int", "low": 1, "high": 10},
    {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
    {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
    {"name": "colsample_bytree", "type": "float", "low": 0.5, "high": 1.0},
    {"name": "min_child_weight", "type": "float", "low": 1.0, "high": 10.0},
    {"name": "gamma", "type": "float", "low": 0.0, "high": 5.0},
    {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10.0, "log": True},
    {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10.0, "log": True},
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_summary(
    round_id: int = 1,
    campaign_id: int = 1,
    metric_name: str = "val_roc_auc",
    objective_direction: str = "maximize",
    best_score: float = 0.90,
    delta_from_prev: float | None = None,
    new_best_in_round: bool = True,
    plateau_signal: bool = False,
    failure_rate: float = 0.0,
    trials_added: int = 40,
    round_completed_trials: int = 40,
    param_importance: dict | None = None,
    best_params: dict | None = None,
    param_ranges_used: dict | None = None,
    generalization_gap: float | None = None,
) -> RoundSummary:
    """Build a mock RoundSummary with sensible defaults."""
    if param_importance is None:
        param_importance = {
            "learning_rate": 0.20,
            "max_depth": 0.15,
            "n_estimators": 0.10,
            "subsample": 0.08,
            "colsample_bytree": 0.07,
            "min_child_weight": 0.05,
            "gamma": 0.04,
            "reg_alpha": 0.03,
            "reg_lambda": 0.02,
        }
    if best_params is None:
        best_params = {
            "learning_rate": 0.1,
            "max_depth": 5,
            "n_estimators": 200,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 3.0,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    if param_ranges_used is None:
        param_ranges_used = {
            "learning_rate": (0.01, 0.5),
            "max_depth": (1, 10),
            "n_estimators": (50, 500),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "min_child_weight": (1.0, 10.0),
            "gamma": (0.0, 5.0),
            "reg_alpha": (1e-8, 10.0),
            "reg_lambda": (1e-8, 10.0),
        }
    return RoundSummary(
        round_id=round_id,
        campaign_id=campaign_id,
        metric_name=metric_name,
        objective_direction=objective_direction,
        best_score=best_score,
        delta_from_prev=delta_from_prev,
        new_best_in_round=new_best_in_round,
        plateau_signal=plateau_signal,
        failure_rate=failure_rate,
        trials_added=trials_added,
        round_completed_trials=round_completed_trials,
        param_importance=param_importance,
        best_params=best_params,
        param_ranges_used=param_ranges_used,
        generalization_gap=generalization_gap,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentReasonerChooseAction:

    def setup_method(self):
        self.reasoner = AgentReasoner()

    def test_high_failure_rate_stops(self):
        """failure_rate=0.15 exceeds FAILURE_THRESHOLD=0.10 → stop."""
        summary = _make_summary(failure_rate=0.15, new_best_in_round=False)
        decision = self.reasoner.decide(
            summary=summary,
            round_number=2,
            current_search_space=DEFAULT_SEARCH_SPACE,
        )
        assert decision.action == "stop"
        assert "failure" in decision.justification.lower()

    def test_round_1_narrows(self):
        """First round (round_number=1) with improvements → narrow_search."""
        summary = _make_summary(
            new_best_in_round=True,
            delta_from_prev=None,  # first round has no delta
        )
        decision = self.reasoner.decide(
            summary=summary,
            round_number=1,
            current_search_space=DEFAULT_SEARCH_SPACE,
        )
        assert decision.action == "narrow_search"

    def test_significant_improvement_continues(self):
        """delta/best > 1% relative improvement → continue."""
        best_score = 0.80
        # delta = 0.02 → 2.5% relative, which is > 1%
        delta = 0.02
        summary = _make_summary(
            best_score=best_score,
            delta_from_prev=delta,
            new_best_in_round=True,
        )
        decision = self.reasoner.decide(
            summary=summary,
            round_number=2,
            current_search_space=DEFAULT_SEARCH_SPACE,
        )
        assert decision.action == "continue"
        assert "significant" in decision.justification.lower() or "improv" in decision.justification.lower()

    def test_improving_with_plateau_increases_budget(self):
        """Improving + plateau_signal=True → increase_budget, new budget = trials_added + 20."""
        summary = _make_summary(
            best_score=0.90,
            delta_from_prev=0.001,  # small relative improvement (< 1%)
            new_best_in_round=True,
            plateau_signal=True,
            trials_added=40,
        )
        decision = self.reasoner.decide(
            summary=summary,
            round_number=2,
            current_search_space=DEFAULT_SEARCH_SPACE,
        )
        assert decision.action == "increase_budget"
        assert decision.proposed_budget == 60  # 40 + 20

    def test_small_improvement_continues(self):
        """delta/best < 1% relative (small) but still improving → continue."""
        best_score = 0.90
        # delta = 0.001 → ~0.11% relative, which is < 1%
        delta = 0.001
        summary = _make_summary(
            best_score=best_score,
            delta_from_prev=delta,
            new_best_in_round=True,
            plateau_signal=False,
        )
        decision = self.reasoner.decide(
            summary=summary,
            round_number=2,
            current_search_space=DEFAULT_SEARCH_SPACE,
        )
        assert decision.action == "continue"

    def test_no_improvement_narrows_when_no_recent_structural(self):
        """Not improving, no previous structural action → narrow_search."""
        summary = _make_summary(
            new_best_in_round=False,
            delta_from_prev=0.0,
        )
        decision = self.reasoner.decide(
            summary=summary,
            round_number=2,
            current_search_space=DEFAULT_SEARCH_SPACE,
            prev_summaries=[],
            prev_decisions=[],  # no structural action in history
        )
        assert decision.action == "narrow_search"

    def test_no_improvement_2_rounds_stops(self):
        """2 consecutive rounds with no improvement and no last structural → stop."""
        # Previous round also had no improvement
        prev_summary = _make_summary(
            round_id=1,
            new_best_in_round=False,
            delta_from_prev=0.0,
        )
        current_summary = _make_summary(
            round_id=2,
            new_best_in_round=False,
            delta_from_prev=0.0,
        )
        decision = self.reasoner.decide(
            summary=current_summary,
            round_number=3,
            current_search_space=DEFAULT_SEARCH_SPACE,
            prev_summaries=[prev_summary],
            prev_decisions=[],  # no narrow/widen in history
        )
        assert decision.action == "stop"

    def test_no_improvement_after_narrow_continues_once(self):
        """Not improving, but last structural was narrow_search and only 1 round since → continue."""
        current_summary = _make_summary(
            round_id=2,
            new_best_in_round=False,
            delta_from_prev=0.0,
            # Ensure no params are at boundary so widen_search is not triggered
            best_params={
                "learning_rate": 0.1,
                "max_depth": 5,
                "n_estimators": 200,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "min_child_weight": 5.0,
                "gamma": 2.5,
                "reg_alpha": 1.0,   # midpoint of (1e-8, 10.0) range — not at boundary
                "reg_lambda": 1.0,  # midpoint of (1e-8, 10.0) range — not at boundary
            },
        )
        prev_decisions = [
            {"action": "narrow_search", "accepted": True},
        ]
        decision = self.reasoner.decide(
            summary=current_summary,
            round_number=3,
            current_search_space=DEFAULT_SEARCH_SPACE,
            prev_summaries=[],  # only 1 round of no improvement
            prev_decisions=prev_decisions,
        )
        assert decision.action == "continue"
        assert "1 round" in decision.justification or "only" in decision.justification.lower()


class TestWidenSearch:

    def setup_method(self):
        self.reasoner = AgentReasoner()

    def test_widens_when_boundary_hit_after_narrow(self):
        """Not improving + last_structural=narrow + best params at boundary → widen_search."""
        # max_depth=15 at edge of [10, 15] — clearly at boundary
        narrowed_space = [
            {"name": "learning_rate", "type": "float", "low": 0.05, "high": 0.2, "log": True},
            {"name": "max_depth", "type": "int", "low": 10, "high": 15},
            {"name": "n_estimators", "type": "int", "low": 100, "high": 300},
        ]
        current_summary = _make_summary(
            round_id=3,
            new_best_in_round=False,
            delta_from_prev=0.0,
            best_params={
                "learning_rate": 0.1,
                "max_depth": 15,  # at upper boundary
                "n_estimators": 200,
            },
            param_ranges_used={
                "learning_rate": (0.05, 0.2),
                "max_depth": (10, 15),
                "n_estimators": (100, 300),
            },
            param_importance={
                "learning_rate": 0.25,
                "max_depth": 0.40,
                "n_estimators": 0.10,
            },
        )
        prev_decisions = [
            {"action": "narrow_search", "accepted": True},
        ]
        decision = self.reasoner.decide(
            summary=current_summary,
            round_number=3,
            current_search_space=narrowed_space,
            prev_summaries=[],
            prev_decisions=prev_decisions,
        )
        assert decision.action == "widen_search"
        assert "boundary" in decision.justification.lower() or "max_depth" in decision.justification

    def test_widen_expands_ranges(self):
        """Verify widened ranges are >= original ranges for boundary-hit params."""
        narrowed_space = [
            {"name": "learning_rate", "type": "float", "low": 0.05, "high": 0.2, "log": True},
            {"name": "max_depth", "type": "int", "low": 10, "high": 15},
        ]
        current_summary = _make_summary(
            round_id=3,
            new_best_in_round=False,
            delta_from_prev=0.0,
            best_params={
                "learning_rate": 0.1,
                "max_depth": 15,  # at upper boundary
            },
            param_ranges_used={
                "learning_rate": (0.05, 0.2),
                "max_depth": (10, 15),
            },
            param_importance={
                "learning_rate": 0.10,
                "max_depth": 0.40,
            },
        )
        prev_decisions = [
            {"action": "narrow_search", "accepted": True},
        ]
        decision = self.reasoner.decide(
            summary=current_summary,
            round_number=3,
            current_search_space=narrowed_space,
            prev_summaries=[],
            prev_decisions=prev_decisions,
        )
        assert decision.action == "widen_search"
        assert decision.proposed_search_space is not None

        # Build a map from name to new spec
        new_space_map = {s["name"]: s for s in decision.proposed_search_space}
        old_space_map = {s["name"]: s for s in narrowed_space}

        # max_depth is at boundary — its new range should be wider
        old_depth = old_space_map["max_depth"]
        new_depth = new_space_map["max_depth"]
        old_range = old_depth["high"] - old_depth["low"]
        new_range = new_depth["high"] - new_depth["low"]
        assert new_range >= old_range, f"Expected wider range for max_depth, got {new_range} <= {old_range}"

    def test_no_widen_when_no_boundary_hit(self):
        """Best params NOT at boundary + after narrow → continue (not widen)."""
        narrowed_space = [
            {"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": True},
            {"name": "max_depth", "type": "int", "low": 1, "high": 10},
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
        ]
        current_summary = _make_summary(
            round_id=3,
            new_best_in_round=False,
            delta_from_prev=0.0,
            best_params={
                "learning_rate": 0.1,   # well within [0.01, 0.5]
                "max_depth": 5,          # well within [1, 10]
                "n_estimators": 200,     # well within [50, 500]
            },
            param_ranges_used={
                "learning_rate": (0.01, 0.5),
                "max_depth": (1, 10),
                "n_estimators": (50, 500),
            },
            param_importance={
                "learning_rate": 0.25,
                "max_depth": 0.20,
                "n_estimators": 0.10,
            },
        )
        prev_decisions = [
            {"action": "narrow_search", "accepted": True},
        ]
        decision = self.reasoner.decide(
            summary=current_summary,
            round_number=3,
            current_search_space=narrowed_space,
            prev_summaries=[],  # only 1 round since narrow
            prev_decisions=prev_decisions,
        )
        # No boundary hit → should not widen; with only 1 round since narrow → continue
        assert decision.action != "widen_search"
        assert decision.action == "continue"
