"""Tests for the report module's formatters, HTML helpers, and full generate_report."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentune.report import (
    _build_decision_card,
    _build_decision_context,
    _build_score_chart,
    _build_search_space_evolution,
    _build_space_change_html,
    _build_status_banner,
    _fmt_gen_gap,
    _fmt_pct,
    _fmt_score,
    _fmt_time,
    generate_report,
)


# ---------------------------------------------------------------------------
# TestFormatters
# ---------------------------------------------------------------------------

class TestFormatters:
    """Test _fmt_score, _fmt_time, _fmt_pct with None, zero, and valid values."""

    def test_fmt_score_none(self):
        assert _fmt_score(None) == "\u2014"

    def test_fmt_score_zero(self):
        assert _fmt_score(0.0) == "0.0000"

    def test_fmt_score_valid(self):
        assert _fmt_score(0.12345) == "0.1235"

    def test_fmt_score_custom_precision(self):
        assert _fmt_score(0.12345, precision=2) == "0.12"

    def test_fmt_time_none(self):
        assert _fmt_time(None) == "\u2014"

    def test_fmt_time_zero(self):
        assert _fmt_time(0) == "\u2014"

    def test_fmt_time_seconds(self):
        assert _fmt_time(42.3) == "42.3s"

    def test_fmt_time_minutes(self):
        assert _fmt_time(120.0) == "2.0m"

    def test_fmt_pct_none(self):
        assert _fmt_pct(None) == "\u2014"

    def test_fmt_pct_zero(self):
        assert _fmt_pct(0.0) == "0.0%"

    def test_fmt_pct_valid(self):
        assert _fmt_pct(0.123) == "12.3%"

    # _fmt_gen_gap tests

    def test_fmt_gen_gap_none(self):
        assert _fmt_gen_gap(None, 0.9) == "—"

    def test_fmt_gen_gap_no_best_score(self):
        """Falls back to _fmt_score when best_score is None."""
        assert _fmt_gen_gap(0.05, None) == "0.0500"

    def test_fmt_gen_gap_zero_best_score(self):
        """Falls back to _fmt_score when best_score is 0."""
        assert _fmt_gen_gap(0.05, 0.0) == "0.0500"

    def test_fmt_gen_gap_relative_small(self):
        """gap=0.02 on best_score=0.9 => 2.2%."""
        assert _fmt_gen_gap(0.02, 0.9) == "2.2%"

    def test_fmt_gen_gap_relative_large_rmse(self):
        """gap=46.16 on best_score=205.0 => 22.5% (the motivating bug case)."""
        assert _fmt_gen_gap(46.16, 205.0) == "22.5%"

    def test_fmt_gen_gap_negative_best_score(self):
        """Uses abs(best_score) for negative scores (e.g., neg_mse)."""
        assert _fmt_gen_gap(5.0, -50.0) == "10.0%"


# ---------------------------------------------------------------------------
# TestDecisionContext
# ---------------------------------------------------------------------------

class TestDecisionContext:
    """Test _build_decision_context with various summary/reasoning combos."""

    def test_empty_summary(self):
        assert _build_decision_context({}) == ""
        assert _build_decision_context({"summary": {}}) == ""
        assert _build_decision_context({"summary": None}) == ""

    def test_with_signals(self):
        d = {
            "summary": {
                "best_score": 0.95,
                "delta_from_prev": 0.01,
                "new_best_in_round": True,
                "plateau_signal": True,
                "generalization_gap": 0.05,
                "param_importance": {"lr": 0.4, "depth": 0.2},
            }
        }
        html = _build_decision_context(d)
        assert "decision-context" in html
        assert "0.9500" in html
        assert "+0.0100" in html
        assert "signal-good" in html  # new best
        assert "Plateau detected" in html
        assert "5.3%" in html  # gen gap: 0.05/0.95 = 5.3%
        assert "lr" in html

    def test_with_signals_no_new_best(self):
        d = {
            "summary": {
                "best_score": 0.9,
                "new_best_in_round": False,
            }
        }
        html = _build_decision_context(d)
        assert "signal-warn" in html
        assert "No new best" in html

    def test_with_structured_reasoning(self):
        d = {
            "summary": {"best_score": 0.9},
            "reasoning": {
                "observation": {
                    "best_score": 0.95,
                    "new_best_in_round": True,
                    "plateau_signal": False,
                    "top_params": [("lr", 0.5)],
                },
                "diagnosis": {
                    "reasons": ["Overfitting detected", "Need more regularization"],
                },
            },
        }
        html = _build_decision_context(d)
        assert "decision-context" in html
        assert "0.9500" in html
        assert "New best" in html
        assert "Overfitting detected" in html
        assert "Diagnosis" in html


# ---------------------------------------------------------------------------
# TestSpaceChangeHtml
# ---------------------------------------------------------------------------

class TestSpaceChangeHtml:
    """Test _build_space_change_html with various param change scenarios."""

    def test_no_proposed_space(self):
        assert _build_space_change_html({}) == ""
        assert _build_space_change_html({"proposed_search_space": None}) == ""

    def test_added_param(self):
        d = {
            "proposed_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
                {"name": "depth", "type": "int", "low": 3, "high": 10},
            ],
            "prev_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
            ],
        }
        html = _build_space_change_html(d)
        assert "space-added" in html
        assert "+ depth" in html

    def test_dropped_param(self):
        d = {
            "proposed_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
            ],
            "prev_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
                {"name": "depth", "type": "int", "low": 3, "high": 10},
            ],
        }
        html = _build_space_change_html(d)
        assert "space-dropped" in html
        assert "- depth" in html

    def test_changed_range(self):
        d = {
            "proposed_search_space": [
                {"name": "lr", "type": "float", "low": 0.001, "high": 0.05},
            ],
            "prev_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
            ],
        }
        html = _build_space_change_html(d)
        assert "space-table" in html
        assert "lr" in html
        # Old and new ranges present
        assert "0.01" in html
        assert "0.001" in html

    def test_no_changes(self):
        """Same space proposed -- no rows, empty string."""
        d = {
            "proposed_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
            ],
            "prev_search_space": [
                {"name": "lr", "type": "float", "low": 0.01, "high": 0.1},
            ],
        }
        assert _build_space_change_html(d) == ""

    def test_categorical_param_added(self):
        d = {
            "proposed_search_space": [
                {"name": "grow_policy", "type": "categorical", "choices": ["depthwise", "lossguide"]},
            ],
            "prev_search_space": [],
        }
        html = _build_space_change_html(d)
        assert "space-added" in html
        assert "grow_policy" in html


# ---------------------------------------------------------------------------
# TestStatusBanner
# ---------------------------------------------------------------------------

class TestStatusBanner:
    """Test _build_status_banner for running, completed, and failed states."""

    def _make_campaign(self, state, **kwargs):
        c = {"state": state, "stop_conditions": None}
        c.update(kwargs)
        return c

    def test_running(self):
        html = _build_status_banner(self._make_campaign("RUNNING"), [])
        assert "banner-running" in html
        assert "in progress" in html

    def test_completed(self):
        html = _build_status_banner(
            self._make_campaign("COMPLETED", termination_reason="patience exhausted"),
            [{"summary": {"best_score": 0.9}}],
        )
        assert "banner-completed" in html
        assert "completed" in html.lower()
        assert "patience exhausted" in html

    def test_failed(self):
        html = _build_status_banner(
            self._make_campaign("FAILED", termination_detail="Out of memory"),
            [],
        )
        assert "banner-failed" in html
        assert "failed" in html.lower()
        assert "Out of memory" in html

    def test_awaiting_agent(self):
        html = _build_status_banner(self._make_campaign("AWAITING_AGENT"), [])
        assert "banner-running" in html

    def test_stopped(self):
        html = _build_status_banner(self._make_campaign("STOPPED"), [])
        assert "banner-stopped" in html

    def test_with_max_rounds(self):
        campaign = self._make_campaign(
            "RUNNING",
            stop_conditions='{"max_rounds": 6}',
        )
        html = _build_status_banner(campaign, [{"summary": True}])
        assert "1" in html
        assert "of 6" in html


# ---------------------------------------------------------------------------
# TestScoreChart
# ---------------------------------------------------------------------------

class TestScoreChart:
    """Test _build_score_chart with various score lists."""

    def test_empty_scores(self):
        html = _build_score_chart([], [], "maximize")
        assert "No data yet" in html

    def test_all_none_scores(self):
        html = _build_score_chart([None, None], [None, None], "maximize")
        assert "No data yet" in html

    def test_with_scores(self):
        html = _build_score_chart([0.8, 0.85, 0.9], [0.78, 0.83, 0.88], "maximize")
        assert "bar-chart" in html
        assert "R1" in html
        assert "R2" in html
        assert "R3" in html
        assert "0.9000" in html
        assert "chart-legend" in html

    def test_with_none_test_scores(self):
        html = _build_score_chart([0.9, 0.95], [None, None], "maximize")
        assert "bar-chart" in html
        assert "0.9000" in html
        # No test bar for None test scores
        assert 'class="bar test"' not in html

    def test_single_score(self):
        html = _build_score_chart([0.5], [0.5], "minimize")
        assert "R1" in html


# ---------------------------------------------------------------------------
# TestSearchSpaceEvolution
# ---------------------------------------------------------------------------

class TestSearchSpaceEvolution:
    """Test _build_search_space_evolution with various round configurations."""

    def test_no_rounds(self):
        html = _build_search_space_evolution([])
        assert "No rounds" in html

    def test_single_round(self):
        rounds = [
            {
                "number": 1,
                "search_space": [
                    {"name": "lr"},
                    {"name": "depth"},
                ],
            }
        ]
        html = _build_search_space_evolution(rounds)
        assert "Round 1" in html
        assert "lr" in html
        assert "depth" in html
        # First round has no added/dropped
        assert "added" not in html
        assert "dropped" not in html

    def test_param_added_between_rounds(self):
        rounds = [
            {
                "number": 1,
                "search_space": [{"name": "lr"}],
            },
            {
                "number": 2,
                "search_space": [{"name": "lr"}, {"name": "depth"}],
            },
        ]
        html = _build_search_space_evolution(rounds)
        assert "Round 1" in html
        assert "Round 2" in html
        assert "added" in html
        assert "+ depth" in html

    def test_param_dropped_between_rounds(self):
        rounds = [
            {
                "number": 1,
                "search_space": [{"name": "lr"}, {"name": "depth"}],
            },
            {
                "number": 2,
                "search_space": [{"name": "lr"}],
            },
        ]
        html = _build_search_space_evolution(rounds)
        assert "dropped" in html
        assert "depth" in html


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """Test full generate_report with mocked CampaignService."""

    def _make_campaign(self):
        return {
            "id": 1,
            "name": "test-campaign",
            "state": "COMPLETED",
            "metric_name": "accuracy",
            "objective_direction": "maximize",
            "backend": "xgboost",
            "dataset": "breast_cancer",
            "stop_conditions": '{"max_rounds": 3}',
            "termination_reason": "patience exhausted",
            "created_at": "2026-01-01 12:00",
        }

    def _make_rounds(self):
        return [
            {
                "id": 10,
                "round_number": 1,
                "state": "COMPLETED",
                "optuna_study_name": "study_1",
                "budget": 40,
                "search_space": '[{"name": "lr", "type": "float", "low": 0.01, "high": 0.3}]',
                "summary": '{"best_score": 0.9, "test_score": 0.88, "delta_from_prev": null, '
                           '"round_best_score": 0.9, "new_best_in_round": true, "plateau_signal": false, '
                           '"generalization_gap": 0.02, "param_importance": {"lr": 0.8}, '
                           '"round_completed_trials": 40, "round_wall_time_seconds": 12.5, '
                           '"total_wall_time_seconds": 12.5, "total_trials": 40, '
                           '"best_params": {"lr": 0.05}}',
            },
            {
                "id": 11,
                "round_number": 2,
                "state": "COMPLETED",
                "optuna_study_name": "study_2",
                "budget": 40,
                "search_space": '[{"name": "lr", "type": "float", "low": 0.01, "high": 0.3}]',
                "summary": '{"best_score": 0.92, "test_score": 0.90, "delta_from_prev": 0.02, '
                           '"round_best_score": 0.92, "new_best_in_round": true, "plateau_signal": false, '
                           '"generalization_gap": 0.02, "param_importance": {"lr": 0.7}, '
                           '"round_completed_trials": 40, "round_wall_time_seconds": 13.0, '
                           '"total_wall_time_seconds": 25.5, "total_trials": 80, '
                           '"best_params": {"lr": 0.03}}',
            },
        ]

    def _make_decisions(self):
        return [
            {
                "round_id": 10,
                "action": "narrow_search",
                "accepted": True,
                "rejection_reason": None,
                "justification": "lr dominates importance; narrowing around best value.",
                "proposed_search_space": '[{"name": "lr", "type": "float", "low": 0.01, "high": 0.1}]',
                "reasoning": None,
            },
        ]

    @patch("agentune.report.CampaignService")
    def test_generate_report_produces_html(self, MockService):
        mock_svc = MagicMock()
        MockService.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = self._make_campaign()
        mock_svc.get_rounds.return_value = self._make_rounds()
        mock_svc.get_campaign_history.return_value = {
            "decisions": self._make_decisions(),
        }

        db = MagicMock()
        html = generate_report(db, "test-campaign")

        assert html.startswith("<!DOCTYPE html>")
        assert "test-campaign" in html
        assert "accuracy" in html
        assert "Score Progression" in html
        assert "Round Details" in html
        assert "Search Space Evolution" in html
        assert "Best Hyperparameters" in html
        assert "Decision Log" in html
        assert "0.9200" in html  # best score
        assert "narrow_search" in html

    @patch("agentune.report.CampaignService")
    def test_generate_report_campaign_not_found(self, MockService):
        mock_svc = MagicMock()
        MockService.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = None

        db = MagicMock()
        with pytest.raises(ValueError, match="not found"):
            generate_report(db, "nonexistent")

    @patch("agentune.report.CampaignService")
    def test_generate_report_no_decisions(self, MockService):
        mock_svc = MagicMock()
        MockService.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = self._make_campaign()
        mock_svc.get_rounds.return_value = self._make_rounds()
        mock_svc.get_campaign_history.return_value = {"decisions": []}

        db = MagicMock()
        html = generate_report(db, "test-campaign")
        assert "<!DOCTYPE html>" in html
        assert "Decision Log" in html


# ---------------------------------------------------------------------------
# TestDecisionLogStyling
# ---------------------------------------------------------------------------

class TestDecisionLogStyling:
    """Test that decision cards have action-specific color classes and icons."""

    def _make_decision(self, action="continue", accepted=True, rejection_reason=None):
        return {
            "round": 1,
            "action": action,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "justification": "Test justification",
            "reasoning": None,
            "proposed_search_space": None,
            "summary": {"best_score": 0.9, "param_importance": {}},
            "prev_search_space": [],
        }

    def test_continue_has_green_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("continue"))
        assert "action-continue" in html
        assert "\u25b6" in html  # ▶

    def test_narrow_search_has_accent_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("narrow_search"))
        assert "action-narrow_search" in html
        assert "\u25c1" in html  # ◁

    def test_revise_search_has_purple_class_and_highlight(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("revise_search"))
        assert "action-revise_search" in html
        assert "decision-highlight" in html
        assert "\u21bb" in html  # ↻

    def test_stop_has_red_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("stop"))
        assert "action-stop" in html
        assert "\u25a0" in html  # ■

    def test_rejected_keeps_rejected_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("narrow_search", accepted=False, rejection_reason="cooldown"))
        assert "rejected" in html
        assert "cooldown" in html
