import pytest
import optuna

from agent_hpo.scheduler import Scheduler
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, RoundSummary,
)


class TestBudgetClipping:
    def test_clips_to_remaining_trials(self):
        sc = StopConditions(max_total_trials=100, patience_rounds=3)
        effective = Scheduler.clip_budget(budget=50, cumulative_trials=80, stop_conditions=sc)
        assert effective == 20

    def test_returns_zero_when_budget_exhausted(self):
        sc = StopConditions(max_total_trials=100, patience_rounds=3)
        effective = Scheduler.clip_budget(budget=50, cumulative_trials=100, stop_conditions=sc)
        assert effective == 0

    def test_no_clip_when_no_cap(self):
        sc = StopConditions(patience_rounds=3)
        effective = Scheduler.clip_budget(budget=50, cumulative_trials=500, stop_conditions=sc)
        assert effective == 50


class TestStopConditionChecks:
    def test_target_score_fires_maximize(self):
        sc = StopConditions(target_score=0.95, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.96, direction="maximize",
                                         total_trials=10, wall_time=10.0) == "target_score"

    def test_target_score_fires_minimize(self):
        sc = StopConditions(target_score=0.05, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.04, direction="minimize",
                                         total_trials=10, wall_time=10.0) == "target_score"

    def test_target_score_not_met(self):
        sc = StopConditions(target_score=0.95, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.90, direction="maximize",
                                         total_trials=10, wall_time=10.0) is None

    def test_max_trials_fires(self):
        sc = StopConditions(max_total_trials=100, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.9, direction="maximize",
                                         total_trials=100, wall_time=10.0) == "max_total_trials"

    def test_max_wall_time_fires(self):
        sc = StopConditions(max_wall_time_seconds=3600.0, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.9, direction="maximize",
                                         total_trials=10, wall_time=3601.0) == "max_wall_time"

    def test_max_rounds_fires_after_completion(self):
        sc = StopConditions(max_rounds=5, patience_rounds=3)
        assert Scheduler.check_rounds_stop(sc, completed_rounds=5) == "max_rounds"

    def test_max_rounds_not_met(self):
        sc = StopConditions(max_rounds=5, patience_rounds=3)
        assert Scheduler.check_rounds_stop(sc, completed_rounds=4) is None


class TestPatienceCheck:
    def test_patience_fires(self):
        ic = ImprovementCriteria(mode="strict_better")
        summaries = [
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.90, round_completed_trials=10),
        ]
        assert Scheduler.check_patience(summaries, ic, "maximize", patience=3) is True

    def test_patience_not_fired(self):
        ic = ImprovementCriteria(mode="strict_better")
        summaries = [
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.91, round_completed_trials=10),
            RoundSummary(best_score=0.91, round_completed_trials=10),
        ]
        assert Scheduler.check_patience(summaries, ic, "maximize", patience=3) is False

    def test_patience_counts_zero_completed_as_no_improvement(self):
        ic = ImprovementCriteria(mode="strict_better")
        summaries = [
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.90, round_completed_trials=0),
            RoundSummary(best_score=0.90, round_completed_trials=0),
        ]
        assert Scheduler.check_patience(summaries, ic, "maximize", patience=2) is True
