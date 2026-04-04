import pytest
import optuna
from agent_hpo.summarizer import RoundSummarizer
from agent_hpo.core.models import RoundSummary


def _make_study_with_trials(n_trials=10, direction="maximize"):
    """Create an Optuna study with completed trials for testing."""
    study = optuna.create_study(direction=direction)

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 10.0)
        y = trial.suggest_int("y", 1, 5)
        trial.set_user_attr("train_accuracy", x * y * 0.02)
        return x * y * 0.01

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


class TestRoundSummarizer:
    def test_basic_summary(self):
        study = _make_study_with_trials(10)
        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=1,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=0,
            trial_end=10,
            prev_best_score=None,
            parent_round_id=None,
            optuna_study_name="test_study",
            action_that_created="init",
            cumulative_wall_time=10.0,
        )
        assert isinstance(summary, RoundSummary)
        assert summary.trials_added == 10
        assert summary.best_score is not None
        assert summary.completed_trials > 0
        assert summary.schema_version == 1

    def test_summary_with_trial_boundaries(self):
        study = _make_study_with_trials(20)
        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=2,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=10,
            trial_end=20,
            prev_best_score=0.1,
            parent_round_id=1,
            optuna_study_name="test_study",
            action_that_created="continue",
            cumulative_wall_time=20.0,
        )
        assert summary.trials_added == 10
        assert summary.total_trials == 20
        assert summary.delta_from_prev is not None

    def test_summary_with_no_completed_trials(self):
        study = optuna.create_study(direction="maximize")
        for i in range(5):
            trial = study.ask()
            trial.suggest_float("x", 0.0, 10.0)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=1,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=0,
            trial_end=5,
            prev_best_score=None,
            parent_round_id=None,
            optuna_study_name="test",
            action_that_created="init",
            cumulative_wall_time=5.0,
        )
        assert summary.round_completed_trials == 0
        assert summary.round_best_score is None
        assert summary.new_best_in_round is False
        assert summary.pruned_rate == 1.0

    def test_convergence_curve_is_round_local(self):
        study = _make_study_with_trials(20)
        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=2,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=10,
            trial_end=20,
            prev_best_score=0.0,
            parent_round_id=None,
            optuna_study_name="test",
            action_that_created="continue",
            cumulative_wall_time=20.0,
        )
        if summary.convergence_curve:
            assert summary.convergence_curve[0][0] >= 0
