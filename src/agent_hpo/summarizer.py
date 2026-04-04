"""Produces immutable, schema-versioned round summaries from Optuna study data."""

from __future__ import annotations

from typing import Any

import optuna
from optuna.trial import TrialState

from agent_hpo.core.models import RoundSummary


class RoundSummarizer:
    def summarize(
        self,
        study: optuna.Study,
        campaign_id: int,
        round_id: int,
        metric_name: str,
        objective_direction: str,
        trial_offset: int,
        trial_end: int,
        prev_best_score: float | None,
        parent_round_id: int | None,
        optuna_study_name: str,
        action_that_created: str,
        cumulative_wall_time: float,
    ) -> RoundSummary:
        all_trials = study.trials
        round_trials = all_trials[trial_offset:trial_end]
        cumulative_trials = all_trials[:trial_end]

        # Classify round trials
        round_complete = [t for t in round_trials if t.state == TrialState.COMPLETE]
        round_failed = [t for t in round_trials if t.state == TrialState.FAIL]
        round_pruned = [t for t in round_trials if t.state == TrialState.PRUNED]
        cum_complete = [t for t in cumulative_trials if t.state == TrialState.COMPLETE]

        trials_added = len(round_trials)
        round_completed = len(round_complete)
        total_trials = len(cumulative_trials)
        completed_trials = len(cum_complete)

        # Rates
        failure_rate = len(round_failed) / trials_added if trials_added > 0 else 0.0
        pruned_rate = len(round_pruned) / trials_added if trials_added > 0 else 0.0

        # Best scores
        is_maximize = objective_direction == "maximize"

        def _best(trials):
            if not trials:
                return None, None
            if is_maximize:
                best = max(trials, key=lambda t: t.value)
            else:
                best = min(trials, key=lambda t: t.value)
            return best.value, best.params

        cum_best_score, cum_best_params = _best(cum_complete)
        round_best_score, _ = _best(round_complete)

        # Delta from prev
        delta = None
        if cum_best_score is not None and prev_best_score is not None:
            delta = cum_best_score - prev_best_score

        # New best in round
        new_best = False
        if round_best_score is not None and prev_best_score is not None:
            if is_maximize:
                new_best = round_best_score > prev_best_score
            else:
                new_best = round_best_score < prev_best_score
        elif round_best_score is not None and prev_best_score is None:
            new_best = True

        # Convergence curve (round-local)
        convergence = []
        running_best = None
        for i, t in enumerate(round_trials):
            if t.state != TrialState.COMPLETE:
                continue
            if running_best is None:
                running_best = t.value
            elif is_maximize:
                running_best = max(running_best, t.value)
            else:
                running_best = min(running_best, t.value)
            convergence.append((i, running_best))

        # Plateau signal
        plateau = False
        if round_completed > 0 and len(convergence) >= 3:
            cutoff = int(len(convergence) * 0.7)
            late_values = [v for _, v in convergence[cutoff:]]
            if late_values and late_values[0] == late_values[-1]:
                plateau = True

        # Parameter importance (best effort)
        param_importance = {}
        try:
            if completed_trials >= 4:
                importance = optuna.importance.get_param_importances(study)
                param_importance = dict(importance)
        except Exception:
            pass

        # Parameter ranges used in this round
        param_ranges = {}
        for t in round_complete:
            for name, val in t.params.items():
                if name not in param_ranges:
                    param_ranges[name] = (val, val)
                else:
                    lo, hi = param_ranges[name]
                    param_ranges[name] = (min(lo, val), max(hi, val))

        # Generalization gap
        gen_gap = None
        if round_complete:
            train_key = f"train_{metric_name}"
            gaps = []
            for t in round_complete:
                if train_key in t.user_attrs:
                    gaps.append(abs(t.value - t.user_attrs[train_key]))
            if gaps:
                gen_gap = sum(gaps) / len(gaps)

        # Wall time for this round
        round_wall = 0.0
        if round_trials:
            start_times = [t.datetime_start for t in round_trials if t.datetime_start]
            end_times = [t.datetime_complete for t in round_trials if t.datetime_complete]
            if start_times and end_times:
                round_wall = (max(end_times) - min(start_times)).total_seconds()

        return RoundSummary(
            schema_version=RoundSummary.CURRENT_SCHEMA_VERSION,
            round_id=round_id,
            campaign_id=campaign_id,
            metric_name=metric_name,
            objective_direction=objective_direction,
            best_score=cum_best_score,
            best_params=cum_best_params,
            delta_from_prev=delta,
            total_trials=total_trials,
            completed_trials=completed_trials,
            trials_added=trials_added,
            round_completed_trials=round_completed,
            new_best_in_round=new_best,
            round_best_score=round_best_score,
            convergence_curve=convergence,
            plateau_signal=plateau,
            param_importance=param_importance,
            param_ranges_used=param_ranges,
            generalization_gap=gen_gap,
            failure_rate=failure_rate,
            pruned_rate=pruned_rate,
            round_wall_time_seconds=round_wall,
            total_wall_time_seconds=cumulative_wall_time + round_wall,
            parent_round_id=parent_round_id,
            optuna_study_name=optuna_study_name,
            action_that_created_this_round=action_that_created,
        )
