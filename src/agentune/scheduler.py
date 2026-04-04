"""Round orchestration: budget clipping, stop conditions, execution control."""

from __future__ import annotations

from agentune.core.models import (
    ImprovementCriteria,
    RoundSummary,
    StopConditions,
)


class Scheduler:
    @staticmethod
    def clip_budget(
        budget: int,
        cumulative_trials: int,
        stop_conditions: StopConditions,
    ) -> int:
        if stop_conditions.max_total_trials is not None:
            remaining = stop_conditions.max_total_trials - cumulative_trials
            return max(0, min(budget, remaining))
        return budget

    @staticmethod
    def check_hard_stop(
        sc: StopConditions,
        best_score: float | None,
        direction: str,
        total_trials: int,
        wall_time: float,
    ) -> str | None:
        if sc.max_total_trials is not None and total_trials >= sc.max_total_trials:
            return "max_total_trials"
        if sc.max_wall_time_seconds is not None and wall_time >= sc.max_wall_time_seconds:
            return "max_wall_time"
        if sc.target_score is not None and best_score is not None:
            if direction == "maximize" and best_score >= sc.target_score:
                return "target_score"
            if direction == "minimize" and best_score <= sc.target_score:
                return "target_score"
        return None

    @staticmethod
    def check_rounds_stop(sc: StopConditions, completed_rounds: int) -> str | None:
        if sc.max_rounds is not None and completed_rounds >= sc.max_rounds:
            return "max_rounds"
        return None

    @staticmethod
    def check_patience(
        summaries: list[RoundSummary],
        improvement_criteria: ImprovementCriteria,
        direction: str,
        patience: int,
    ) -> bool:
        if len(summaries) < patience:
            return False

        recent = summaries[-patience:]
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]

            if curr.round_completed_trials == 0:
                continue

            if prev.best_score is None or curr.best_score is None:
                continue

            if improvement_criteria.is_improvement(curr.best_score, prev.best_score, direction):
                return False

        return True
