"""Round orchestration: executes one study round end-to-end."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import optuna

from agentune.backends import get_backend
from agentune.core.campaign import CampaignService
from agentune.core.db import Database
from agentune.core.locking import LeaseManager
from agentune.core.models import (
    DatasetSplit,
    ImprovementCriteria,
    ParamSpec,
    RoundSummary,
    StopConditions,
)
from agentune.core.state import CampaignState, RoundState
from agentune.scheduler import Scheduler
from agentune.summarizer import RoundSummarizer


@dataclass
class RunResult:
    status: str  # AWAITING_AGENT, COMPLETED, FAILED
    round_number: int
    stop_reason: str | None = None


class RoundRunner:
    def __init__(self, db: Database, dataset: DatasetSplit) -> None:
        self._db = db
        self._dataset = dataset
        self._service = CampaignService(db)
        self._summarizer = RoundSummarizer()

    def _load_json(self, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _cumulative_wall_time(self, rounds: list[dict]) -> float:
        total = 0.0
        for round_row in rounds:
            summary = round_row.get("summary")
            if summary:
                total = self._load_json(summary).get("total_wall_time_seconds", total)
        return total

    def _previous_best_score(self, rounds: list[dict]) -> float | None:
        for round_row in reversed(rounds):
            summary = round_row.get("summary")
            if not summary:
                continue
            best_score = self._load_json(summary).get("best_score")
            if best_score is not None:
                return best_score
        return None

    def _round_summaries(self, rounds: list[dict]) -> list[RoundSummary]:
        summaries = []
        for round_row in rounds:
            summary = round_row.get("summary")
            if not summary:
                continue
            summaries.append(RoundSummary.from_dict(self._load_json(summary)))
        return summaries

    def _complete_without_execution(
        self,
        campaign_id: int,
        round_row: dict,
        stop_reason: str,
    ) -> RunResult:
        for state in (
            RoundState.RUNNING,
            RoundState.SUMMARIZING,
            RoundState.AWAITING_AGENT,
            RoundState.CLOSED,
        ):
            self._service.transition_round(round_row["id"], state)
        self._service.transition_campaign(
            campaign_id, CampaignState.COMPLETED,
            termination_reason=stop_reason,
        )
        return RunResult("COMPLETED", round_row["round_number"], stop_reason)

    def _complete_after_summary(
        self,
        campaign_id: int,
        round_id: int,
        round_number: int,
        stop_reason: str,
    ) -> RunResult:
        for state in (RoundState.AWAITING_AGENT, RoundState.CLOSED):
            self._service.transition_round(round_id, state)
        self._service.transition_campaign(
            campaign_id, CampaignState.COMPLETED,
            termination_reason=stop_reason,
        )
        return RunResult("COMPLETED", round_number, stop_reason)

    def run_next_round(self, campaign_id: int) -> RunResult:
        lease = LeaseManager(self._db)
        lease.acquire(campaign_id)
        try:
            return self._execute(campaign_id, lease)
        except Exception as exc:
            return self._handle_failure(campaign_id, exc)
        finally:
            with suppress(Exception):
                lease.release(campaign_id)

    def _handle_failure(self, campaign_id: int, exc: Exception) -> RunResult:
        """Mark round and campaign as FAILED with details."""
        detail = f"{type(exc).__name__}: {exc}"

        try:
            campaign = self._service.get_campaign(campaign_id)
            if campaign["state"] == "CREATED":
                self._service.transition_campaign(campaign_id, CampaignState.RUNNING)

            rounds = self._service.get_rounds(campaign_id)
            if rounds:
                current_round = rounds[-1]
                current_state = RoundState(current_round["state"])
                if not current_state.is_terminal and current_state != RoundState.FAILED:
                    self._service.transition_round(
                        current_round["id"], RoundState.FAILED,
                        failed_from=current_state.value,
                    )
                round_number = current_round["round_number"]
            else:
                round_number = 0

            self._service.transition_campaign(
                campaign_id, CampaignState.FAILED,
                termination_reason="failed",
                termination_detail=detail,
            )
        except Exception:
            pass  # best-effort — don't mask the original error

        return RunResult("FAILED", round_number, detail)

    def _execute(self, campaign_id: int, lease: LeaseManager) -> RunResult:
        campaign = self._service.get_campaign(campaign_id)
        stop_cond = StopConditions.from_dict(self._load_json(campaign["stop_conditions"]))
        improvement = ImprovementCriteria.from_dict(self._load_json(campaign["improvement_criteria"]))

        # Transition campaign to RUNNING if CREATED
        if campaign["state"] == "CREATED":
            self._service.transition_campaign(campaign_id, CampaignState.RUNNING)

        rounds = self._service.get_rounds(campaign_id)
        current_round = rounds[-1]

        if current_round["state"] != "PROPOSED":
            raise RuntimeError(f"Expected PROPOSED round, got {current_round['state']}")

        round_number = current_round["round_number"]

        # Pre-round budget clipping (trial cap)
        cumulative_trials = sum(r.get("trial_end", 0) or 0 for r in rounds[:-1])
        effective_budget = Scheduler.clip_budget(
            current_round["budget"], cumulative_trials, stop_cond
        )
        if effective_budget <= 0:
            return self._complete_without_execution(campaign_id, current_round, "max_total_trials")

        # Pre-round wall time check
        cumulative_wall = self._cumulative_wall_time(rounds[:-1])
        if stop_cond.max_wall_time_seconds and cumulative_wall >= stop_cond.max_wall_time_seconds:
            return self._complete_without_execution(campaign_id, current_round, "max_wall_time")

        # Compute Optuna timeout from remaining wall time
        optuna_timeout = None
        if stop_cond.max_wall_time_seconds:
            optuna_timeout = max(1.0, stop_cond.max_wall_time_seconds - cumulative_wall)

        # Setup Optuna study with persistent storage
        storage = optuna.storages.RDBStorage(self._db.optuna_storage_url)
        sampler_config = self._load_json(campaign["sampler_config"])
        sampler = optuna.samplers.TPESampler(seed=sampler_config.get("seed", 42))

        study_name = current_round["optuna_study_name"]
        is_new_study = current_round.get("parent_round_id") is not None
        if is_new_study:
            # Structural action (narrow/widen/revise) — need a fresh study.
            # Delete stale study from previous failed runs if it exists.
            with suppress(KeyError):
                optuna.delete_study(study_name=study_name, storage=storage)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=campaign["objective_direction"],
                sampler=sampler,
            )
        else:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
            except KeyError:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage,
                    direction=campaign["objective_direction"],
                    sampler=sampler,
                )

        # Create backend and objective
        backend_cls = get_backend(campaign["backend"])
        backend = backend_cls()
        search_space_raw = self._load_json(current_round["search_space"])
        search_space = [ParamSpec.from_dict(s) for s in search_space_raw]
        objective = backend.create_objective(self._dataset, campaign["metric_name"], search_space)

        # Run
        self._service.transition_round(current_round["id"], RoundState.RUNNING)
        lease.refresh(campaign_id)

        trial_offset = current_round["trial_offset"]
        study.optimize(objective, n_trials=effective_budget, timeout=optuna_timeout, show_progress_bar=False)
        trial_end = len(study.trials)

        self._service.complete_round_execution(current_round["id"], trial_end=trial_end)

        # Summarize
        self._service.transition_round(current_round["id"], RoundState.SUMMARIZING)
        summary = self._summarizer.summarize(
            study=study,
            campaign_id=campaign_id,
            round_id=current_round["id"],
            metric_name=campaign["metric_name"],
            objective_direction=campaign["objective_direction"],
            trial_offset=trial_offset,
            trial_end=trial_end,
            prev_best_score=self._previous_best_score(rounds[:-1]),
            parent_round_id=current_round.get("parent_round_id"),
            optuna_study_name=study_name,
            action_that_created="init" if round_number == 1 else "agent",
            cumulative_wall_time=cumulative_wall,
        )
        # Evaluate best params on held-out test set
        if summary.best_params:
            with suppress(Exception):
                test_score = backend.evaluate_test(
                    self._dataset, campaign["metric_name"], summary.best_params,
                )
                summary.test_score = test_score

        self._service.write_summary(current_round["id"], summary.to_dict())

        # Post-round hard stop check
        hard_stop = Scheduler.check_hard_stop(
            stop_cond, summary.best_score, campaign["objective_direction"],
            summary.total_trials, summary.total_wall_time_seconds,
        )
        completed_rounds = round_number
        if not hard_stop:
            rounds_stop = Scheduler.check_rounds_stop(stop_cond, completed_rounds)
            if rounds_stop:
                hard_stop = rounds_stop

        if hard_stop:
            return self._complete_after_summary(campaign_id, current_round["id"], round_number, hard_stop)

        # Patience check
        all_summaries = self._round_summaries(rounds)
        all_summaries.append(summary)

        if Scheduler.check_patience(all_summaries, improvement, campaign["objective_direction"], stop_cond.patience_rounds):
            return self._complete_after_summary(campaign_id, current_round["id"], round_number, "patience")

        # Check pause requested
        campaign = self._service.get_campaign(campaign_id)
        if campaign.get("pause_requested") or campaign["state"] == "PAUSE_REQUESTED":
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            # Transition from PAUSE_REQUESTED to PAUSED
            if campaign["state"] == "PAUSE_REQUESTED":
                self._service.transition_campaign(campaign_id, CampaignState.PAUSED)
            with self._db.connection() as conn:
                conn.execute(
                    "UPDATE campaigns SET pause_requested = false WHERE id = %s", (campaign_id,)
                )
            return RunResult("AWAITING_AGENT", round_number)

        self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
        return RunResult("AWAITING_AGENT", round_number)
