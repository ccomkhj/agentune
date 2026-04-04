"""Round orchestration: executes one study round end-to-end."""

from __future__ import annotations

import json
from dataclasses import dataclass

import optuna

from agent_hpo.backends import get_backend
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.db import Database
from agent_hpo.core.locking import LeaseManager
from agent_hpo.core.models import (
    ImprovementCriteria,
    ParamSpec,
    RoundSummary,
    StopConditions,
    DatasetSplit,
)
from agent_hpo.core.state import CampaignState, RoundState
from agent_hpo.scheduler import Scheduler
from agent_hpo.summarizer import RoundSummarizer


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

    def _cumulative_wall_time(self, rounds: list[dict]) -> float:
        total = 0.0
        for r in rounds:
            s = r.get("summary")
            if s:
                if isinstance(s, str):
                    s = json.loads(s)
                total = s.get("total_wall_time_seconds", total)
        return total

    def run_next_round(self, campaign_id: int) -> RunResult:
        lease = LeaseManager(self._db)
        lease.acquire(campaign_id)
        try:
            return self._execute(campaign_id, lease)
        finally:
            try:
                lease.release(campaign_id)
            except Exception:
                pass

    def _execute(self, campaign_id: int, lease: LeaseManager) -> RunResult:
        campaign = self._service.get_campaign(campaign_id)
        stop_cond = StopConditions.from_dict(
            campaign["stop_conditions"] if isinstance(campaign["stop_conditions"], dict)
            else json.loads(campaign["stop_conditions"])
        )
        improvement = ImprovementCriteria.from_dict(
            campaign["improvement_criteria"] if isinstance(campaign["improvement_criteria"], dict)
            else json.loads(campaign["improvement_criteria"])
        )

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
            self._service.transition_round(current_round["id"], RoundState.RUNNING)
            self._service.transition_round(current_round["id"], RoundState.SUMMARIZING)
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, "max_total_trials")

        # Pre-round wall time check
        cumulative_wall = self._cumulative_wall_time(rounds[:-1])
        if stop_cond.max_wall_time_seconds and cumulative_wall >= stop_cond.max_wall_time_seconds:
            self._service.transition_round(current_round["id"], RoundState.RUNNING)
            self._service.transition_round(current_round["id"], RoundState.SUMMARIZING)
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, "max_wall_time")

        # Compute Optuna timeout from remaining wall time
        optuna_timeout = None
        if stop_cond.max_wall_time_seconds:
            optuna_timeout = max(1.0, stop_cond.max_wall_time_seconds - cumulative_wall)

        # Setup Optuna study with persistent storage
        storage = optuna.storages.RDBStorage(self._db.optuna_storage_url)
        sampler_config = campaign["sampler_config"]
        if isinstance(sampler_config, str):
            sampler_config = json.loads(sampler_config)
        sampler = optuna.samplers.TPESampler(seed=sampler_config.get("seed", 42))

        study_name = current_round["optuna_study_name"]
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
        search_space_raw = current_round["search_space"]
        if isinstance(search_space_raw, str):
            search_space_raw = json.loads(search_space_raw)
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
        prev_best = None
        for r in reversed(rounds[:-1]):
            if r.get("summary"):
                s = r["summary"]
                if isinstance(s, str):
                    s = json.loads(s)
                if s.get("best_score") is not None:
                    prev_best = s["best_score"]
                    break

        summary = self._summarizer.summarize(
            study=study,
            campaign_id=campaign_id,
            round_id=current_round["id"],
            metric_name=campaign["metric_name"],
            objective_direction=campaign["objective_direction"],
            trial_offset=trial_offset,
            trial_end=trial_end,
            prev_best_score=prev_best,
            parent_round_id=current_round.get("parent_round_id"),
            optuna_study_name=study_name,
            action_that_created="init" if round_number == 1 else "agent",
            cumulative_wall_time=cumulative_wall,
        )
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
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, hard_stop)

        # Patience check
        all_summaries = []
        for r in rounds:
            s = r.get("summary")
            if s:
                if isinstance(s, str):
                    s = json.loads(s)
                all_summaries.append(RoundSummary.from_dict(s))
        all_summaries.append(summary)

        if Scheduler.check_patience(all_summaries, improvement, campaign["objective_direction"], stop_cond.patience_rounds):
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, "patience")

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
