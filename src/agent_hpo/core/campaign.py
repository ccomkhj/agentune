"""Campaign and round management — the core service layer."""

from __future__ import annotations

import json
from typing import Any

from psycopg.rows import dict_row

from agent_hpo.core.db import Database
from agent_hpo.core.models import (
    ActionProposal,
    CampaignConfig,
    ParamSpec,
)
from agent_hpo.core.state import (
    CampaignState,
    RoundState,
    validate_campaign_transition,
    validate_round_transition,
)

COOLDOWN_ROUNDS = 2
STRUCTURAL_ACTIONS = {"narrow_search", "widen_search"}
OPPOSITE_ACTIONS = {"narrow_search": "widen_search", "widen_search": "narrow_search"}


class CampaignService:
    def __init__(self, db: Database) -> None:
        self._db = db

    # --- Campaign CRUD ---

    def create_campaign(self, name: str, config: CampaignConfig) -> dict:
        study_name = f"{name}_round_1"
        search_space_dicts = [p.to_dict() for p in config.initial_search_space]

        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO campaigns "
                "(name, metric_name, objective_direction, backend, sampler_config, "
                "initial_search_space, improvement_criteria, stop_conditions, trials_per_round) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    name,
                    config.metric_name,
                    config.objective_direction,
                    config.backend,
                    json.dumps(config.sampler_config),
                    json.dumps(search_space_dicts),
                    json.dumps(config.improvement_criteria.to_dict()),
                    json.dumps(config.stop_conditions.to_dict()),
                    config.trials_per_round,
                ),
            )
            campaign = cur.fetchone()

            # Create system-generated round 1
            conn.execute(
                "INSERT INTO study_rounds "
                "(campaign_id, round_number, optuna_study_name, search_space, budget, trial_offset) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    campaign["id"],
                    1,
                    study_name,
                    json.dumps(search_space_dicts),
                    config.trials_per_round,
                    0,
                ),
            )
        return campaign

    def get_campaign(self, campaign_id: int) -> dict:
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM campaigns WHERE id = %s", (campaign_id,)
            )
            return cur.fetchone()

    def get_rounds(self, campaign_id: int) -> list[dict]:
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM study_rounds WHERE campaign_id = %s ORDER BY round_number",
                (campaign_id,),
            )
            return cur.fetchall()

    def get_round(self, round_id: int) -> dict:
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM study_rounds WHERE id = %s", (round_id,)
            )
            return cur.fetchone()

    # --- State transitions ---

    def transition_campaign(self, campaign_id: int, to_state: CampaignState) -> None:
        campaign = self.get_campaign(campaign_id)
        from_state = CampaignState(campaign["state"])
        validate_campaign_transition(from_state, to_state)
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE campaigns SET state = %s, updated_at = now() WHERE id = %s",
                (to_state.value, campaign_id),
            )

    def transition_round(self, round_id: int, to_state: RoundState, failed_from: str | None = None) -> None:
        round_row = self.get_round(round_id)
        from_state = RoundState(round_row["state"])
        validate_round_transition(from_state, to_state)
        with self._db.connection() as conn:
            if to_state == RoundState.FAILED:
                conn.execute(
                    "UPDATE study_rounds SET state = %s, failed_from = %s, updated_at = now() WHERE id = %s",
                    (to_state.value, failed_from or from_state.value, round_id),
                )
            elif to_state == RoundState.RETRYING:
                conn.execute(
                    "UPDATE study_rounds SET state = %s, retry_count = retry_count + 1, updated_at = now() WHERE id = %s",
                    (to_state.value, round_id),
                )
            else:
                conn.execute(
                    "UPDATE study_rounds SET state = %s, updated_at = now() WHERE id = %s",
                    (to_state.value, round_id),
                )

    # --- Round execution helpers ---

    def complete_round_execution(self, round_id: int, trial_end: int) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE study_rounds SET trial_end = %s, updated_at = now() WHERE id = %s",
                (trial_end, round_id),
            )

    def write_summary(self, round_id: int, summary: dict) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE study_rounds SET summary = %s, summary_schema_version = %s, updated_at = now() WHERE id = %s",
                (json.dumps(summary), summary.get("schema_version", 1), round_id),
            )

    # --- Proposal validation and execution ---

    def submit_proposal(self, campaign_id: int, proposal: ActionProposal) -> dict:
        campaign = self.get_campaign(campaign_id)
        if campaign["state"] != "RUNNING":
            return self._record_decision(
                campaign_id, proposal, accepted=False,
                reason=f"Campaign is {campaign['state']}, proposals only accepted when RUNNING",
            )
        rounds = self.get_rounds(campaign_id)
        if not rounds or rounds[-1]["state"] != "AWAITING_AGENT":
            current_state = rounds[-1]["state"] if rounds else "no rounds"
            return self._record_decision(
                campaign_id, proposal, accepted=False,
                reason=f"Latest round is {current_state}, proposals only accepted when AWAITING_AGENT",
            )

        try:
            proposal.validate()
        except ValueError as e:
            return self._record_decision(campaign_id, proposal, accepted=False, reason=str(e))

        # Validate search space against backend param definitions
        if proposal.action in ("narrow_search", "widen_search") and proposal.proposed_search_space:
            from agent_hpo.backends import get_backend
            backend_cls = get_backend(campaign["backend"])
            backend = backend_cls()
            valid_names = {p.name for p in backend.param_definitions()}
            proposed_names = {p["name"] for p in proposal.proposed_search_space}
            invalid = proposed_names - valid_names
            if invalid:
                return self._record_decision(
                    campaign_id, proposal, accepted=False,
                    reason=f"Unknown params in proposed_search_space: {invalid}",
                )

        # Cooldown check
        if proposal.action in STRUCTURAL_ACTIONS:
            rejection = self._check_cooldown(campaign_id, proposal.action)
            if rejection:
                return self._record_decision(campaign_id, proposal, accepted=False, reason=rejection)

        # Get the current round (the one the agent is responding to)
        current_round = rounds[-1]

        if proposal.action == "stop":
            decision = self._record_decision(campaign_id, proposal, accepted=True)
            self.transition_round(current_round["id"], RoundState.RESOLVED)
            self.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return decision

        # Determine new round params
        if proposal.action in ("continue", "increase_budget"):
            new_study_name = current_round["optuna_study_name"]
            new_search_space = current_round["search_space"]
            trial_offset = current_round.get("trial_end", 0) or 0
            parent_id = None
            budget = proposal.proposed_budget if proposal.action == "increase_budget" else current_round["budget"]
        else:
            # narrow or widen — new Optuna study
            new_round_number = len(rounds) + 1
            new_study_name = f"{campaign['name']}_round_{new_round_number}"
            new_search_space = json.dumps(proposal.proposed_search_space)
            trial_offset = 0
            parent_id = current_round["id"]
            budget = current_round["budget"]

        # Mark current round as resolved
        self.transition_round(current_round["id"], RoundState.RESOLVED)

        # Create new round
        new_round_number = len(rounds) + 1
        with self._db.connection() as conn:
            search_space_val = new_search_space if isinstance(new_search_space, str) else json.dumps(new_search_space)
            conn.execute(
                "INSERT INTO study_rounds "
                "(campaign_id, round_number, optuna_study_name, search_space, budget, trial_offset, parent_round_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (campaign_id, new_round_number, new_study_name, search_space_val, budget, trial_offset, parent_id),
            )

        return self._record_decision(campaign_id, proposal, accepted=True)

    def _check_cooldown(self, campaign_id: int, action: str) -> str | None:
        opposite = OPPOSITE_ACTIONS.get(action)
        if not opposite:
            return None

        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT action, round_id FROM agent_decisions "
                "WHERE campaign_id = %s AND accepted = true AND action = ANY(%s) "
                "ORDER BY created_at DESC LIMIT 1",
                (campaign_id, list(STRUCTURAL_ACTIONS)),
            )
            last_structural = cur.fetchone()

        if not last_structural:
            return None

        if last_structural["action"] != opposite:
            return None

        rounds = self.get_rounds(campaign_id)
        decision_round_idx = None
        for i, r in enumerate(rounds):
            if r["id"] == last_structural["round_id"]:
                decision_round_idx = i
                break

        if decision_round_idx is None:
            return None

        created_round_idx = decision_round_idx + 1
        rounds_since = len(rounds) - 1 - created_round_idx
        if rounds_since < COOLDOWN_ROUNDS:
            return (
                f"Cooldown violation: {opposite} was applied {rounds_since} round(s) ago, "
                f"must wait {COOLDOWN_ROUNDS} rounds before reversing with {action}"
            )
        return None

    def _record_decision(
        self, campaign_id: int, proposal: ActionProposal, accepted: bool, reason: str | None = None
    ) -> dict:
        rounds = self.get_rounds(campaign_id)
        current_round_id = rounds[-1]["id"]
        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO agent_decisions "
                "(campaign_id, round_id, action, justification, proposed_search_space, "
                "proposed_budget, reference_round_ids, accepted, rejection_reason, reasoning) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    campaign_id,
                    current_round_id,
                    proposal.action,
                    proposal.justification,
                    json.dumps(proposal.proposed_search_space) if proposal.proposed_search_space else None,
                    proposal.proposed_budget,
                    json.dumps(proposal.reference_round_ids),
                    accepted,
                    reason,
                    json.dumps(proposal.reasoning) if proposal.reasoning else None,
                ),
            )
            return cur.fetchone()

    # --- Query helpers for MCP/CLI ---

    def get_campaign_history(self, campaign_id: int) -> dict:
        rounds = self.get_rounds(campaign_id)
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM agent_decisions WHERE campaign_id = %s ORDER BY created_at",
                (campaign_id,),
            )
            decisions = cur.fetchall()
        return {"rounds": rounds, "decisions": decisions}
