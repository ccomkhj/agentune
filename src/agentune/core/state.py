"""Campaign and round state machines with validated transitions."""

from __future__ import annotations

from enum import Enum


class InvalidTransitionError(Exception):
    pass


class CampaignState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"

    @property
    def is_terminal(self) -> bool:
        return self in (self.COMPLETED, self.FAILED, self.STOPPED)


CAMPAIGN_TRANSITIONS: dict[CampaignState, set[CampaignState]] = {
    CampaignState.CREATED: {CampaignState.RUNNING},
    CampaignState.RUNNING: {
        CampaignState.PAUSE_REQUESTED,
        CampaignState.COMPLETED,
        CampaignState.FAILED,
        CampaignState.STOPPED,
    },
    CampaignState.PAUSE_REQUESTED: {CampaignState.PAUSED},
    CampaignState.PAUSED: {CampaignState.RUNNING},
    CampaignState.COMPLETED: set(),
    CampaignState.FAILED: set(),
    CampaignState.STOPPED: set(),
}


def validate_campaign_transition(
    from_state: CampaignState, to_state: CampaignState
) -> None:
    allowed = CAMPAIGN_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidTransitionError(
            f"Campaign transition {from_state.value} → {to_state.value} is not allowed. "
            f"Allowed from {from_state.value}: {[s.value for s in allowed]}"
        )


class RoundState(Enum):
    PROPOSED = "PROPOSED"
    RUNNING = "RUNNING"
    SUMMARIZING = "SUMMARIZING"
    AWAITING_AGENT = "AWAITING_AGENT"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"

    @property
    def is_terminal(self) -> bool:
        return self in (self.RESOLVED, self.CLOSED)


ROUND_TRANSITIONS: dict[RoundState, set[RoundState]] = {
    RoundState.PROPOSED: {RoundState.RUNNING, RoundState.FAILED},
    RoundState.RUNNING: {RoundState.SUMMARIZING, RoundState.FAILED},
    RoundState.SUMMARIZING: {RoundState.AWAITING_AGENT, RoundState.FAILED},
    RoundState.AWAITING_AGENT: {RoundState.RESOLVED, RoundState.CLOSED},
    RoundState.RESOLVED: set(),
    RoundState.CLOSED: set(),
    RoundState.FAILED: {RoundState.RETRYING},
    RoundState.RETRYING: {RoundState.RUNNING, RoundState.SUMMARIZING},
}


def validate_round_transition(
    from_state: RoundState, to_state: RoundState
) -> None:
    allowed = ROUND_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidTransitionError(
            f"Round transition {from_state.value} → {to_state.value} is not allowed. "
            f"Allowed from {from_state.value}: {[s.value for s in allowed]}"
        )
