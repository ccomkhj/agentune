import pytest
from agentune.core.state import (
    CampaignState,
    RoundState,
    InvalidTransitionError,
    validate_campaign_transition,
    validate_round_transition,
)


class TestCampaignState:
    def test_created_to_running(self):
        validate_campaign_transition(CampaignState.CREATED, CampaignState.RUNNING)

    def test_running_to_completed(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.COMPLETED)

    def test_running_to_pause_requested(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.PAUSE_REQUESTED)

    def test_pause_requested_to_paused(self):
        validate_campaign_transition(CampaignState.PAUSE_REQUESTED, CampaignState.PAUSED)

    def test_paused_to_running(self):
        validate_campaign_transition(CampaignState.PAUSED, CampaignState.RUNNING)

    def test_running_to_failed(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.FAILED)

    def test_running_to_stopped(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.STOPPED)

    def test_invalid_created_to_completed(self):
        with pytest.raises(InvalidTransitionError):
            validate_campaign_transition(CampaignState.CREATED, CampaignState.COMPLETED)

    def test_invalid_stopped_to_running(self):
        with pytest.raises(InvalidTransitionError):
            validate_campaign_transition(CampaignState.STOPPED, CampaignState.RUNNING)

    def test_invalid_completed_to_running(self):
        with pytest.raises(InvalidTransitionError):
            validate_campaign_transition(CampaignState.COMPLETED, CampaignState.RUNNING)

    def test_terminal_states(self):
        assert CampaignState.COMPLETED.is_terminal
        assert CampaignState.FAILED.is_terminal
        assert CampaignState.STOPPED.is_terminal
        assert not CampaignState.RUNNING.is_terminal


class TestRoundState:
    def test_proposed_to_running(self):
        validate_round_transition(RoundState.PROPOSED, RoundState.RUNNING)

    def test_running_to_summarizing(self):
        validate_round_transition(RoundState.RUNNING, RoundState.SUMMARIZING)

    def test_summarizing_to_awaiting_agent(self):
        validate_round_transition(RoundState.SUMMARIZING, RoundState.AWAITING_AGENT)

    def test_awaiting_agent_to_resolved(self):
        validate_round_transition(RoundState.AWAITING_AGENT, RoundState.RESOLVED)

    def test_awaiting_agent_to_closed(self):
        validate_round_transition(RoundState.AWAITING_AGENT, RoundState.CLOSED)

    def test_running_to_failed(self):
        validate_round_transition(RoundState.RUNNING, RoundState.FAILED)

    def test_summarizing_to_failed(self):
        validate_round_transition(RoundState.SUMMARIZING, RoundState.FAILED)

    def test_failed_to_retrying(self):
        validate_round_transition(RoundState.FAILED, RoundState.RETRYING)

    def test_retrying_to_running(self):
        validate_round_transition(RoundState.RETRYING, RoundState.RUNNING)

    def test_retrying_to_summarizing(self):
        validate_round_transition(RoundState.RETRYING, RoundState.SUMMARIZING)

    def test_invalid_proposed_to_resolved(self):
        with pytest.raises(InvalidTransitionError):
            validate_round_transition(RoundState.PROPOSED, RoundState.RESOLVED)

    def test_terminal_states(self):
        assert RoundState.RESOLVED.is_terminal
        assert RoundState.CLOSED.is_terminal
        assert not RoundState.RUNNING.is_terminal
