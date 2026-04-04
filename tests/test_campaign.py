import json
import pytest
from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, ActionProposal,
)
from agent_hpo.core.state import CampaignState, RoundState


@pytest.fixture
def db(test_db_url):
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


@pytest.fixture
def service(db):
    return CampaignService(db)


@pytest.fixture
def sample_config():
    return CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=[
            ParamSpec(name="max_depth", type="int", low=1, high=15),
            ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
        ],
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=10, patience_rounds=3),
        trials_per_round=50,
    )


def _advance_round_to_awaiting(service, campaign_id, round_id, trial_end=50):
    """Helper to advance a round through RUNNING -> SUMMARIZING -> AWAITING_AGENT."""
    service.transition_round(round_id, RoundState.RUNNING)
    service.complete_round_execution(round_id, trial_end=trial_end)
    service.transition_round(round_id, RoundState.SUMMARIZING)
    service.write_summary(round_id, {"schema_version": 1})
    service.transition_round(round_id, RoundState.AWAITING_AGENT)


class TestCampaignCreation:
    def test_create_campaign(self, service, sample_config):
        campaign = service.create_campaign("test-campaign", sample_config)
        assert campaign["name"] == "test-campaign"
        assert campaign["state"] == "CREATED"

    def test_create_campaign_creates_round_1(self, service, sample_config):
        campaign = service.create_campaign("test-campaign", sample_config)
        rounds = service.get_rounds(campaign["id"])
        assert len(rounds) == 1
        assert rounds[0]["round_number"] == 1
        assert rounds[0]["state"] == "PROPOSED"
        assert rounds[0]["trial_offset"] == 0

    def test_duplicate_name_rejected(self, service, sample_config):
        service.create_campaign("test-campaign", sample_config)
        with pytest.raises(Exception):
            service.create_campaign("test-campaign", sample_config)


class TestStateTransitions:
    def test_transition_campaign_state(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        updated = service.get_campaign(campaign["id"])
        assert updated["state"] == "RUNNING"

    def test_invalid_transition_rejected(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        with pytest.raises(Exception):
            service.transition_campaign(campaign["id"], CampaignState.COMPLETED)

    def test_transition_round_state(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        rounds = service.get_rounds(campaign["id"])
        service.transition_round(rounds[0]["id"], RoundState.RUNNING)
        updated_round = service.get_round(rounds[0]["id"])
        assert updated_round["state"] == "RUNNING"


class TestProposalValidation:
    def test_continue_creates_round_reusing_study(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        r1 = service.get_rounds(campaign["id"])[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])

        proposal = ActionProposal(
            action="continue",
            justification="Score still improving",
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["round_number"] == 2
        assert r2["optuna_study_name"] == r1["optuna_study_name"]
        assert r2["trial_offset"] == 50

    def test_narrow_creates_new_study(self, service, sample_config):
        campaign = service.create_campaign("test-narrow", sample_config)
        r1 = service.get_rounds(campaign["id"])[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])

        proposal = ActionProposal(
            action="narrow_search",
            justification="Focus on learning_rate",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": True},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["optuna_study_name"] != r1["optuna_study_name"]
        assert r2["parent_round_id"] == r1["id"]
        assert r2["trial_offset"] == 0

    def test_widen_creates_new_study(self, service, sample_config):
        campaign = service.create_campaign("test-widen", sample_config)
        r1 = service.get_rounds(campaign["id"])[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])

        proposal = ActionProposal(
            action="widen_search",
            justification="Explore broader range for max_depth",
            proposed_search_space=[
                {"name": "max_depth", "type": "int", "low": 1, "high": 20},
                {"name": "learning_rate", "type": "float", "low": 0.0001, "high": 2.0, "log": True},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["optuna_study_name"] != r1["optuna_study_name"]
        assert r2["parent_round_id"] == r1["id"]
        assert r2["trial_offset"] == 0

    def test_increase_budget_reuses_study(self, service, sample_config):
        campaign = service.create_campaign("test-budget", sample_config)
        r1 = service.get_rounds(campaign["id"])[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])

        proposal = ActionProposal(
            action="increase_budget",
            justification="Need more trials, still converging",
            proposed_budget=100,
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["optuna_study_name"] == r1["optuna_study_name"]
        assert r2["budget"] == 100
        assert r2["trial_offset"] == 50

    def test_stop_does_not_create_round(self, service, sample_config):
        campaign = service.create_campaign("test-stop", sample_config)
        r1 = service.get_rounds(campaign["id"])[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])

        proposal = ActionProposal(
            action="stop",
            justification="Diminishing returns",
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        assert len(rounds) == 1

        updated = service.get_campaign(campaign["id"])
        assert updated["state"] == "COMPLETED"

    def test_cooldown_rejects_immediate_reversal(self, service, sample_config):
        campaign = service.create_campaign("test-cooldown", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)

        # Complete round 1
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])

        # Narrow in round 2
        narrow_proposal = ActionProposal(
            action="narrow_search",
            justification="Focus",
            proposed_search_space=[{"name": "max_depth", "type": "int", "low": 3, "high": 10}],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], narrow_proposal)
        assert result["accepted"] is True

        r1_updated = service.get_round(r1["id"])
        assert r1_updated["state"] == "RESOLVED"

        # Complete round 2
        r2 = service.get_rounds(campaign["id"])[1]
        _advance_round_to_awaiting(service, campaign["id"], r2["id"])

        # Immediately try to widen — should be rejected (cooldown)
        widen_proposal = ActionProposal(
            action="widen_search",
            justification="Expand",
            proposed_search_space=[{"name": "max_depth", "type": "int", "low": 1, "high": 20}],
            reference_round_ids=[r2["id"]],
        )
        result = service.submit_proposal(campaign["id"], widen_proposal)
        assert result["accepted"] is False
        assert "cooldown" in result["rejection_reason"].lower()
