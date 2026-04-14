import json
import pytest
from agentune.core.db import Database
from agentune.core.campaign import CampaignService
from agentune.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, ActionProposal,
)
from agentune.core.state import CampaignState, RoundState


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
        dataset="breast_cancer",
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


    def test_campaign_stores_mode(self, service, sample_config):
        campaign = service.create_campaign("test-mode", sample_config)
        retrieved = service.get_campaign(campaign["id"])
        assert retrieved["mode"] == "standard"

    def test_campaign_stores_strong_exploration_mode(self, service):
        config = CampaignConfig(
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
            dataset="breast_cancer",
            mode="strong-exploration",
        )
        campaign = service.create_campaign("test-explore", config)
        retrieved = service.get_campaign(campaign["id"])
        assert retrieved["mode"] == "strong-exploration"


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


class TestDatasetPersistence:
    def test_campaign_stores_dataset(self, service, sample_config):
        campaign = service.create_campaign("test-ds", sample_config)
        retrieved = service.get_campaign(campaign["id"])
        assert retrieved["dataset"] == "breast_cancer"
        assert retrieved["split_seed"] == 42

    def test_campaign_without_dataset_defaults_to_none(self, service):
        """CampaignConfig without dataset defaults to None."""
        config = CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[ParamSpec(name="max_depth", type="int", low=1, high=15)],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(patience_rounds=3),
            trials_per_round=50,
        )
        assert config.dataset is None


class TestTerminationMetadata:
    def test_completed_campaign_has_termination_reason(self, service, sample_config):
        campaign = service.create_campaign("test-term", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_campaign(
            campaign["id"], CampaignState.COMPLETED,
            termination_reason="patience", termination_detail="No improvement for 3 rounds",
        )
        updated = service.get_campaign(campaign["id"])
        assert updated["termination_reason"] == "patience"
        assert updated["termination_detail"] == "No improvement for 3 rounds"

    def test_failed_campaign_has_termination_reason(self, service, sample_config):
        campaign = service.create_campaign("test-fail", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_campaign(
            campaign["id"], CampaignState.FAILED,
            termination_reason="failed", termination_detail="RuntimeError: OOM",
        )
        updated = service.get_campaign(campaign["id"])
        assert updated["termination_reason"] == "failed"
        assert updated["termination_detail"] == "RuntimeError: OOM"

    def test_stopped_campaign_has_termination_reason(self, service, sample_config):
        campaign = service.create_campaign("test-stopped", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_campaign(
            campaign["id"], CampaignState.STOPPED,
            termination_reason="manual_stop",
        )
        updated = service.get_campaign(campaign["id"])
        assert updated["termination_reason"] == "manual_stop"


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


class TestReviseSearch:
    def _make_summary_with_plateau(self):
        """Return a summary dict indicating plateau and weak importance."""
        return {
            "schema_version": 1,
            "best_score": 0.95,
            "plateau_signal": True,
            "param_importance": {"max_depth": 0.05, "learning_rate": 0.05},
            "param_ranges_used": {},
            "new_best_in_round": False,
            "delta_from_prev": 0.0,
        }

    def test_revise_search_accepted_with_param_swap(self, service, sample_config):
        """revise_search should be accepted when it adds/drops params and eligibility met."""
        campaign = service.create_campaign("test-revise", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], self._make_summary_with_plateau())

        proposal = ActionProposal(
            action="revise_search",
            justification="Plateau detected, dropping max_depth, adding max_leaves",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.001, "high": 1.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

    def test_revise_search_rejected_no_structural_change(self, service, sample_config):
        """revise_search with same params (just range changes) should be rejected."""
        campaign = service.create_campaign("test-revise-nochange", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], self._make_summary_with_plateau())

        proposal = ActionProposal(
            action="revise_search",
            justification="Same params, different ranges",
            proposed_search_space=[
                {"name": "max_depth", "type": "int", "low": 2, "high": 10},
                {"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": True},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is False
        assert "must add or drop" in result["rejection_reason"].lower()

    def test_revise_search_rejected_too_much_churn(self, service):
        """revise_search swapping more than MAX_CHURN params should be rejected."""
        config = CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[
                ParamSpec(name="max_depth", type="int", low=1, high=15),
                ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
                ParamSpec(name="n_estimators", type="int", low=50, high=500),
                ParamSpec(name="min_child_weight", type="float", low=1.0, high=10.0),
                ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(patience_rounds=3),
            trials_per_round=50,
            dataset="breast_cancer",
        )
        campaign = service.create_campaign("test-churn", config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], {
            "schema_version": 1, "plateau_signal": True,
            "param_importance": {}, "param_ranges_used": {},
            "new_best_in_round": False,
        })

        # Drop 4 params, add 4 new ones = 8 swaps, exceeds limit
        proposal = ActionProposal(
            action="revise_search",
            justification="Total overhaul",
            proposed_search_space=[
                {"name": "gamma", "type": "float", "low": 0.0, "high": 5.0},
                {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10.0, "log": True},
                {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
                {"name": "max_bin", "type": "int", "low": 64, "high": 512},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is False
        assert "churn" in result["rejection_reason"].lower()

    def test_revise_search_validates_against_full_catalog(self, service, sample_config):
        """revise_search should accept params from available_params, not just defaults."""
        campaign = service.create_campaign("test-revise-catalog", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], {
            "schema_version": 1, "plateau_signal": True,
            "param_importance": {}, "param_ranges_used": {},
            "new_best_in_round": False,
        })

        proposal = ActionProposal(
            action="revise_search",
            justification="Adding max_leaves from extended catalog",
            proposed_search_space=[
                {"name": "max_depth", "type": "int", "low": 1, "high": 15},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

    def test_revise_search_resets_cooldown(self, service, sample_config):
        """After revise_search, narrow_search should be allowed immediately."""
        campaign = service.create_campaign("test-revise-cooldown", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)

        # Round 1: narrow_search
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], {
            "schema_version": 1, "plateau_signal": True,
            "param_importance": {}, "param_ranges_used": {},
            "new_best_in_round": False,
        })
        narrow = ActionProposal(
            action="narrow_search",
            justification="Focus",
            proposed_search_space=[{"name": "max_depth", "type": "int", "low": 3, "high": 10}],
            reference_round_ids=[r1["id"]],
        )
        service.submit_proposal(campaign["id"], narrow)

        # Round 2: revise_search (resets cooldown)
        r2 = service.get_rounds(campaign["id"])[1]
        _advance_round_to_awaiting(service, campaign["id"], r2["id"])
        service.write_summary(r2["id"], {
            "schema_version": 1, "plateau_signal": True,
            "param_importance": {}, "param_ranges_used": {},
            "new_best_in_round": False,
        })
        revise = ActionProposal(
            action="revise_search",
            justification="Fresh start",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.001, "high": 1.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
            ],
            reference_round_ids=[r2["id"]],
        )
        result = service.submit_proposal(campaign["id"], revise)
        assert result["accepted"] is True

        # Round 3: widen_search should be allowed (cooldown was reset by revise_search)
        r3 = service.get_rounds(campaign["id"])[2]
        _advance_round_to_awaiting(service, campaign["id"], r3["id"])
        widen = ActionProposal(
            action="widen_search",
            justification="Expand after revise",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.0001, "high": 2.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 512},
            ],
            reference_round_ids=[r3["id"]],
        )
        result = service.submit_proposal(campaign["id"], widen)
        assert result["accepted"] is True


class TestStrongExplorationMode:
    """Tests that strong-exploration mode relaxes guardrails."""

    @pytest.fixture
    def explore_config(self):
        return CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[
                ParamSpec(name="max_depth", type="int", low=1, high=15),
                ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
                ParamSpec(name="n_estimators", type="int", low=50, high=500),
                ParamSpec(name="min_child_weight", type="float", low=1.0, high=10.0),
                ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(max_rounds=10, patience_rounds=3),
            trials_per_round=50,
            dataset="breast_cancer",
            mode="strong-exploration",
        )

    def test_revise_search_allowed_without_plateau(self, service, explore_config):
        """In strong-exploration, revise_search is allowed even when round shows improvement."""
        campaign = service.create_campaign("test-explore-revise", explore_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], {
            "schema_version": 1,
            "best_score": 0.95,
            "plateau_signal": False,
            "param_importance": {"max_depth": 0.45, "learning_rate": 0.30},
            "param_ranges_used": {},
            "new_best_in_round": True,
            "delta_from_prev": 0.01,
        })

        proposal = ActionProposal(
            action="revise_search",
            justification="Exploring different param set despite improvement",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.001, "high": 1.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
                {"name": "max_bin", "type": "int", "low": 32, "high": 1024},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

    def test_high_churn_allowed(self, service, explore_config):
        """In strong-exploration, swapping more than 3 params is allowed."""
        campaign = service.create_campaign("test-explore-churn", explore_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], {
            "schema_version": 1, "plateau_signal": True,
            "param_importance": {}, "param_ranges_used": {},
            "new_best_in_round": False,
        })

        proposal = ActionProposal(
            action="revise_search",
            justification="Complete param overhaul",
            proposed_search_space=[
                {"name": "gamma", "type": "float", "low": 0.0, "high": 5.0},
                {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10.0, "log": True},
                {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
                {"name": "max_bin", "type": "int", "low": 32, "high": 1024},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

    def test_no_cooldown_between_reversals(self, service, explore_config):
        """In strong-exploration, no cooldown between narrow and widen."""
        campaign = service.create_campaign("test-explore-cooldown", explore_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)

        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        narrow = ActionProposal(
            action="narrow_search",
            justification="Focus",
            proposed_search_space=[
                {"name": "max_depth", "type": "int", "low": 3, "high": 10},
                {"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": True},
                {"name": "n_estimators", "type": "int", "low": 100, "high": 400},
                {"name": "min_child_weight", "type": "float", "low": 2.0, "high": 8.0},
                {"name": "subsample", "type": "float", "low": 0.6, "high": 0.9},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], narrow)
        assert result["accepted"] is True

        r2 = service.get_rounds(campaign["id"])[1]
        _advance_round_to_awaiting(service, campaign["id"], r2["id"])
        widen = ActionProposal(
            action="widen_search",
            justification="Expand back",
            proposed_search_space=[
                {"name": "max_depth", "type": "int", "low": 1, "high": 20},
                {"name": "learning_rate", "type": "float", "low": 0.0001, "high": 2.0, "log": True},
                {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
                {"name": "min_child_weight", "type": "float", "low": 1.0, "high": 10.0},
                {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            ],
            reference_round_ids=[r2["id"]],
        )
        result = service.submit_proposal(campaign["id"], widen)
        assert result["accepted"] is True

    def test_standard_mode_still_enforces_guardrails(self, service, sample_config):
        """Verify standard mode still rejects revise_search when improvement exists."""
        campaign = service.create_campaign("test-standard-guard", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        r1 = service.get_rounds(campaign["id"])[0]
        _advance_round_to_awaiting(service, campaign["id"], r1["id"])
        service.write_summary(r1["id"], {
            "schema_version": 1,
            "best_score": 0.95,
            "plateau_signal": False,
            "param_importance": {"max_depth": 0.45, "learning_rate": 0.30},
            "param_ranges_used": {},
            "new_best_in_round": True,
            "delta_from_prev": 0.01,
        })

        proposal = ActionProposal(
            action="revise_search",
            justification="Want to explore",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.001, "high": 1.0, "log": True},
                {"name": "max_leaves", "type": "int", "low": 0, "high": 256},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is False
        assert "not eligible" in result["rejection_reason"].lower()
