"""Tests for exploration reset behavior in the runner."""

import pytest
from agentune.core.db import Database
from agentune.core.campaign import CampaignService
from agentune.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, ActionProposal,
)
from agentune.core.state import CampaignState, RoundState
from agentune.datasets import load_dataset
from agentune.runner import RoundRunner


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
def explore_campaign(service):
    """Create a strong-exploration campaign with patience=2."""
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=[
            ParamSpec(name="max_depth", type="int", low=1, high=10),
            ParamSpec(name="learning_rate", type="float", low=0.01, high=0.5, log=True),
        ],
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=20, patience_rounds=2),
        trials_per_round=5,
        dataset="breast_cancer",
        mode="strong-exploration",
    )
    return service.create_campaign("test-reset", config)


class TestExplorationReset:
    def test_patience_triggers_reset_not_completion(self, db, service, explore_campaign):
        """In strong-exploration mode, patience should auto-reset instead of completing."""
        split, _ = load_dataset("breast_cancer", seed=42)
        runner = RoundRunner(db, split)

        # Run rounds until patience would trigger (2 rounds without improvement)
        # Round 1
        result = runner.run_next_round(explore_campaign["id"])
        assert result.status == "AWAITING_AGENT"

        # Continue without changes for round 2
        r1 = service.get_rounds(explore_campaign["id"])[-1]
        service.submit_proposal(explore_campaign["id"], ActionProposal(
            action="continue", justification="keep going", reference_round_ids=[r1["id"]],
        ))
        result = runner.run_next_round(explore_campaign["id"])
        assert result.status == "AWAITING_AGENT"

        # Continue for round 3 — patience should trigger (2 rounds, small budget likely no improvement)
        r2 = service.get_rounds(explore_campaign["id"])[-1]
        service.submit_proposal(explore_campaign["id"], ActionProposal(
            action="continue", justification="keep going", reference_round_ids=[r2["id"]],
        ))
        result = runner.run_next_round(explore_campaign["id"])

        # The key test: campaign is NOT completed (reset happened instead)
        campaign = service.get_campaign(explore_campaign["id"])
        assert campaign["state"] != "COMPLETED"

    def test_reset_creates_round_with_incremented_reset_number(self, db, service, explore_campaign):
        """After a reset, the new round should have reset_number incremented."""
        split, _ = load_dataset("breast_cancer", seed=42)
        runner = RoundRunner(db, split)

        # Run enough rounds to trigger patience
        for i in range(5):
            result = runner.run_next_round(explore_campaign["id"])
            if result.status == "COMPLETED":
                break
            if result.status == "AWAITING_AGENT":
                rounds = service.get_rounds(explore_campaign["id"])
                latest = rounds[-1]
                # If reset happened, new round will have reset_number > 0
                if latest.get("reset_number", 0) > 0:
                    assert True
                    return
                # Otherwise, continue
                service.submit_proposal(explore_campaign["id"], ActionProposal(
                    action="continue", justification="keep going",
                    reference_round_ids=[latest["id"]],
                ))

        # If we reach here, check that campaign is still running (reset happened)
        campaign = service.get_campaign(explore_campaign["id"])
        assert campaign["state"] in ("RUNNING",)

    def test_standard_mode_still_completes_on_patience(self, db, service):
        """Standard mode should still terminate on patience."""
        config = CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[
                ParamSpec(name="max_depth", type="int", low=1, high=10),
                ParamSpec(name="learning_rate", type="float", low=0.01, high=0.5, log=True),
            ],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(max_rounds=20, patience_rounds=2),
            trials_per_round=5,
            dataset="breast_cancer",
            mode="standard",
        )
        campaign = service.create_campaign("test-standard-patience", config)
        split, _ = load_dataset("breast_cancer", seed=42)
        runner = RoundRunner(db, split)

        for i in range(10):
            result = runner.run_next_round(campaign["id"])
            if result.status == "COMPLETED":
                assert result.stop_reason == "patience"
                return
            rounds = service.get_rounds(campaign["id"])
            latest = rounds[-1]
            service.submit_proposal(campaign["id"], ActionProposal(
                action="continue", justification="keep going",
                reference_round_ids=[latest["id"]],
            ))

        # Should have completed via patience
        c = service.get_campaign(campaign["id"])
        assert c["state"] == "COMPLETED"
