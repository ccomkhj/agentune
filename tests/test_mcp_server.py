import json
import pytest
from agentune.mcp_server import (
    handle_list_campaigns,
    handle_get_campaign_status,
    handle_get_round_summary,
    handle_get_campaign_history,
    handle_submit_action_proposal,
    handle_run_next_round,
)
from agentune.core.state import CampaignState, RoundState
from agentune.core.db import Database
from agentune.core.campaign import CampaignService
from agentune.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec,
)


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
def campaign_with_round(service):
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=[ParamSpec(name="max_depth", type="int", low=1, high=15)],
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(patience_rounds=3),
        trials_per_round=50,
        dataset="breast_cancer",
    )
    return service.create_campaign("mcp-test", config)


class TestMcpHandlers:
    def test_list_campaigns(self, db, campaign_with_round):
        result = handle_list_campaigns(db)
        assert len(result) >= 1
        assert any(c["name"] == "mcp-test" for c in result)

    def test_get_campaign_status(self, db, campaign_with_round):
        result = handle_get_campaign_status(db, "mcp-test")
        assert result["name"] == "mcp-test"
        assert result["state"] == "CREATED"

    def test_get_campaign_status_not_found(self, db):
        with pytest.raises(ValueError, match="not found"):
            handle_get_campaign_status(db, "nonexistent")

    def test_get_campaign_history(self, db, campaign_with_round):
        result = handle_get_campaign_history(db, "mcp-test")
        assert "rounds" in result
        assert len(result["rounds"]) == 1


class TestRunNextRound:
    def test_run_next_round_handler(self, db):
        """run_next_round should execute the next PROPOSED round and return a result."""
        service = CampaignService(db)
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
            stop_conditions=StopConditions(max_rounds=2, patience_rounds=3, max_total_trials=10),
            trials_per_round=5,
            dataset="breast_cancer",
        )
        service.create_campaign("mcp-run-test", config)
        result = handle_run_next_round(db, "mcp-run-test")
        assert result["status"] in ("AWAITING_AGENT", "COMPLETED", "FAILED")
        assert result["round_number"] == 1

    def test_run_next_round_not_found(self, db):
        with pytest.raises(ValueError, match="not found"):
            handle_run_next_round(db, "nonexistent")
