import json
import pytest
from agent_hpo.mcp_server import (
    handle_list_campaigns,
    handle_get_campaign_status,
    handle_get_round_summary,
    handle_get_campaign_history,
    handle_submit_action_proposal,
)
from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
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
