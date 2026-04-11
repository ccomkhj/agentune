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


class TestModeInResponse:
    def test_get_campaign_status_includes_mode(self, db, service):
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
            mode="strong-exploration",
        )
        service.create_campaign("test-mode-mcp", config)
        result = handle_get_campaign_status(db, "test-mode-mcp")
        assert result["mode"] == "strong-exploration"


class TestTestScoreStripping:
    """test_score must be stripped from MCP responses while a campaign is active."""

    SUMMARY_WITH_TEST_SCORE = {
        "schema_version": 1,
        "best_score": 0.95,
        "test_score": 0.93,
        "param_importance": {"max_depth": 0.6},
        "plateau_signal": False,
    }

    @pytest.fixture
    def active_campaign(self, service, db):
        """Create a campaign in RUNNING state with a round that has a summary containing test_score."""
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
        campaign = service.create_campaign("strip-test", config)
        # Transition campaign to RUNNING
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        # Transition round through to AWAITING_AGENT and write summary
        rounds = service.get_rounds(campaign["id"])
        round_id = rounds[0]["id"]
        service.transition_round(round_id, RoundState.RUNNING)
        service.transition_round(round_id, RoundState.SUMMARIZING)
        service.write_summary(round_id, self.SUMMARY_WITH_TEST_SCORE)
        service.transition_round(round_id, RoundState.AWAITING_AGENT)
        return campaign

    def test_get_round_summary_strips_test_score_for_active(self, db, active_campaign):
        result = handle_get_round_summary(db, "strip-test")
        summary = result["summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        assert "test_score" not in summary
        assert summary["best_score"] == 0.95

    def test_get_campaign_status_strips_test_score_for_active(self, db, active_campaign):
        result = handle_get_campaign_status(db, "strip-test")
        summary = result["latest_round"]["summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        assert "test_score" not in summary

    def test_get_campaign_history_strips_test_score_for_active(self, db, active_campaign):
        result = handle_get_campaign_history(db, "strip-test")
        for round_row in result["rounds"]:
            summary = round_row.get("summary")
            if summary is not None:
                if isinstance(summary, str):
                    summary = json.loads(summary)
                assert "test_score" not in summary

    def test_get_round_summary_keeps_test_score_for_completed(self, db, service, active_campaign):
        service.transition_campaign(active_campaign["id"], CampaignState.COMPLETED,
                                    termination_reason="max_rounds")
        result = handle_get_round_summary(db, "strip-test")
        summary = result["summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        assert "test_score" in summary
        assert summary["test_score"] == 0.93

    def test_get_campaign_status_keeps_test_score_for_completed(self, db, service, active_campaign):
        service.transition_campaign(active_campaign["id"], CampaignState.COMPLETED,
                                    termination_reason="max_rounds")
        result = handle_get_campaign_status(db, "strip-test")
        summary = result["latest_round"]["summary"]
        if isinstance(summary, str):
            summary = json.loads(summary)
        assert "test_score" in summary
        assert summary["test_score"] == 0.93

    def test_get_campaign_history_keeps_test_score_for_completed(self, db, service, active_campaign):
        service.transition_campaign(active_campaign["id"], CampaignState.COMPLETED,
                                    termination_reason="max_rounds")
        result = handle_get_campaign_history(db, "strip-test")
        for round_row in result["rounds"]:
            summary = round_row.get("summary")
            if summary is not None:
                if isinstance(summary, str):
                    summary = json.loads(summary)
                assert "test_score" in summary
