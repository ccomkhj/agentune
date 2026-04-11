import json
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from agentune.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCliInit:
    def test_init_creates_campaign(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        result = runner.invoke(cli, [
            "init", "my-campaign",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
            "--dataset", "breast_cancer",
            "--trials-per-round", "20",
            "--max-rounds", "5",
            "--patience", "3",
        ])
        assert result.exit_code == 0, result.output
        assert "my-campaign" in result.output
        assert "CREATED" in result.output

    def test_init_defaults_metric_direction_from_dataset(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        result = runner.invoke(cli, [
            "init", "defaults-test",
            "--backend", "xgboost",
            "--dataset", "breast_cancer",
        ])
        assert result.exit_code == 0, result.output
        assert "defaults-test" in result.output
        assert "Warning" not in result.output

    def test_init_warns_on_metric_mismatch(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        result = runner.invoke(cli, [
            "init", "metric-warn-test",
            "--backend", "xgboost",
            "--metric", "rmse",
            "--direction", "minimize",
            "--dataset", "breast_cancer",
        ])
        assert result.exit_code == 0, result.output
        warning_lines = [l for l in result.output.strip().split("\n") if "Warning" in l]
        # Should warn about both metric and direction
        assert len(warning_lines) == 2
        assert "accuracy" in warning_lines[0]  # canonical metric
        assert "maximize" in warning_lines[1]  # canonical direction

    def test_init_warns_on_direction_mismatch_only(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        result = runner.invoke(cli, [
            "init", "dir-warn-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "minimize",
            "--dataset", "breast_cancer",
        ])
        assert result.exit_code == 0, result.output
        # Should warn about direction but not metric
        lines = result.output.strip().split("\n")
        warning_lines = [l for l in lines if "Warning" in l]
        assert len(warning_lines) == 1
        assert "direction" in warning_lines[0].lower() or "maximize" in warning_lines[0]

    def test_init_duplicate_fails(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        runner.invoke(cli, [
            "init", "dup-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
            "--dataset", "breast_cancer",
        ])
        result = runner.invoke(cli, [
            "init", "dup-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
            "--dataset", "breast_cancer",
        ])
        assert result.exit_code != 0


    def test_init_with_mode(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        result = runner.invoke(cli, [
            "init", "test-explore",
            "--backend", "xgboost",
            "--dataset", "breast_cancer",
            "--mode", "strong-exploration",
        ])
        assert result.exit_code == 0, result.output
        assert "test-explore" in result.output

        # Verify mode was persisted
        from agentune.core.db import Database
        from agentune.core.campaign import CampaignService
        db = Database(test_db_url)
        db.setup_schema()
        svc = CampaignService(db)
        campaign = svc.get_campaign_by_name("test-explore")
        assert campaign["mode"] == "strong-exploration"
        db.close()


class TestCliStatus:
    def test_status_shows_campaign(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENTUNE_DB_URL", test_db_url)
        runner.invoke(cli, [
            "init", "status-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
            "--dataset", "breast_cancer",
        ])
        result = runner.invoke(cli, ["status", "status-test"])
        assert result.exit_code == 0
        assert "CREATED" in result.output


class TestRunDemoMode:
    """Test --demo flag on the run command."""

    @patch("agentune.cli.CampaignService")
    @patch("agentune.cli._get_db")
    def test_demo_prints_narration_block(self, mock_get_db, mock_svc_cls):
        """When --demo is passed, output includes round score and signals."""
        from agentune.runner import RunResult

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        mock_svc = MagicMock()
        mock_svc_cls.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = {
            "id": 1, "name": "test", "state": "RUNNING",
            "metric_name": "accuracy", "objective_direction": "maximize",
            "dataset": "breast_cancer", "split_seed": 42,
            "stop_conditions": '{"max_rounds": 6}',
        }
        mock_svc.get_rounds.return_value = [
            {"round_number": 2, "summary": {
                "best_score": 0.95, "delta_from_prev": 0.01,
                "param_importance": {"lr": 0.4, "depth": 0.3},
                "plateau_signal": False,
            }},
        ]
        mock_svc.get_campaign_history.return_value = {
            "decisions": [
                {"round_id": 1, "action": "continue", "accepted": True,
                 "justification": "Still improving"},
            ],
        }

        # Mock the runner that's imported inside the function
        with patch("agentune.cli.load_dataset") as mock_load, \
             patch("agentune.cli.RoundRunner") as mock_runner_cls:
            mock_load.return_value = (MagicMock(), {"metric": "accuracy", "direction": "maximize"})
            mock_runner = MagicMock()
            mock_runner_cls.return_value = mock_runner
            mock_runner.run_next_round.return_value = RunResult(
                status="AWAITING_AGENT", round_number=2,
            )

            runner = CliRunner()
            result = runner.invoke(cli, ["run", "test", "--dataset", "breast_cancer", "--demo"])

        assert result.exit_code == 0
        assert "Round 2" in result.output
        assert "0.9500" in result.output

    @patch("agentune.cli.CampaignService")
    @patch("agentune.cli._get_db")
    def test_no_demo_prints_minimal(self, mock_get_db, mock_svc_cls):
        """Without --demo, just the status line."""
        from agentune.runner import RunResult

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        mock_svc = MagicMock()
        mock_svc_cls.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = {
            "id": 1, "name": "test", "state": "RUNNING",
            "dataset": "breast_cancer", "split_seed": 42,
        }

        with patch("agentune.cli.load_dataset") as mock_load, \
             patch("agentune.cli.RoundRunner") as mock_runner_cls:
            mock_load.return_value = (MagicMock(), {"metric": "accuracy", "direction": "maximize"})
            mock_runner = MagicMock()
            mock_runner_cls.return_value = mock_runner
            mock_runner.run_next_round.return_value = RunResult(
                status="AWAITING_AGENT", round_number=2,
            )

            runner = CliRunner()
            result = runner.invoke(cli, ["run", "test", "--dataset", "breast_cancer"])

        assert result.exit_code == 0
        assert "Round 2: AWAITING_AGENT" in result.output
