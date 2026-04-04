import json
import pytest
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
