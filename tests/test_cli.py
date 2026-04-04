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
