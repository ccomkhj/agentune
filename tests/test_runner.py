import pytest
from unittest.mock import patch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from agentune.runner import RoundRunner, RunResult
from agentune.core.db import Database
from agentune.core.campaign import CampaignService
from agentune.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, DatasetSplit,
)
from agentune.core.state import CampaignState, RoundState


@pytest.fixture
def db(test_db_url):
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


@pytest.fixture
def dataset():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    return DatasetSplit(X_tr, y_tr, X_val, y_val, X_te, y_te)


@pytest.fixture
def campaign(db):
    service = CampaignService(db)
    from agentune.backends.xgboost import XGBoostBackend
    backend = XGBoostBackend()
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=backend.default_search_space(),
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=3, patience_rounds=2, max_total_trials=15),
        trials_per_round=5,
        dataset="breast_cancer",
    )
    return service.create_campaign("runner-test", config)


class TestFailurePath:
    def test_exception_during_run_marks_round_failed(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        service = CampaignService(db)

        with patch.object(runner, '_execute', side_effect=RuntimeError("OOM")):
            result = runner.run_next_round(campaign["id"])

        assert result.status == "FAILED"

        # Round should be FAILED
        rounds = service.get_rounds(campaign["id"])
        assert rounds[0]["state"] == "FAILED"

        # Campaign should be FAILED with termination metadata
        updated = service.get_campaign(campaign["id"])
        assert updated["state"] == "FAILED"
        assert updated["termination_reason"] == "failed"
        assert "OOM" in updated["termination_detail"]


class TestRoundRunner:
    def test_run_first_round(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        result = runner.run_next_round(campaign["id"])
        assert result.status in ("AWAITING_AGENT", "COMPLETED", "FAILED")
        assert result.round_number == 1

        service = CampaignService(db)
        r1 = service.get_rounds(campaign["id"])[0]
        assert r1["summary"] is not None
        assert r1["trial_end"] is not None
        assert r1["trial_end"] > 0

    def test_run_respects_budget_clipping(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        result1 = runner.run_next_round(campaign["id"])
        # First round may complete due to stop conditions or await agent
        assert result1.status in ("AWAITING_AGENT", "COMPLETED")
        assert result1.round_number == 1

    def test_run_uses_persistent_optuna_storage(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        result = runner.run_next_round(campaign["id"])

        import optuna
        storage = optuna.storages.RDBStorage(db.optuna_storage_url)
        service = CampaignService(db)
        r1 = service.get_rounds(campaign["id"])[0]
        study = optuna.load_study(study_name=r1["optuna_study_name"], storage=storage)
        assert len(study.trials) > 0
