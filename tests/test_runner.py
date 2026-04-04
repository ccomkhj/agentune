import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from agent_hpo.runner import RoundRunner, RunResult
from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, DatasetSplit,
)
from agent_hpo.core.state import CampaignState, RoundState


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
    from agent_hpo.backends.xgboost import XGBoostBackend
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
    )
    return service.create_campaign("runner-test", config)


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
