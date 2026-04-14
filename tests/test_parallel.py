"""Tests for parallel trial execution."""

import time
import optuna
import pytest
from agentune.parallel import ParallelOptimizer


class TestParallelOptimizer:
    def test_runs_correct_number_of_trials(self):
        study = optuna.create_study(direction="minimize")

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x ** 2

        optimizer = ParallelOptimizer(n_jobs=2)
        optimizer.optimize(study, objective, n_trials=20)
        assert len(study.trials) == 20

    def test_single_job_falls_back_to_serial(self):
        study = optuna.create_study(direction="minimize")

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return x ** 2

        optimizer = ParallelOptimizer(n_jobs=1)
        optimizer.optimize(study, objective, n_trials=10)
        assert len(study.trials) == 10

    def test_respects_timeout(self):
        study = optuna.create_study(direction="minimize")

        def slow_objective(trial):
            x = trial.suggest_float("x", -10, 10)
            time.sleep(0.5)
            return x ** 2

        optimizer = ParallelOptimizer(n_jobs=2)
        start = time.time()
        optimizer.optimize(study, slow_objective, n_trials=100, timeout=1.5)
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert len(study.trials) < 100


class TestCampaignNJobs:
    def test_campaign_stores_n_jobs(self, test_db_url):
        from agentune.core.db import Database
        from agentune.core.campaign import CampaignService
        from agentune.core.models import (
            CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec,
        )

        db = Database(test_db_url)
        db.setup_schema()
        service = CampaignService(db)
        config = CampaignConfig(
            metric_name="rmse",
            objective_direction="minimize",
            backend="xgboost",
            sampler_config={"seed": 42},
            initial_search_space=[ParamSpec(name="lr", type="float", low=0.01, high=1.0)],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(max_rounds=3),
            trials_per_round=10,
            n_jobs=4,
        )
        campaign = service.create_campaign("parallel-test", config)
        assert campaign["n_jobs"] == 4
        db.close()

    def test_campaign_defaults_n_jobs_to_1(self, test_db_url):
        from agentune.core.db import Database
        from agentune.core.campaign import CampaignService
        from agentune.core.models import (
            CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec,
        )

        db = Database(test_db_url)
        db.setup_schema()
        service = CampaignService(db)
        config = CampaignConfig(
            metric_name="rmse",
            objective_direction="minimize",
            backend="xgboost",
            sampler_config={"seed": 42},
            initial_search_space=[ParamSpec(name="lr", type="float", low=0.01, high=1.0)],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(max_rounds=3),
            trials_per_round=10,
        )
        campaign = service.create_campaign("serial-test", config)
        assert campaign["n_jobs"] == 1
        db.close()
