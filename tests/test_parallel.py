"""Tests for parallel trial execution."""

import pytest


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
