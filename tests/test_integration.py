"""End-to-end test: full campaign lifecycle without a real agent."""

import pytest
import json
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec,
    ActionProposal, DatasetSplit,
)
from agent_hpo.core.state import CampaignState, RoundState
from agent_hpo.backends.xgboost import XGBoostBackend
from agent_hpo.summarizer import RoundSummarizer
from agent_hpo.scheduler import Scheduler


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


def test_full_campaign_lifecycle(db, dataset):
    service = CampaignService(db)
    backend = XGBoostBackend()
    summarizer = RoundSummarizer()

    # 1. Create campaign
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=backend.default_search_space(),
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=3, patience_rounds=2),
        trials_per_round=5,
    )
    campaign = service.create_campaign("integration-test", config)
    assert campaign["state"] == "CREATED"

    rounds = service.get_rounds(campaign["id"])
    r1 = rounds[0]
    assert r1["round_number"] == 1
    assert r1["state"] == "PROPOSED"

    # 2. Run round 1
    service.transition_campaign(campaign["id"], CampaignState.RUNNING)
    service.transition_round(r1["id"], RoundState.RUNNING)

    search_space_raw = r1["search_space"]
    if isinstance(search_space_raw, str):
        search_space_raw = json.loads(search_space_raw)
    search_space = [ParamSpec.from_dict(s) for s in search_space_raw]
    objective = backend.create_objective(dataset, "accuracy", search_space)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=r1["budget"], show_progress_bar=False)

    service.complete_round_execution(r1["id"], trial_end=len(study.trials))

    # 3. Summarize
    service.transition_round(r1["id"], RoundState.SUMMARIZING)
    summary = summarizer.summarize(
        study=study,
        campaign_id=campaign["id"],
        round_id=r1["id"],
        metric_name="accuracy",
        objective_direction="maximize",
        trial_offset=0,
        trial_end=len(study.trials),
        prev_best_score=None,
        parent_round_id=None,
        optuna_study_name=r1["optuna_study_name"],
        action_that_created="init",
        cumulative_wall_time=0.0,
    )
    service.write_summary(r1["id"], summary.to_dict())
    service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

    assert summary.best_score is not None
    assert summary.round_completed_trials > 0

    # 4. Agent proposes continue
    proposal = ActionProposal(
        action="continue",
        justification=f"Round 1 achieved {summary.best_score:.4f}, still improving",
        reference_round_ids=[r1["id"]],
    )
    decision = service.submit_proposal(campaign["id"], proposal)
    assert decision["accepted"] is True

    # Verify round 2 was created
    rounds = service.get_rounds(campaign["id"])
    assert len(rounds) == 2
    r2 = rounds[1]
    assert r2["round_number"] == 2
    assert r2["optuna_study_name"] == r1["optuna_study_name"]
    r1_updated = service.get_round(r1["id"])
    assert r2["trial_offset"] == r1_updated["trial_end"]

    # 5. Agent proposes stop after round 2
    service.transition_round(r2["id"], RoundState.RUNNING)
    study.optimize(objective, n_trials=r2["budget"], show_progress_bar=False)
    service.complete_round_execution(r2["id"], trial_end=len(study.trials))
    service.transition_round(r2["id"], RoundState.SUMMARIZING)

    summary2 = summarizer.summarize(
        study=study,
        campaign_id=campaign["id"],
        round_id=r2["id"],
        metric_name="accuracy",
        objective_direction="maximize",
        trial_offset=r2["trial_offset"],
        trial_end=len(study.trials),
        prev_best_score=summary.best_score,
        parent_round_id=None,
        optuna_study_name=r2["optuna_study_name"],
        action_that_created="continue",
        cumulative_wall_time=summary.total_wall_time_seconds,
    )
    service.write_summary(r2["id"], summary2.to_dict())
    service.transition_round(r2["id"], RoundState.AWAITING_AGENT)

    stop_proposal = ActionProposal(
        action="stop",
        justification="Reached sufficient accuracy",
        reference_round_ids=[r1["id"], r2["id"]],
    )
    decision = service.submit_proposal(campaign["id"], stop_proposal)
    assert decision["accepted"] is True

    # Verify campaign completed
    final = service.get_campaign(campaign["id"])
    assert final["state"] == "COMPLETED"
