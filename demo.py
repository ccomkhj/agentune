"""
Demo: Agent-driven HPO on California Housing dataset.

Runs a full multi-round campaign where the agent observes round summaries,
diagnoses the optimization state, and makes data-driven decisions about
how to adjust the search space.

Usage:
    uv run python demo.py
"""

import json
import sys

from agent_hpo.agent import AgentReasoner
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.db import Database
from agent_hpo.core.models import (
    CampaignConfig,
    ImprovementCriteria,
    RoundSummary,
    StopConditions,
)
from agent_hpo.backends.xgboost import XGBoostBackend
from agent_hpo.datasets import load_dataset
from agent_hpo.runner import RoundRunner

DB_URL = "postgresql://agent_hpo:agent_hpo@localhost:5432/agent_hpo"
CAMPAIGN_NAME = "california-housing-agent-demo"


def main():
    db = Database(DB_URL)
    db.setup_schema()
    service = CampaignService(db)
    agent = AgentReasoner()

    # Load dataset
    dataset, meta = load_dataset("california_housing", seed=42)
    backend = XGBoostBackend()

    # Create campaign
    print("\n" + "#" * 72)
    print("#  Agent-Driven HPO Demo: California Housing (RMSE, minimize)")
    print("#" * 72)

    config = CampaignConfig(
        metric_name="rmse",
        objective_direction="minimize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=backend.default_search_space(),
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(
            max_rounds=6,
            patience_rounds=3,
            max_total_trials=240,
        ),
        trials_per_round=40,
    )

    try:
        campaign = service.create_campaign(CAMPAIGN_NAME, config)
    except Exception:
        # Campaign exists — need a unique name
        import uuid
        CAMPAIGN_NAME_UNIQUE = f"{CAMPAIGN_NAME}-{uuid.uuid4().hex[:6]}"
        campaign = service.create_campaign(CAMPAIGN_NAME_UNIQUE, config)

    campaign_id = campaign["id"]
    runner = RoundRunner(db, dataset)
    prev_summaries: list[RoundSummary] = []
    prev_decisions: list[dict] = []
    total_trials = 0

    print(f"\n  Campaign: {campaign['name']} (id={campaign_id})")
    print(f"  Dataset: California Housing (20,640 samples)")
    print(f"  Metric: RMSE (minimize)")
    print(f"  Budget: 40 trials/round, max 240 total, patience=3")
    print(f"  Backend: XGBoost with 9 hyperparameters")
    print()

    for round_num in range(1, 7):  # max 6 rounds
        # --- Run the round ---
        print(f"\n{'─' * 72}")
        print(f"  RUNNING Round {round_num} ...")
        print(f"{'─' * 72}")

        result = runner.run_next_round(campaign_id)
        print(f"  Status: {result.status}" + (f" (stop: {result.stop_reason})" if result.stop_reason else ""))

        # Get the round summary
        rounds = service.get_rounds(campaign_id)
        current_round = rounds[-1]
        summary_raw = current_round["summary"]
        if summary_raw is None:
            # Round was CLOSED without summary (e.g., budget exhausted before running)
            if result.status == "COMPLETED":
                # Get best score from previous round
                for r in reversed(rounds):
                    if r.get("summary"):
                        s = r["summary"]
                        if isinstance(s, str):
                            s = json.loads(s)
                        print(f"\n  Campaign completed: {result.stop_reason}")
                        print(f"  Final {s['metric_name']}: {s['best_score']:.6f}")
                        break
                break
            continue
        if isinstance(summary_raw, str):
            summary_raw = json.loads(summary_raw)
        summary = RoundSummary.from_dict(summary_raw)
        total_trials += summary.trials_added

        # Campaign ended by stop condition
        if result.status == "COMPLETED":
            print(f"\n  Campaign completed: {result.stop_reason}")
            print(f"  Final {summary.metric_name}: {summary.best_score:.6f}")
            break

        # --- Agent decides ---
        # Get current search space
        search_space_raw = current_round["search_space"]
        if isinstance(search_space_raw, str):
            search_space_raw = json.loads(search_space_raw)

        decision = agent.decide(
            summary=summary,
            round_number=round_num,
            current_search_space=search_space_raw,
            prev_summaries=prev_summaries if prev_summaries else None,
            prev_decisions=prev_decisions if prev_decisions else None,
        )

        # Print the full reasoning report
        print(decision.format_report())

        # Submit to core
        proposal = decision.to_proposal()
        result_decision = service.submit_proposal(campaign_id, proposal)

        if not result_decision["accepted"]:
            print(f"\n  PROPOSAL REJECTED: {result_decision['rejection_reason']}")
            print("  Falling back to 'continue'...")
            from agent_hpo.core.models import ActionProposal
            fallback = ActionProposal(
                action="continue",
                justification=f"Fallback after rejection: {result_decision['rejection_reason']}",
                reference_round_ids=[current_round["id"]],
            )
            result_decision = service.submit_proposal(campaign_id, fallback)
            if not result_decision["accepted"]:
                print(f"  Fallback also rejected. Stopping.")
                break
        else:
            print(f"\n  Proposal ACCEPTED by core.")

        # Track history
        prev_summaries.append(summary)
        prev_decisions.append({
            "action": decision.action,
            "accepted": result_decision["accepted"],
            "round_id": current_round["id"],
        })

        # If agent chose stop
        if decision.action == "stop":
            break

    # --- Final Report ---
    print("\n\n" + "=" * 72)
    print("  CAMPAIGN FINAL REPORT")
    print("=" * 72)
    campaign_final = service.get_campaign(campaign_id)
    rounds_final = service.get_rounds(campaign_id)
    print(f"  State: {campaign_final['state']}")
    print(f"  Total rounds: {len(rounds_final)}")
    print(f"  Total trials: {total_trials}")
    print()

    print(f"  {'Round':>5s}  {'RMSE':>10s}  {'Delta':>10s}  {'Trials':>6s}  {'State':12s}  {'Decision'}")
    print(f"  {'─' * 5}  {'─' * 10}  {'─' * 10}  {'─' * 6}  {'─' * 12}  {'─' * 20}")
    decisions_all = service.get_campaign_history(campaign_id)["decisions"]
    decision_map = {}
    for d in decisions_all:
        decision_map[d["round_id"]] = d["action"]

    for r in rounds_final:
        s = r.get("summary")
        if s:
            if isinstance(s, str):
                s = json.loads(s)
            rmse = f"{s['best_score']:.6f}"
            delta = f"{s.get('delta_from_prev', 0) or 0:+.6f}" if s.get("delta_from_prev") is not None else "   first"
            trials = str(s["trials_added"])
        else:
            rmse = "N/A"
            delta = "N/A"
            trials = "0"
        action = decision_map.get(r["id"], "")
        print(f"  {r['round_number']:5d}  {rmse:>10s}  {delta:>10s}  {trials:>6s}  {r['state']:12s}  {action}")

    # Best params
    last_summary = None
    for r in reversed(rounds_final):
        if r.get("summary"):
            last_summary = r["summary"]
            if isinstance(last_summary, str):
                last_summary = json.loads(last_summary)
            break
    if last_summary:
        print(f"\n  Final best RMSE: {last_summary['best_score']:.6f}")
        print(f"  Best parameters:")
        for k, v in last_summary["best_params"].items():
            if isinstance(v, float):
                print(f"    {k:25s} {v:.6f}")
            else:
                print(f"    {k:25s} {v}")

    db.close()
    print()


if __name__ == "__main__":
    main()
