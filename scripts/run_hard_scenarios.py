"""Run 3 hard optimization scenarios end-to-end via MCP handlers.

Scenarios:
  1. Covertype  — 20k rows, 7-class forest cover, 54 features
  2. Credit-g   — 1000 rows, imbalanced binary, mixed-type features
  3. Phoneme    — 5404 rows, noisy binary speech classification

Each campaign: 4 rounds, 30 trials/round, patience=3.
After each round the script reads the summary and makes a simple decision.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Point at dockerised Postgres
os.environ.setdefault("AGENTUNE_DB_URL", "postgresql://agentune:agentune@localhost:5432/agentune")

from agentune.core.campaign import CampaignService
from agentune.core.db import Database
from agentune.core.models import (
    CampaignConfig,
    ImprovementCriteria,
    ParamSpec,
    StopConditions,
)
from agentune.mcp_server import (
    handle_get_campaign_status,
    handle_get_round_summary,
    handle_run_next_round,
    handle_submit_action_proposal,
)

SCENARIOS = [
    {
        "name": "hard-covertype",
        "dataset": "covertype",
        "metric": "accuracy",
        "direction": "maximize",
        "trials_per_round": 30,
        "max_rounds": 6,
        "patience": 4,
    },
    {
        "name": "hard-credit-g",
        "dataset": "credit_g",
        "metric": "accuracy",
        "direction": "maximize",
        "trials_per_round": 30,
        "max_rounds": 6,
        "patience": 4,
    },
    {
        "name": "hard-phoneme",
        "dataset": "phoneme",
        "metric": "accuracy",
        "direction": "maximize",
        "trials_per_round": 30,
        "max_rounds": 6,
        "patience": 4,
    },
]


def _print_sep(char: str = "=", width: int = 72) -> None:
    print(char * width)


# XGBoost param bounds
PARAM_BOUNDS = {
    "subsample": (0.1, 1.0), "colsample_bytree": (0.1, 1.0),
    "colsample_bylevel": (0.1, 1.0), "colsample_bynode": (0.1, 1.0),
    "learning_rate": (0.001, 2.0), "gamma": (0.0, 10.0),
    "min_child_weight": (0.1, 20.0), "reg_alpha": (1e-8, 100.0),
    "reg_lambda": (1e-8, 100.0), "max_depth": (1, 15),
    "n_estimators": (10, 1000), "max_leaves": (0, 512),
    "max_bin": (32, 1024), "scale_pos_weight": (0.1, 20.0),
}

# Extended params available for revise_search
EXTENDED_PARAMS = {
    "max_leaves": {"type": "int", "low": 0, "high": 256},
    "max_bin": {"type": "int", "low": 64, "high": 512},
    "colsample_bylevel": {"type": "float", "low": 0.3, "high": 1.0},
    "colsample_bynode": {"type": "float", "low": 0.3, "high": 1.0},
    "scale_pos_weight": {"type": "float", "low": 0.5, "high": 10.0},
    "grow_policy": {"type": "categorical", "choices": ["depthwise", "lossguide"]},
}


def _detect_boundary_hits(summary: dict) -> list[str]:
    """Find params whose best values are near their search space boundaries."""
    best_params = summary.get("best_params", {})
    ranges_used = summary.get("param_ranges_used", {})
    hits = []
    for pname, pval in best_params.items():
        if not isinstance(pval, (int, float)):
            continue
        lo_bound, hi_bound = PARAM_BOUNDS.get(pname, (None, None))
        if lo_bound is None:
            continue
        span = hi_bound - lo_bound
        if span <= 0:
            continue
        # Hit if within 5% of either boundary
        if (pval - lo_bound) / span < 0.05 or (hi_bound - pval) / span < 0.05:
            hits.append(pname)
    return hits


def _build_narrowed_space(top_params: list, best_params: dict, current_space: list[dict]) -> list[dict]:
    """Build a tighter search space around best values for the top params.
    Preserves log flag from the current space."""
    # Build lookup for current space's log settings
    log_lookup = {p["name"]: p.get("log", False) for p in current_space}

    narrowed = []
    for pname, _ in top_params[:3]:
        pval = best_params.get(pname)
        if pval is None or not isinstance(pval, (int, float)):
            continue
        lo_bound, hi_bound = PARAM_BOUNDS.get(pname, (0.001, pval * 3))
        low = max(lo_bound, pval * 0.5)
        high = min(hi_bound, pval * 2.0)
        if low >= high:
            low, high = lo_bound, hi_bound
        is_log = log_lookup.get(pname, False)
        if is_log and low <= 0:
            low = max(1e-8, low)
        if isinstance(pval, int) or (isinstance(pval, float) and pval == int(pval) and pval > 2):
            narrowed.append({"name": pname, "type": "int", "low": max(1, int(low)), "high": max(int(low) + 1, int(high)), "log": False})
        else:
            narrowed.append({"name": pname, "type": "float", "low": low, "high": high, "log": is_log})
    return narrowed


def _build_widened_space(current_space: list[dict], boundary_hits: list[str]) -> list[dict]:
    """Widen the current search space for params hitting boundaries.
    Always preserves the log flag from the original space to avoid Optuna conflicts."""
    widened = []
    for p in current_space:
        pname = p["name"]
        if pname in boundary_hits and pname in PARAM_BOUNDS:
            lo_bound, hi_bound = PARAM_BOUNDS[pname]
            old_lo = p.get("low", lo_bound)
            old_hi = p.get("high", hi_bound)
            span = old_hi - old_lo
            new_lo = max(lo_bound, old_lo - span * 0.5)
            new_hi = min(hi_bound, old_hi + span * 0.5)
            is_log = p.get("log", False)
            if is_log and new_lo <= 0:
                new_lo = max(1e-8, new_lo)
            entry = {"name": pname, "type": p["type"], "low": new_lo, "high": new_hi, "log": is_log}
            if "choices" in p:
                entry["choices"] = p["choices"]
            widened.append(entry)
        else:
            # Copy unchanged but ensure log is always explicit
            cp = dict(p)
            if cp["type"] in ("float", "int") and "log" not in cp:
                cp["log"] = False
            widened.append(cp)
    return widened


def _build_revised_space(current_space: list[dict], summary: dict) -> list[dict] | None:
    """Drop weakest param, add a promising extended param. Returns None if no good swap."""
    importance = summary.get("param_importance", {})
    current_names = {p["name"] for p in current_space}

    # Find weakest param to drop (lowest importance, >0 so we have data)
    ranked = sorted(
        [(n, importance.get(n, 0.0)) for n in current_names],
        key=lambda x: x[1],
    )
    if not ranked:
        return None
    drop_name = ranked[0][0]

    # Find best extended param to add (not already in space)
    candidates = [n for n in EXTENDED_PARAMS if n not in current_names]
    if not candidates:
        return None
    add_name = candidates[0]  # pick first available

    # Build new space: keep all except dropped, add new
    revised = [p for p in current_space if p["name"] != drop_name]
    ext = EXTENDED_PARAMS[add_name]
    entry = {"name": add_name, **ext}
    revised.append(entry)

    return revised


def _decide(
    summary: dict,
    round_number: int,
    direction: str,
    current_space: list[dict],
    history: list[dict],
) -> dict:
    """Dynamic decision agent using all 5 actions based on signals."""
    importance = summary.get("param_importance", {})
    plateau = summary.get("plateau_signal", False)
    new_best = summary.get("new_best_in_round", False)
    best_score = summary.get("best_score")
    best_params = summary.get("best_params", {})
    gen_gap = summary.get("generalization_gap")
    completed_trials = summary.get("round_completed_trials", 0)

    top_params = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    max_importance = top_params[0][1] if top_params else 0
    boundary_hits = _detect_boundary_hits(summary)
    past_actions = [h.get("action") for h in history]
    consecutive_no_improve = 0
    for h in reversed(history):
        if h.get("new_best") is False:
            consecutive_no_improve += 1
        else:
            break

    print(f"    plateau={plateau}  new_best={new_best}  best_score={best_score}  gen_gap={gen_gap}")
    if top_params:
        print(f"    top params: {', '.join(f'{n}={v:.1%}' for n, v in top_params[:3])}")
    if boundary_hits:
        print(f"    boundary hits: {boundary_hits}")

    ref = [round_number]

    # === Decision logic (priority order) ===

    # 1. WIDEN: single param hitting boundary — safe range extension
    if boundary_hits and len(boundary_hits) == 1 and not new_best and plateau:
        widened = _build_widened_space(current_space, boundary_hits)
        return {
            "action": "widen_search",
            "justification": f"Round {round_number}: {boundary_hits[0]} hitting boundary. Widening range.",
            "proposed_search_space": widened,
            "reference_round_ids": ref,
        }

    # 2. REVISE: plateau + no dominant param OR multiple boundary hits = need different params
    if plateau and not new_best and (max_importance < 0.15 or len(boundary_hits) >= 2) and "revise_search" not in past_actions:
        revised = _build_revised_space(current_space, summary)
        if revised:
            current_names = {p["name"] for p in current_space}
            revised_names = {p["name"] for p in revised}
            added = revised_names - current_names
            dropped = current_names - revised_names
            return {
                "action": "revise_search",
                "justification": f"Round {round_number}: plateau with no dominant param (max importance {max_importance:.1%}). Swapping: drop {dropped}, add {added}.",
                "proposed_search_space": revised,
                "reference_round_ids": ref,
            }

    # 3. NARROW: plateau + dominant param = focus on what matters
    if plateau and top_params and max_importance > 0.25:
        narrowed = _build_narrowed_space(top_params, best_params, current_space)
        if narrowed:
            dominant = top_params[0][0]
            return {
                "action": "narrow_search",
                "justification": f"Round {round_number}: plateau with {dominant} dominant at {max_importance:.1%}. Narrowing top params around best values.",
                "proposed_search_space": narrowed,
                "reference_round_ids": ref,
            }

    # 4. INCREASE_BUDGET: improving but slowly, more trials might help TPE converge
    if new_best and plateau and completed_trials > 0:
        new_budget = int(completed_trials * 1.5)
        return {
            "action": "increase_budget",
            "justification": f"Round {round_number}: found new best but plateau in late trials. Increasing budget from {completed_trials} to {new_budget} for better TPE convergence.",
            "proposed_budget": new_budget,
            "reference_round_ids": ref,
        }

    # 5. CONTINUE: still making progress
    return {
        "action": "continue",
        "justification": f"Round {round_number}: score={best_score}, {'improving' if new_best else 'stable'}. Continuing exploration.",
        "reference_round_ids": ref,
    }


def run_scenario(db: Database, scenario: dict) -> dict:
    """Run one full campaign loop, return final status."""
    name = scenario["name"]
    _print_sep()
    print(f"SCENARIO: {name}  ({scenario['dataset']}, {scenario['metric']} {scenario['direction']})")
    _print_sep()

    service = CampaignService(db)

    # Create campaign
    from agentune.backends.xgboost import XGBoostBackend

    backend = XGBoostBackend()
    config = CampaignConfig(
        metric_name=scenario["metric"],
        objective_direction=scenario["direction"],
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=backend.default_search_space(),
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(
            max_rounds=scenario["max_rounds"],
            patience_rounds=scenario["patience"],
        ),
        trials_per_round=scenario["trials_per_round"],
        dataset=scenario["dataset"],
    )

    try:
        service.create_campaign(name, config)
        print(f"  Campaign '{name}' created.")
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            print(f"  Campaign '{name}' already exists, skipping creation.")
        else:
            raise

    # Autonomous loop
    round_scores = []
    round_history: list[dict] = []  # track signals for decision-making
    loop_start = time.time()

    for iteration in range(scenario["max_rounds"] + 1):  # +1 safety margin
        status = handle_get_campaign_status(db, name)
        if status["state"] in ("COMPLETED", "FAILED", "STOPPED"):
            print(f"\n  Campaign terminal: {status['state']}")
            if status.get("termination_reason"):
                print(f"  Reason: {status['termination_reason']}")
            break

        # Run next round
        print(f"\n  Round {iteration + 1}: running trials...", end=" ", flush=True)
        t0 = time.time()
        result = handle_run_next_round(db, name)
        dt = time.time() - t0
        print(f"done in {dt:.1f}s  status={result['status']}")

        if result["status"] == "COMPLETED":
            print(f"  Hard stop: {result.get('stop_reason', '?')}")
            break

        if result["status"] == "FAILED":
            print(f"  FAILED: {result.get('stop_reason', '?')}")
            break

        # Read summary and current round's search space
        round_data = handle_get_round_summary(db, name, result["round_number"])
        summary = round_data.get("summary", {})
        if isinstance(summary, str):
            summary = json.loads(summary)
        current_space = round_data.get("search_space", [])
        if isinstance(current_space, str):
            current_space = json.loads(current_space)

        best = summary.get("best_score")
        round_scores.append(best)
        print(f"  Summary: best_score={best}  trials_in_round={summary.get('round_completed_trials', '?')}")

        # Decide using full context
        proposal = _decide(
            summary, result["round_number"], scenario["direction"],
            current_space, round_history,
        )
        print(f"  Decision: {proposal['action']}")
        decision = handle_submit_action_proposal(db, name, proposal)

        # Track history for future decisions
        round_history.append({
            "round": result["round_number"],
            "action": proposal["action"],
            "accepted": decision.get("accepted", False),
            "new_best": summary.get("new_best_in_round", False),
            "best_score": best,
        })

        if not decision.get("accepted"):
            print(f"  REJECTED: {decision.get('rejection_reason')}")
            # Fall back to continue
            fallback = {
                "action": "continue",
                "justification": f"Fallback after rejection: {decision.get('rejection_reason')}",
                "reference_round_ids": [result["round_number"]],
            }
            decision = handle_submit_action_proposal(db, name, fallback)
            round_history[-1]["action"] = "continue"
            round_history[-1]["accepted"] = decision.get("accepted", False)
            print(f"  Fallback continue: accepted={decision.get('accepted')}")

    total_time = time.time() - loop_start
    final = handle_get_campaign_status(db, name)

    _print_sep("-")
    print(f"  RESULT: {name}")
    print(f"    State:   {final['state']}")
    print(f"    Rounds:  {final['total_rounds']}")
    print(f"    Scores:  {' -> '.join(f'{s:.4f}' if s else '?' for s in round_scores)}")
    if final.get("termination_reason"):
        print(f"    Reason:  {final['termination_reason']}")
    print(f"    Time:    {total_time:.1f}s")
    _print_sep("-")

    return {
        "name": name,
        "state": final["state"],
        "rounds": final["total_rounds"],
        "scores": round_scores,
        "time": total_time,
        "termination_reason": final.get("termination_reason"),
    }


def main() -> None:
    db = Database(os.environ["AGENTUNE_DB_URL"])
    db.setup_schema()

    results = []
    for scenario in SCENARIOS:
        try:
            r = run_scenario(db, scenario)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR in {scenario['name']}: {e}")
            results.append({"name": scenario["name"], "state": "ERROR", "error": str(e)})

    db.close()

    # Final summary
    print("\n")
    _print_sep("=")
    print("FINAL SUMMARY")
    _print_sep("=")
    print(f"{'Scenario':<20s} {'State':<12s} {'Rounds':>6s} {'Best Score':>12s} {'Time':>8s} {'Reason'}")
    _print_sep("-")
    for r in results:
        scores = r.get("scores", [])
        best = max(scores) if scores else None
        best_str = f"{best:.4f}" if best else "N/A"
        time_str = f"{r.get('time', 0):.1f}s"
        reason = r.get("termination_reason", r.get("error", ""))
        print(f"{r['name']:<20s} {r['state']:<12s} {r.get('rounds', '?'):>6} {best_str:>12s} {time_str:>8s} {reason}")
    _print_sep("=")


if __name__ == "__main__":
    main()
