"""CLI: thin Click wrapper over the core service layer."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, NoReturn

import click

from agentune.backends import get_backend
from agentune.core.campaign import CampaignService
from agentune.core.db import Database
from agentune.core.models import (
    CampaignConfig,
    ImprovementCriteria,
    StopConditions,
)
from agentune.core.state import CampaignState


def _get_db() -> Database:
    url = os.environ.get("AGENTUNE_DB_URL", "postgresql://localhost:5432/agentune")
    db = Database(url)
    db.setup_schema()
    return db


def _load_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _exit_with_error(message: str) -> NoReturn:
    click.echo(message, err=True)
    sys.exit(1)


def _exit_for_exception(error: Exception) -> NoReturn:
    _exit_with_error(f"Error: {error}")


def _get_campaign_or_exit(service: CampaignService, name: str) -> dict:
    campaign = service.get_campaign_by_name(name)
    if campaign is None:
        _exit_with_error(f"Campaign '{name}' not found")
    return campaign


def _print_best_params(best_params: dict[str, Any]) -> None:
    click.echo("\n    Best params:")
    for name, value in best_params.items():
        if isinstance(value, float):
            click.echo(f"      {name:25s} {value:.6f}")
            continue
        click.echo(f"      {name:25s} {value}")


def _print_search_space_changes(changes: list[dict[str, Any]]) -> None:
    click.echo("\n  SEARCH SPACE CHANGES:")
    click.echo(f"    {'Param':25s} {'Old Range':>20s}  →  {'New Range':20s} {'Reason'}")
    click.echo(f"    {'─' * 25} {'─' * 20}     {'─' * 20} {'─' * 30}")
    for change in changes:
        if change.get("param_type") == "categorical":
            old_range = "categorical"
            new_range = "categorical"
        else:
            old_range = f"[{change['old_low']:.4g}, {change['old_high']:.4g}]"
            new_range = f"[{change['new_low']:.4g}, {change['new_high']:.4g}]"
        click.echo(f"    {change['param_name']:25s} {old_range:>20s}  →  {new_range:20s} {change['reason']}")


@click.group()
def cli() -> None:
    """Agent-driven hyperparameter optimization."""
    pass


@cli.command()
@click.argument("name")
@click.option("--backend", default="xgboost")
@click.option("--metric", default=None, help="Metric to optimize (default: inferred from dataset)")
@click.option("--direction", default=None, type=click.Choice(["minimize", "maximize"]),
              help="Optimization direction (default: inferred from dataset)")
@click.option("--dataset", required=True, help="Dataset name: breast_cancer, california_housing, digits, covertype, credit_g, phoneme, store_sales, rossmann")
@click.option("--trials-per-round", default=50, type=int)
@click.option("--max-rounds", default=None, type=int)
@click.option("--max-trials", default=None, type=int)
@click.option("--max-wall-time", default=None, type=float)
@click.option("--patience", default=3, type=int)
@click.option("--target-score", default=None, type=float)
@click.option("--improvement-mode", default="strict_better",
              type=click.Choice(["strict_better", "min_absolute_delta", "min_relative_delta"]))
@click.option("--improvement-threshold", default=0.0, type=float)
@click.option("--sampler-seed", default=42, type=int)
@click.option("--split-seed", default=42, type=int)
@click.option("--target", default="target", help="Target column name for custom CSV/parquet datasets")
def init(
    name: str,
    backend: str,
    metric: str | None,
    direction: str | None,
    dataset: str,
    trials_per_round: int,
    max_rounds: int | None,
    max_trials: int | None,
    max_wall_time: float | None,
    patience: int,
    target_score: float | None,
    improvement_mode: str,
    improvement_threshold: float,
    sampler_seed: int,
    split_seed: int,
    target: str,
) -> None:
    """Create a new optimization campaign."""
    from agentune.datasets import DATASETS, _is_file_path

    if _is_file_path(dataset):
        if metric is None:
            _exit_with_error("Custom dataset requires --metric (accuracy, rmse, log_loss)")
        if direction is None:
            _exit_with_error("Custom dataset requires --direction (minimize or maximize)")
        # Encode target into descriptor for run_next_round
        dataset = f"{dataset}:target={target}"
    else:
        dataset_info = DATASETS.get(dataset)
        if dataset_info is None:
            _exit_with_error(f"Unknown dataset '{dataset}'. Available: {', '.join(DATASETS)}")

        canonical_metric = dataset_info["metric"]
        canonical_direction = dataset_info["direction"]

        if metric is None:
            metric = canonical_metric
        elif metric != canonical_metric:
            click.echo(
                f"Warning: dataset '{dataset}' is typically used with metric "
                f"'{canonical_metric}', but you specified '{metric}'."
            )

        if direction is None:
            direction = canonical_direction
        elif direction != canonical_direction:
            click.echo(
                f"Warning: dataset '{dataset}' is typically used with direction "
                f"'{canonical_direction}', but you specified '{direction}'."
            )

    db = _get_db()
    service = CampaignService(db)

    backend_cls = get_backend(backend)
    backend_instance = backend_cls()
    search_space = backend_instance.default_search_space()

    config = CampaignConfig(
        metric_name=metric,
        objective_direction=direction,
        backend=backend,
        sampler_config={"name": "TPESampler", "seed": sampler_seed},
        initial_search_space=search_space,
        improvement_criteria=ImprovementCriteria(mode=improvement_mode, threshold=improvement_threshold),
        stop_conditions=StopConditions(
            max_rounds=max_rounds,
            max_total_trials=max_trials,
            patience_rounds=patience,
            max_wall_time_seconds=max_wall_time,
            target_score=target_score,
        ),
        trials_per_round=trials_per_round,
        dataset=dataset,
        split_seed=split_seed,
    )

    try:
        campaign = service.create_campaign(name, config)
        click.echo(f"Campaign '{name}' created (id={campaign['id']}, state={campaign['state']})")
        click.echo(f"Round 1 ready with {trials_per_round} trials budget")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def status(name: str) -> None:
    """Show campaign status and latest round summary."""
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)

        click.echo(f"Campaign: {campaign['name']}")
        click.echo(f"State: {campaign['state']}")
        click.echo(f"Metric: {campaign['metric_name']} ({campaign['objective_direction']})")
        click.echo(f"Backend: {campaign['backend']}")
        if campaign.get("termination_reason"):
            click.echo(f"Termination: {campaign['termination_reason']}")
            if campaign.get("termination_detail"):
                click.echo(f"Detail: {campaign['termination_detail']}")

        rounds = service.get_rounds(campaign["id"])
        click.echo(f"Rounds: {len(rounds)}")
        if rounds:
            latest = rounds[-1]
            click.echo(f"Latest round: #{latest['round_number']} ({latest['state']})")
            if latest.get("summary"):
                summary = _load_json(latest["summary"])
                click.echo(f"Best score: {summary.get('best_score', 'N/A')}")
    finally:
        db.close()


@cli.command()
@click.argument("name")
def pause(name: str) -> None:
    """Pause a running campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)
        service.transition_campaign(campaign["id"], CampaignState.PAUSE_REQUESTED)
        click.echo(f"Campaign '{name}' pause requested")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def resume(name: str) -> None:
    """Resume a paused campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        click.echo(f"Campaign '{name}' resumed")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def stop(name: str) -> None:
    """Manually stop a campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)
        rounds = service.get_rounds(campaign["id"])
        active_round = rounds[-1] if rounds else None
        if active_round and active_round["state"] in ("RUNNING", "SUMMARIZING"):
            _exit_with_error(
                f"Cannot stop: round {active_round['round_number']} is {active_round['state']}. "
                f"Use 'agentune pause' to request a safe stop after the round completes."
            )

        service.transition_campaign(
            campaign["id"], CampaignState.STOPPED,
            termination_reason="manual_stop",
        )
        click.echo(f"Campaign '{name}' stopped")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def history(name: str) -> None:
    """Show full campaign history: rounds and decisions."""
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)
        data = service.get_campaign_history(campaign["id"])
        for r in data["rounds"]:
            click.echo(f"Round #{r['round_number']}: {r['state']} (study: {r['optuna_study_name']})")
        for d in data["decisions"]:
            status = "accepted" if d["accepted"] else f"rejected: {d['rejection_reason']}"
            click.echo(f"Decision: {d['action']} — {status}")
    finally:
        db.close()


@cli.command()
@click.argument("name")
@click.option("--round", "round_num", default=None, type=int, help="Show decision for a specific round only")
def decisions(name: str, round_num: int | None) -> None:
    """Show agent decision reasoning for each round.

    Displays the full observe → diagnose → decide chain stored in Postgres,
    showing what the agent saw, how it interpreted the signals, and why it
    chose a specific action.
    """
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)
        data = service.get_campaign_history(campaign["id"])
        decisions_list = data["decisions"]
        rounds_list = data["rounds"]

        # Build round number → round mapping
        round_map = {r["id"]: r for r in rounds_list}

        if not decisions_list:
            click.echo("No agent decisions recorded yet.")
            return

        click.echo(f"\n{'#' * 72}")
        click.echo(f"#  Agent Decision History: {campaign['name']}")
        click.echo(f"#  {campaign['metric_name']} ({campaign['objective_direction']}) | {campaign['state']}")
        click.echo(f"{'#' * 72}")

        for d in decisions_list:
            round_info = round_map.get(d["round_id"])
            rnum = round_info["round_number"] if round_info else "?"

            if round_num is not None and rnum != round_num:
                continue

            reasoning = _load_json(d.get("reasoning"))

            accepted_str = "ACCEPTED" if d["accepted"] else "REJECTED"
            click.echo(f"\n{'=' * 72}")
            click.echo(f"  Decision after Round {rnum} | {d['action']} | {accepted_str}")
            click.echo(f"  {d['created_at']}")
            click.echo(f"{'=' * 72}")

            if not reasoning:
                # Legacy decision without structured reasoning
                click.echo(f"\n  Justification: {d['justification']}")
                if not d["accepted"]:
                    click.echo(f"  Rejection: {d['rejection_reason']}")
                continue

            obs = reasoning.get("observation", {})
            diag = reasoning.get("diagnosis", {})
            changes = reasoning.get("search_space_changes", [])

            # --- Observation ---
            direction_sym = "↓" if obs.get("direction") == "minimize" else "↑"
            best = obs.get("best_score")
            best_str = f"{best:.6f}" if best is not None else "N/A"
            click.echo(f"\n  OBSERVED:")
            click.echo(f"    {obs.get('metric_name', '?')}: {best_str} {direction_sym} (cumulative best)")
            rbest = obs.get("round_best_score")
            if rbest is not None:
                click.echo(f"    Round best: {rbest:.6f}")
            delta = obs.get("delta_from_prev")
            if delta is not None:
                sign = "+" if delta > 0 else ""
                click.echo(f"    Delta: {sign}{delta:.6f}")
            click.echo(f"    New best this round: {obs.get('new_best_in_round', '?')}")
            click.echo(f"    Trials: {obs.get('round_completed_trials', '?')}/{obs.get('trials_added', '?')}")
            click.echo(f"    Plateau: {obs.get('plateau_signal', '?')}")
            gap = obs.get("generalization_gap")
            if gap is not None:
                click.echo(f"    Generalization gap: {gap:.4f}")

            top_params = obs.get("top_params", [])
            if top_params:
                click.echo(f"\n    Param importance:")
                for pname, pval in top_params[:5]:
                    click.echo(f"      {pname:25s} {pval:6.1%}")

            best_params = obs.get("best_params", {})
            if best_params:
                _print_best_params(best_params)

            # --- Diagnosis ---
            reasons = diag.get("reasons", [])
            if reasons:
                click.echo(f"\n  DIAGNOSIS:")
                for r in reasons:
                    click.echo(f"    - {r}")

            # --- Action ---
            click.echo(f"\n  ACTION: {d['action']}")

            # --- Search space changes ---
            if changes:
                _print_search_space_changes(changes)

            budget = d.get("proposed_budget")
            if budget is not None:
                click.echo(f"\n  Budget: {budget} trials")

            # --- Justification ---
            click.echo(f"\n  JUSTIFICATION:")
            click.echo(f"    {d['justification']}")

            if not d["accepted"]:
                click.echo(f"\n  REJECTED: {d['rejection_reason']}")

        click.echo()
    finally:
        db.close()


@cli.command(name="export")
@click.argument("name")
def export_cmd(name: str) -> None:
    """Export best params from a campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        campaign = _get_campaign_or_exit(service, name)
        rounds = service.get_rounds(campaign["id"])
        best_summary = None
        for round_row in reversed(rounds):
            if not round_row.get("summary"):
                continue
            summary = _load_json(round_row["summary"])
            if summary.get("best_params"):
                best_summary = summary
                break

        if best_summary:
            click.echo(json.dumps(best_summary["best_params"], indent=2))
        else:
            click.echo("No completed trials yet")
    finally:
        db.close()


@cli.command()
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output file path (default: <name>-report.html)")
def report(name: str, output: str | None) -> None:
    """Generate an HTML report for a campaign."""
    from agentune.report import generate_report

    db = _get_db()
    try:
        html = generate_report(db, name)
        out_path = output or f"{name}-report.html"
        with open(out_path, "w") as f:
            f.write(html)
        click.echo(f"Report written to {out_path}")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()


@cli.command()
@click.argument("name")
@click.option("--dataset", required=True, help="Dataset name (see 'agentune init --help' for available datasets). Metric and direction are stored in the campaign config.")
@click.option("--split-seed", default=42, type=int)
def run(name: str, dataset: str, split_seed: int) -> None:
    """Execute the next study round for a campaign."""
    from agentune.datasets import load_dataset
    from agentune.runner import RoundRunner

    db = _get_db()
    try:
        campaign = _get_campaign_or_exit(CampaignService(db), name)
        split, _ = load_dataset(dataset, seed=split_seed)
        runner = RoundRunner(db, split)
        result = runner.run_next_round(campaign["id"])

        click.echo(f"Round {result.round_number}: {result.status}")
        if result.stop_reason:
            click.echo(f"Stop reason: {result.stop_reason}")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()


@cli.command()
@click.argument("name")
@click.option("--dataset", required=True)
@click.option("--total-trials", required=True, type=int)
@click.option("--backend", default="xgboost", help="Backend to use (default: xgboost)")
@click.option("--split-seed", default=42, type=int)
@click.option("--sampler-seed", default=42, type=int)
def baseline(
    name: str,
    dataset: str,
    total_trials: int,
    backend: str,
    split_seed: int,
    sampler_seed: int,
) -> None:
    """Run a plain Optuna baseline with the same budget for comparison."""
    import optuna
    from agentune.datasets import load_dataset
    from agentune.backends import get_backend

    split, meta = load_dataset(dataset, seed=split_seed)
    backend_cls = get_backend(backend)
    backend_obj = backend_cls()
    search_space = backend_obj.default_search_space()
    objective = backend_obj.create_objective(split, meta["metric"], search_space)

    study = optuna.create_study(
        direction=meta["direction"],
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    start = time.time()
    study.optimize(objective, n_trials=total_trials, show_progress_bar=True)
    wall_time = time.time() - start

    click.echo(f"Baseline '{name}' ({backend}): best={study.best_value:.4f} trials={total_trials} time={wall_time:.1f}s")
    click.echo(f"Best params: {json.dumps(study.best_params, indent=2)}")
