"""CLI: thin Click wrapper over the core service layer."""

from __future__ import annotations

import json
import os
import sys
import time

import click

from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig,
    ImprovementCriteria,
    StopConditions,
)
from agent_hpo.core.state import CampaignState
from agent_hpo.backends import get_backend


def _get_db() -> Database:
    url = os.environ.get("AGENT_HPO_DB_URL", "postgresql://localhost:5432/agent_hpo")
    db = Database(url)
    db.setup_schema()
    return db


@click.group()
def cli():
    """Agent-driven hyperparameter optimization."""
    pass


@cli.command()
@click.argument("name")
@click.option("--backend", default="xgboost")
@click.option("--metric", required=True)
@click.option("--direction", required=True, type=click.Choice(["minimize", "maximize"]))
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
def init(name, backend, metric, direction, trials_per_round, max_rounds,
         max_trials, max_wall_time, patience, target_score,
         improvement_mode, improvement_threshold, sampler_seed):
    """Create a new optimization campaign."""
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
    )

    try:
        campaign = service.create_campaign(name, config)
        click.echo(f"Campaign '{name}' created (id={campaign['id']}, state={campaign['state']})")
        click.echo(f"Round 1 ready with {trials_per_round} trials budget")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def status(name):
    """Show campaign status and latest round summary."""
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT * FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)

        click.echo(f"Campaign: {campaign['name']}")
        click.echo(f"State: {campaign['state']}")
        click.echo(f"Metric: {campaign['metric_name']} ({campaign['objective_direction']})")
        click.echo(f"Backend: {campaign['backend']}")

        rounds = service.get_rounds(campaign["id"])
        click.echo(f"Rounds: {len(rounds)}")
        if rounds:
            latest = rounds[-1]
            click.echo(f"Latest round: #{latest['round_number']} ({latest['state']})")
            if latest.get("summary"):
                summary = latest["summary"]
                if isinstance(summary, str):
                    summary = json.loads(summary)
                click.echo(f"Best score: {summary.get('best_score', 'N/A')}")
    finally:
        db.close()


@cli.command()
@click.argument("name")
def pause(name):
    """Pause a running campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)
        service.transition_campaign(campaign["id"], CampaignState.PAUSE_REQUESTED)
        click.echo(f"Campaign '{name}' pause requested")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def resume(name):
    """Resume a paused campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        click.echo(f"Campaign '{name}' resumed")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def stop(name):
    """Manually stop a campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT * FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)

        rounds = service.get_rounds(campaign["id"])
        active_round = rounds[-1] if rounds else None
        if active_round and active_round["state"] in ("RUNNING", "SUMMARIZING"):
            click.echo(
                f"Cannot stop: round {active_round['round_number']} is {active_round['state']}. "
                f"Use 'agent-hpo pause' to request a safe stop after the round completes.",
                err=True,
            )
            sys.exit(1)

        service.transition_campaign(campaign["id"], CampaignState.STOPPED)
        click.echo(f"Campaign '{name}' stopped")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.argument("name")
def history(name):
    """Show full campaign history: rounds and decisions."""
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)

        data = service.get_campaign_history(campaign["id"])
        for r in data["rounds"]:
            click.echo(f"Round #{r['round_number']}: {r['state']} (study: {r['optuna_study_name']})")
        for d in data["decisions"]:
            status = "accepted" if d["accepted"] else f"rejected: {d['rejection_reason']}"
            click.echo(f"Decision: {d['action']} — {status}")
    finally:
        db.close()


@cli.command(name="export")
@click.argument("name")
def export_cmd(name):
    """Export best params from a campaign."""
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)

        rounds = service.get_rounds(campaign["id"])
        best_summary = None
        for r in reversed(rounds):
            if r.get("summary"):
                s = r["summary"]
                if isinstance(s, str):
                    s = json.loads(s)
                if s.get("best_params"):
                    best_summary = s
                    break

        if best_summary:
            click.echo(json.dumps(best_summary["best_params"], indent=2))
        else:
            click.echo("No completed trials yet")
    finally:
        db.close()


@cli.command()
@click.argument("name")
@click.option("--dataset", required=True, help="Dataset name: breast_cancer, california_housing, digits")
@click.option("--split-seed", default=42, type=int)
def run(name, dataset, split_seed):
    """Execute the next study round for a campaign."""
    from agent_hpo.datasets import load_dataset
    from agent_hpo.runner import RoundRunner

    db = _get_db()
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)

        split, _ = load_dataset(dataset, seed=split_seed)
        runner = RoundRunner(db, split)
        result = runner.run_next_round(campaign["id"])

        click.echo(f"Round {result.round_number}: {result.status}")
        if result.stop_reason:
            click.echo(f"Stop reason: {result.stop_reason}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        db.close()


@cli.command()
@click.argument("name")
@click.option("--dataset", required=True)
@click.option("--total-trials", required=True, type=int)
@click.option("--split-seed", default=42, type=int)
@click.option("--sampler-seed", default=42, type=int)
def baseline(name, dataset, total_trials, split_seed, sampler_seed):
    """Run a plain Optuna baseline with the same budget for comparison."""
    import optuna
    from agent_hpo.datasets import load_dataset
    from agent_hpo.backends.xgboost import XGBoostBackend

    split, meta = load_dataset(dataset, seed=split_seed)
    backend = XGBoostBackend()
    search_space = backend.default_search_space()
    objective = backend.create_objective(split, meta["metric"], search_space)

    study = optuna.create_study(
        direction=meta["direction"],
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    start = time.time()
    study.optimize(objective, n_trials=total_trials, show_progress_bar=True)
    wall_time = time.time() - start

    click.echo(f"Baseline '{name}': best={study.best_value:.4f} trials={total_trials} time={wall_time:.1f}s")
    click.echo(f"Best params: {json.dumps(study.best_params, indent=2)}")
