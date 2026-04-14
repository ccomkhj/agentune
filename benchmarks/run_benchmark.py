"""Benchmark: agent-driven campaign vs plain Optuna baseline."""

from __future__ import annotations

import json
import time

import click
import optuna

from agentune.datasets import load_dataset, DATASETS
from agentune.backends.xgboost import XGBoostBackend
from agentune.core.models import ParamSpec


@click.command()
@click.option("--dataset", type=click.Choice(list(DATASETS.keys())), required=True)
@click.option("--total-trials", default=200, type=int)
@click.option("--seeds", default="42,123,456")
def main(dataset: str, total_trials: int, seeds: str):
    """Run baseline benchmark: plain Optuna with fixed budget."""
    seed_list = [int(s) for s in seeds.split(",")]
    backend = XGBoostBackend()
    search_space = backend.default_search_space()

    results = []
    for seed in seed_list:
        split, meta = load_dataset(dataset, seed=seed)
        objective = backend.create_objective(split, meta["metric"], search_space)

        study = optuna.create_study(
            direction=meta["direction"],
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        start = time.time()
        study.optimize(objective, n_trials=total_trials, show_progress_bar=True)
        wall_time = time.time() - start

        # Test set evaluation
        best_params = study.best_params
        from xgboost import XGBClassifier, XGBRegressor
        if meta["metric"] == "rmse":
            model = XGBRegressor(**best_params, verbosity=0, nthread=1)
        else:
            model = XGBClassifier(**best_params, verbosity=0, nthread=1)
        model.fit(split.X_train, split.y_train)

        from sklearn.metrics import accuracy_score, mean_squared_error
        if meta["metric"] == "accuracy":
            test_score = accuracy_score(split.y_test, model.predict(split.X_test))
        else:
            test_score = mean_squared_error(split.y_test, model.predict(split.X_test)) ** 0.5

        results.append({
            "seed": seed,
            "best_val_score": study.best_value,
            "test_score": test_score,
            "total_trials": total_trials,
            "wall_time_seconds": wall_time,
        })

        click.echo(f"Seed {seed}: val={study.best_value:.4f} test={test_score:.4f} time={wall_time:.1f}s")

    click.echo("\n--- Summary ---")
    click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
