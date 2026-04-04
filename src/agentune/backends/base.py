"""Backend protocol and shared helpers."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import optuna

from agentune.core.models import DatasetSplit, ParamSpec


def suggest_from_param_spec(trial: optuna.Trial, spec: ParamSpec) -> Any:
    """Suggest a value from an Optuna trial using a ParamSpec."""
    if spec.type == "float":
        return trial.suggest_float(spec.name, spec.low, spec.high, log=spec.log)
    elif spec.type == "int":
        return trial.suggest_int(spec.name, int(spec.low), int(spec.high), log=spec.log)
    elif spec.type == "categorical":
        return trial.suggest_categorical(spec.name, spec.choices)
    else:
        raise ValueError(f"Unknown param type: {spec.type}")


class ObjectiveBackend(Protocol):
    def create_objective(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        search_space: list[ParamSpec],
    ) -> Callable[[optuna.Trial], float]: ...

    def default_search_space(self) -> list[ParamSpec]: ...

    def available_params(self) -> list[ParamSpec]: ...

    def param_definitions(self) -> list[ParamSpec]: ...
