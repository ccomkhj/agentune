"""Backend protocol and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class ParamKnowledge:
    """What a hyperparameter does and how to tune it."""
    name: str
    description: str
    role: str  # "regularization", "tree_structure", "sampling", "learning"
    effect_when_high: str
    effect_when_low: str
    interactions: list[str] = field(default_factory=list)


@dataclass
class DiagnosticPattern:
    """A signal pattern and what action it suggests."""
    signal: str  # what to look for
    diagnosis: str  # what it means
    recommended_action: str  # what to do
    params_to_adjust: list[str] = field(default_factory=list)


@dataclass
class TuningGuide:
    """Backend-specific tuning knowledge for the agent."""
    backend_name: str
    overview: str
    params: list[ParamKnowledge]
    diagnostics: list[DiagnosticPattern]
    tuning_order: list[str]  # recommended order to focus on params

    def to_dict(self) -> dict:
        return {
            "backend": self.backend_name,
            "overview": self.overview,
            "params": [
                {
                    "name": p.name,
                    "description": p.description,
                    "role": p.role,
                    "effect_when_high": p.effect_when_high,
                    "effect_when_low": p.effect_when_low,
                    "interactions": p.interactions,
                }
                for p in self.params
            ],
            "diagnostics": [
                {
                    "signal": d.signal,
                    "diagnosis": d.diagnosis,
                    "recommended_action": d.recommended_action,
                    "params_to_adjust": d.params_to_adjust,
                }
                for d in self.diagnostics
            ],
            "tuning_order": self.tuning_order,
        }


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

    def evaluate_test(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        params: dict,
    ) -> float: ...

    def tuning_guide(self) -> TuningGuide: ...
