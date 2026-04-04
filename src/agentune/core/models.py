"""Data models for agentune campaigns, rounds, summaries, and proposals."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, ClassVar, Literal


@dataclass
class ParamSpec:
    """Defines one hyperparameter's search space."""

    name: str
    type: Literal["float", "int", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list | None = None

    def validate(self) -> None:
        if self.type in ("float", "int"):
            if self.low is None or self.high is None:
                raise ValueError(f"ParamSpec '{self.name}': float/int params require low and high")
            if self.low >= self.high:
                raise ValueError(f"ParamSpec '{self.name}': low must be < high")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(f"ParamSpec '{self.name}': categorical params require choices")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ParamSpec:
        return cls(**d)


@dataclass
class DatasetSplit:
    """Train/validation/test arrays, pre-split."""

    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any
    y_test: Any


@dataclass
class ImprovementCriteria:
    """Defines what counts as 'improvement' for patience-based stop conditions."""

    mode: Literal["strict_better", "min_absolute_delta", "min_relative_delta"]
    threshold: float = 0.0

    def is_improvement(
        self, new_best: float, prev_best: float, direction: str
    ) -> bool:
        if direction == "maximize":
            delta = new_best - prev_best
        else:
            delta = prev_best - new_best

        if delta <= 0:
            return False

        if self.mode == "strict_better":
            return True
        elif self.mode == "min_absolute_delta":
            return delta >= self.threshold
        elif self.mode == "min_relative_delta":
            if prev_best == 0:
                return delta >= self.threshold
            return delta / abs(prev_best) >= self.threshold
        return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ImprovementCriteria:
        return cls(**d)


@dataclass
class StopConditions:
    """Campaign stop conditions. First condition met wins."""

    max_rounds: int | None = None
    max_total_trials: int | None = None
    max_wall_time_seconds: float | None = None
    patience_rounds: int = 3
    target_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StopConditions:
        return cls(**d)


@dataclass
class CampaignConfig:
    """Configuration for creating a new campaign."""

    metric_name: str
    objective_direction: Literal["minimize", "maximize"]
    backend: str
    sampler_config: dict
    initial_search_space: list[ParamSpec]
    improvement_criteria: ImprovementCriteria
    stop_conditions: StopConditions
    trials_per_round: int
    dataset: str
    split_seed: int = 42


@dataclass
class RoundSummary:
    """Immutable, schema-versioned summary of a completed study round."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int = 1
    round_id: int = 0
    campaign_id: int = 0

    # Campaign context
    metric_name: str = ""
    objective_direction: str = ""

    # Performance (cumulative)
    best_score: float | None = None
    best_params: dict | None = None
    delta_from_prev: float | None = None
    total_trials: int = 0
    completed_trials: int = 0

    # Performance (round-local)
    trials_added: int = 0
    round_completed_trials: int = 0
    new_best_in_round: bool = False
    round_best_score: float | None = None

    # Convergence (round-local)
    convergence_curve: list[tuple[int, float]] = field(default_factory=list)
    plateau_signal: bool = False

    # Parameter analysis
    param_importance: dict[str, float] = field(default_factory=dict)
    param_ranges_used: dict[str, tuple] = field(default_factory=dict)

    # Test set evaluation
    test_score: float | None = None

    # Health
    generalization_gap: float | None = None
    failure_rate: float = 0.0
    pruned_rate: float = 0.0

    # Cost
    round_wall_time_seconds: float = 0.0
    total_wall_time_seconds: float = 0.0

    # Lineage
    parent_round_id: int | None = None
    optuna_study_name: str = ""
    action_that_created_this_round: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RoundSummary:
        # Handle tuple conversion for convergence_curve
        if "convergence_curve" in d:
            d["convergence_curve"] = [tuple(p) for p in d["convergence_curve"]]
        return cls(**d)


@dataclass
class ActionProposal:
    """Agent's proposed next action after reviewing a round summary."""

    action: Literal["continue", "narrow_search", "widen_search", "increase_budget", "revise_search", "stop"]
    justification: str
    proposed_search_space: list[dict] | None = None
    proposed_budget: int | None = None
    reference_round_ids: list[int] = field(default_factory=list)
    reasoning: dict | None = None

    def validate(self) -> None:
        if self.action in ("narrow_search", "widen_search", "revise_search") and not self.proposed_search_space:
            raise ValueError(
                f"Action '{self.action}' requires proposed_search_space"
            )
        if self.action == "increase_budget" and self.proposed_budget is None:
            raise ValueError("Action 'increase_budget' requires proposed_budget")
        if not self.reference_round_ids:
            raise ValueError("reference_round_ids must not be empty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ActionProposal:
        return cls(**d)
