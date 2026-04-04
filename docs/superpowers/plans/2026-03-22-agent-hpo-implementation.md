# Agent-HPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pip-installable library that combines Optuna-based HPO with an LLM agent layer, where the agent decides between bounded study rounds and the optimizer runs deterministically within each round.

**Architecture:** Core service layer accessed by CLI (thin wrapper) and MCP server (agent control plane). Optuna handles trial-level optimization. Campaign schema (Postgres, 3 tables) tracks agent control state. Lease-based concurrency. Immutable round summaries drive agent reasoning.

**Tech Stack:** Python 3.11+, Optuna, XGBoost, PostgreSQL, psycopg (v3), Click (CLI), mcp (MCP SDK), scikit-learn (benchmarks), pytest

**Spec:** `docs/superpowers/specs/2026-03-22-agent-hpo-design.md`

---

## File Structure

```
agent_param_optimization/
├── pyproject.toml
├── src/
│   └── agent_hpo/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── models.py        # All dataclasses: ParamSpec, CampaignConfig, RoundSummary, etc.
│       │   ├── state.py         # CampaignState, RoundState enums + transition validators
│       │   ├── db.py            # Connection pool, schema setup, session helpers
│       │   ├── campaign.py      # CampaignService: CRUD, round management, proposal validation
│       │   └── locking.py       # Lease/claim: acquire, refresh, release
│       ├── backends/
│       │   ├── __init__.py      # BACKEND_REGISTRY dict
│       │   ├── base.py          # ObjectiveBackend protocol, suggest_from_param_spec helper
│       │   └── xgboost.py       # XGBoostBackend
│       ├── summarizer.py        # RoundSummarizer: Optuna study → RoundSummary
│       ├── scheduler.py         # Scheduler: stop conditions, budget clipping
│       ├── runner.py            # RoundRunner: full round orchestration (Optuna RDBStorage, backend, summarizer)
│       ├── datasets.py          # DatasetLoader: breast_cancer, california_housing, digits (packaged)
│       ├── cli.py               # Click CLI: init, run, status, history, pause, resume, stop, export, baseline
│       ├── mcp_server.py        # MCP server: 5 tools over core
│       └── skills/              # Packaged as package data
│           ├── hpo-overview.md
│           ├── hpo-interpret-summary.md
│           └── hpo-action-guidelines.md
├── benchmarks/
│   └── run_benchmark.py         # BenchmarkRunner: agent vs baseline, 3 seeds, report
├── tests/
│   ├── conftest.py              # Fixtures: test DB, cleanup, sample campaigns
│   ├── test_models.py
│   ├── test_state.py
│   ├── test_db.py
│   ├── test_campaign.py
│   ├── test_locking.py
│   ├── test_backend_xgboost.py
│   ├── test_summarizer.py
│   ├── test_scheduler.py
│   ├── test_runner.py
│   ├── test_cli.py
│   └── test_mcp_server.py
└── alembic/                     # DB migrations (optional v0: raw SQL in db.py)
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/agent_hpo/__init__.py`
- Create: `src/agent_hpo/core/__init__.py`
- Create: `src/agent_hpo/backends/__init__.py`

- [ ] **Step 1: Create pyproject.toml**


```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agent-hpo"
version = "0.1.0"
description = "Agent-driven hyperparameter optimization with Optuna"
requires-python = ">=3.11"
dependencies = [
    "optuna>=3.6,<4",
    "psycopg[binary]>=3.1,<4",
    "xgboost>=2.0,<3",
    "scikit-learn>=1.4,<2",
    "click>=8.1,<9",
    "mcp>=1.0,<2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
]

[project.scripts]
agent-hpo = "agent_hpo.cli:cli"

[tool.hatch.build.targets.wheel]
packages = ["src/agent_hpo"]

[tool.hatch.build.targets.wheel.shared-data]

[tool.hatch.build.targets.sdist]

[tool.hatch.build]
artifacts = ["src/agent_hpo/skills/*.md"]
```

- [ ] **Step 2: Create package init files**

`src/agent_hpo/__init__.py`:
```python
"""Agent-driven hyperparameter optimization with Optuna."""

__version__ = "0.1.0"
```

`src/agent_hpo/core/__init__.py`:
```python
"""Core service layer: models, state machines, campaign management."""
```

`src/agent_hpo/backends/__init__.py`:
```python
"""Model backend registry."""

BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str) -> type:
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[name]
```

- [ ] **Step 3: Install in dev mode and verify**

Run: `cd /Users/huijokim/personal/agent_param_optimization && pip install -e ".[dev]"`
Expected: Successful install, `agent-hpo --help` shows an error (cli.py doesn't exist yet — that's fine)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: project scaffolding with pyproject.toml and package structure"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/agent_hpo/core/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write tests for data models**

```python
# tests/test_models.py
import pytest
from agent_hpo.core.models import (
    ParamSpec,
    CampaignConfig,
    StopConditions,
    ImprovementCriteria,
    RoundSummary,
    ActionProposal,
    DatasetSplit,
)
import numpy as np


class TestParamSpec:
    def test_float_param(self):
        p = ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True)
        assert p.name == "learning_rate"
        assert p.type == "float"
        assert p.log is True

    def test_int_param(self):
        p = ParamSpec(name="max_depth", type="int", low=1, high=15)
        assert p.type == "int"
        assert p.low == 1

    def test_categorical_param(self):
        p = ParamSpec(name="booster", type="categorical", choices=["gbtree", "dart"])
        assert p.choices == ["gbtree", "dart"]

    def test_float_param_requires_low_high(self):
        p = ParamSpec(name="lr", type="float", low=None, high=1.0)
        with pytest.raises(ValueError):
            p.validate()

    def test_categorical_requires_choices(self):
        p = ParamSpec(name="b", type="categorical", choices=None)
        with pytest.raises(ValueError):
            p.validate()


class TestImprovementCriteria:
    def test_strict_better_maximize(self):
        ic = ImprovementCriteria(mode="strict_better", threshold=0.0)
        assert ic.is_improvement(0.91, 0.90, "maximize") is True
        assert ic.is_improvement(0.90, 0.90, "maximize") is False

    def test_strict_better_minimize(self):
        ic = ImprovementCriteria(mode="strict_better", threshold=0.0)
        assert ic.is_improvement(0.89, 0.90, "minimize") is True
        assert ic.is_improvement(0.90, 0.90, "minimize") is False

    def test_min_absolute_delta(self):
        ic = ImprovementCriteria(mode="min_absolute_delta", threshold=0.01)
        assert ic.is_improvement(0.92, 0.90, "maximize") is True
        assert ic.is_improvement(0.905, 0.90, "maximize") is False

    def test_min_relative_delta(self):
        ic = ImprovementCriteria(mode="min_relative_delta", threshold=0.05)
        # 5% of 0.90 = 0.045, so 0.95 (delta 0.05) passes
        assert ic.is_improvement(0.95, 0.90, "maximize") is True
        assert ic.is_improvement(0.94, 0.90, "maximize") is False

    def test_min_relative_delta_zero_prev(self):
        """Falls back to absolute delta when prev_best is 0."""
        ic = ImprovementCriteria(mode="min_relative_delta", threshold=0.05)
        assert ic.is_improvement(0.06, 0.0, "maximize") is True
        assert ic.is_improvement(0.04, 0.0, "maximize") is False


class TestStopConditions:
    def test_serialization_roundtrip(self):
        sc = StopConditions(
            max_rounds=10,
            max_total_trials=500,
            max_wall_time_seconds=3600.0,
            patience_rounds=3,
            target_score=0.95,
        )
        d = sc.to_dict()
        sc2 = StopConditions.from_dict(d)
        assert sc == sc2

    def test_optional_fields_none(self):
        sc = StopConditions(
            max_rounds=None,
            max_total_trials=None,
            max_wall_time_seconds=None,
            patience_rounds=3,
            target_score=None,
        )
        assert sc.max_rounds is None


class TestActionProposal:
    def test_continue_valid(self):
        ap = ActionProposal(
            action="continue",
            justification="Score improved by 0.02 in round 3",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[3],
        )
        ap.validate()

    def test_narrow_requires_search_space(self):
        ap = ActionProposal(
            action="narrow_search",
            justification="Focus on high-importance params",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[3],
        )
        with pytest.raises(ValueError, match="proposed_search_space"):
            ap.validate()

    def test_increase_budget_requires_budget(self):
        ap = ActionProposal(
            action="increase_budget",
            justification="Need more trials",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[3],
        )
        with pytest.raises(ValueError, match="proposed_budget"):
            ap.validate()

    def test_empty_references_rejected(self):
        ap = ActionProposal(
            action="continue",
            justification="keep going",
            proposed_search_space=None,
            proposed_budget=None,
            reference_round_ids=[],
        )
        with pytest.raises(ValueError, match="reference_round_ids"):
            ap.validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/huijokim/personal/agent_param_optimization && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_hpo.core.models'`

- [ ] **Step 3: Implement models**

```python
# src/agent_hpo/core/models.py
"""Data models for agent-hpo campaigns, rounds, summaries, and proposals."""

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

    action: Literal["continue", "narrow_search", "widen_search", "increase_budget", "stop"]
    justification: str
    proposed_search_space: list[dict] | None = None
    proposed_budget: int | None = None
    reference_round_ids: list[int] = field(default_factory=list)

    def validate(self) -> None:
        if self.action in ("narrow_search", "widen_search") and not self.proposed_search_space:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/core/models.py tests/test_models.py
git commit -m "feat: data models with validation and serialization"
```

---

### Task 3: State Machines

**Files:**
- Create: `src/agent_hpo/core/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write tests for state transitions**

```python
# tests/test_state.py
import pytest
from agent_hpo.core.state import (
    CampaignState,
    RoundState,
    InvalidTransitionError,
    validate_campaign_transition,
    validate_round_transition,
)


class TestCampaignState:
    def test_created_to_running(self):
        validate_campaign_transition(CampaignState.CREATED, CampaignState.RUNNING)

    def test_running_to_completed(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.COMPLETED)

    def test_running_to_pause_requested(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.PAUSE_REQUESTED)

    def test_pause_requested_to_paused(self):
        validate_campaign_transition(CampaignState.PAUSE_REQUESTED, CampaignState.PAUSED)

    def test_paused_to_running(self):
        validate_campaign_transition(CampaignState.PAUSED, CampaignState.RUNNING)

    def test_running_to_failed(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.FAILED)

    def test_running_to_stopped(self):
        validate_campaign_transition(CampaignState.RUNNING, CampaignState.STOPPED)

    def test_invalid_created_to_completed(self):
        with pytest.raises(InvalidTransitionError):
            validate_campaign_transition(CampaignState.CREATED, CampaignState.COMPLETED)

    def test_invalid_stopped_to_running(self):
        with pytest.raises(InvalidTransitionError):
            validate_campaign_transition(CampaignState.STOPPED, CampaignState.RUNNING)

    def test_invalid_completed_to_running(self):
        with pytest.raises(InvalidTransitionError):
            validate_campaign_transition(CampaignState.COMPLETED, CampaignState.RUNNING)

    def test_terminal_states(self):
        assert CampaignState.COMPLETED.is_terminal
        assert CampaignState.FAILED.is_terminal
        assert CampaignState.STOPPED.is_terminal
        assert not CampaignState.RUNNING.is_terminal


class TestRoundState:
    def test_proposed_to_running(self):
        validate_round_transition(RoundState.PROPOSED, RoundState.RUNNING)

    def test_running_to_summarizing(self):
        validate_round_transition(RoundState.RUNNING, RoundState.SUMMARIZING)

    def test_summarizing_to_awaiting_agent(self):
        validate_round_transition(RoundState.SUMMARIZING, RoundState.AWAITING_AGENT)

    def test_awaiting_agent_to_resolved(self):
        validate_round_transition(RoundState.AWAITING_AGENT, RoundState.RESOLVED)

    def test_awaiting_agent_to_closed(self):
        validate_round_transition(RoundState.AWAITING_AGENT, RoundState.CLOSED)

    def test_running_to_failed(self):
        validate_round_transition(RoundState.RUNNING, RoundState.FAILED)

    def test_summarizing_to_failed(self):
        validate_round_transition(RoundState.SUMMARIZING, RoundState.FAILED)

    def test_failed_to_retrying(self):
        validate_round_transition(RoundState.FAILED, RoundState.RETRYING)

    def test_retrying_to_running(self):
        validate_round_transition(RoundState.RETRYING, RoundState.RUNNING)

    def test_retrying_to_summarizing(self):
        validate_round_transition(RoundState.RETRYING, RoundState.SUMMARIZING)

    def test_invalid_proposed_to_resolved(self):
        with pytest.raises(InvalidTransitionError):
            validate_round_transition(RoundState.PROPOSED, RoundState.RESOLVED)

    def test_terminal_states(self):
        assert RoundState.RESOLVED.is_terminal
        assert RoundState.CLOSED.is_terminal
        assert not RoundState.RUNNING.is_terminal
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement state machines**

```python
# src/agent_hpo/core/state.py
"""Campaign and round state machines with validated transitions."""

from __future__ import annotations

from enum import Enum


class InvalidTransitionError(Exception):
    pass


class CampaignState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"

    @property
    def is_terminal(self) -> bool:
        return self in (self.COMPLETED, self.FAILED, self.STOPPED)


CAMPAIGN_TRANSITIONS: dict[CampaignState, set[CampaignState]] = {
    CampaignState.CREATED: {CampaignState.RUNNING},
    CampaignState.RUNNING: {
        CampaignState.PAUSE_REQUESTED,
        CampaignState.COMPLETED,
        CampaignState.FAILED,
        CampaignState.STOPPED,
    },
    CampaignState.PAUSE_REQUESTED: {CampaignState.PAUSED},
    CampaignState.PAUSED: {CampaignState.RUNNING},
    CampaignState.COMPLETED: set(),
    CampaignState.FAILED: set(),
    CampaignState.STOPPED: set(),
}


def validate_campaign_transition(
    from_state: CampaignState, to_state: CampaignState
) -> None:
    allowed = CAMPAIGN_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidTransitionError(
            f"Campaign transition {from_state.value} → {to_state.value} is not allowed. "
            f"Allowed from {from_state.value}: {[s.value for s in allowed]}"
        )


class RoundState(Enum):
    PROPOSED = "PROPOSED"
    RUNNING = "RUNNING"
    SUMMARIZING = "SUMMARIZING"
    AWAITING_AGENT = "AWAITING_AGENT"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"

    @property
    def is_terminal(self) -> bool:
        return self in (self.RESOLVED, self.CLOSED)


ROUND_TRANSITIONS: dict[RoundState, set[RoundState]] = {
    RoundState.PROPOSED: {RoundState.RUNNING},
    RoundState.RUNNING: {RoundState.SUMMARIZING, RoundState.FAILED},
    RoundState.SUMMARIZING: {RoundState.AWAITING_AGENT, RoundState.FAILED},
    RoundState.AWAITING_AGENT: {RoundState.RESOLVED, RoundState.CLOSED},
    RoundState.RESOLVED: set(),
    RoundState.CLOSED: set(),
    RoundState.FAILED: {RoundState.RETRYING},
    RoundState.RETRYING: {RoundState.RUNNING, RoundState.SUMMARIZING},
}


def validate_round_transition(
    from_state: RoundState, to_state: RoundState
) -> None:
    allowed = ROUND_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise InvalidTransitionError(
            f"Round transition {from_state.value} → {to_state.value} is not allowed. "
            f"Allowed from {from_state.value}: {[s.value for s in allowed]}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/core/state.py tests/test_state.py
git commit -m "feat: campaign and round state machines with transition validation"
```

---

### Task 4: Database Layer

**Files:**
- Create: `src/agent_hpo/core/db.py`
- Create: `tests/conftest.py`
- Create: `tests/test_db.py`

**Prerequisites:** A running Postgres instance. Tests use a `agent_hpo_test` database.

- [ ] **Step 1: Write tests for DB layer**

```python
# tests/test_db.py
import pytest
from agent_hpo.core.db import Database


@pytest.fixture
def db(test_db_url):
    """Uses the test_db_url fixture from conftest.py."""
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


class TestDatabase:
    def test_setup_schema_creates_tables(self, db):
        with db.connection() as conn:
            cur = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN ('campaigns', 'study_rounds', 'agent_decisions') "
                "ORDER BY table_name"
            )
            tables = [row[0] for row in cur.fetchall()]
        assert tables == ["agent_decisions", "campaigns", "study_rounds"]

    def test_setup_schema_is_idempotent(self, db):
        db.setup_schema()
        db.setup_schema()
        with db.connection() as conn:
            cur = conn.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'campaigns'"
            )
            assert cur.fetchone()[0] == 1
```

- [ ] **Step 2: Write conftest.py with test DB fixture**

```python
# tests/conftest.py
import os
import pytest
import psycopg

TEST_DB_URL = os.environ.get(
    "AGENT_HPO_TEST_DB_URL",
    "postgresql://localhost:5432/agent_hpo_test",
)


@pytest.fixture(scope="session")
def test_db_url():
    """Provide test database URL."""
    # Ensure the test database exists
    base_url = TEST_DB_URL.rsplit("/", 1)[0] + "/postgres"
    db_name = TEST_DB_URL.rsplit("/", 1)[1].split("?")[0]
    with psycopg.connect(base_url, autocommit=True) as conn:
        cur = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        )
        if not cur.fetchone():
            conn.execute(f"CREATE DATABASE {db_name}")
    return TEST_DB_URL


@pytest.fixture(autouse=True)
def clean_db(test_db_url):
    """Truncate all tables before each test."""
    yield
    with psycopg.connect(test_db_url) as conn:
        conn.execute("TRUNCATE agent_decisions, study_rounds, campaigns CASCADE")
        conn.commit()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement database layer**

```python
# src/agent_hpo/core/db.py
"""Database connection pool and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS campaigns (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    state           TEXT NOT NULL DEFAULT 'CREATED',
    pause_requested BOOLEAN NOT NULL DEFAULT false,
    metric_name     TEXT NOT NULL,
    objective_direction TEXT NOT NULL,
    backend         TEXT NOT NULL,
    sampler_config  JSONB NOT NULL,
    initial_search_space JSONB NOT NULL,
    improvement_criteria JSONB NOT NULL,
    stop_conditions JSONB NOT NULL,
    trials_per_round INT NOT NULL,
    claimed_by      TEXT,
    claim_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS study_rounds (
    id              SERIAL PRIMARY KEY,
    campaign_id     INT NOT NULL REFERENCES campaigns(id),
    round_number    INT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'PROPOSED',
    failed_from     TEXT,
    parent_round_id INT REFERENCES study_rounds(id),
    optuna_study_name TEXT NOT NULL,
    search_space    JSONB NOT NULL,
    budget          INT NOT NULL,
    trial_offset    INT NOT NULL,
    trial_end       INT,
    summary         JSONB,
    summary_schema_version INT,
    retry_count     INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(campaign_id, round_number)
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id              SERIAL PRIMARY KEY,
    campaign_id     INT NOT NULL REFERENCES campaigns(id),
    round_id        INT NOT NULL REFERENCES study_rounds(id),
    action          TEXT NOT NULL,
    justification   TEXT NOT NULL,
    proposed_search_space JSONB,
    proposed_budget INT,
    reference_round_ids JSONB NOT NULL,
    accepted        BOOLEAN NOT NULL,
    rejection_reason TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Database:
    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: psycopg.Connection | None = None

    @property
    def optuna_storage_url(self) -> str:
        """Return the DB URL formatted for Optuna's RDBStorage."""
        return self._db_url

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._db_url, row_factory=dict_row)
        return self._conn

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection, None, None]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def setup_schema(self) -> None:
        with self.connection() as conn:
            conn.execute(SCHEMA_SQL)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_db.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent_hpo/core/db.py tests/conftest.py tests/test_db.py
git commit -m "feat: database layer with schema setup and connection management"
```

---

### Task 5: Lease/Claim Locking

**Files:**
- Create: `src/agent_hpo/core/locking.py`
- Create: `tests/test_locking.py`

- [ ] **Step 1: Write tests for lease model**

```python
# tests/test_locking.py
import pytest
from agent_hpo.core.db import Database
from agent_hpo.core.locking import LeaseManager, LeaseError


@pytest.fixture
def db(test_db_url):
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


@pytest.fixture
def campaign_id(db):
    """Insert a minimal campaign row and return its id."""
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO campaigns (name, metric_name, objective_direction, backend, "
            "sampler_config, initial_search_space, improvement_criteria, stop_conditions, trials_per_round) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            ("test", "accuracy", "maximize", "xgboost", "{}", "[]", "{}", "{}", 50),
        )
        return cur.fetchone()["id"]


class TestLeaseManager:
    def test_acquire_and_release(self, db, campaign_id):
        lm = LeaseManager(db, worker_id="worker-1")
        lm.acquire(campaign_id)
        # Verify claimed
        with db.connection() as conn:
            cur = conn.execute("SELECT claimed_by FROM campaigns WHERE id = %s", (campaign_id,))
            assert cur.fetchone()["claimed_by"] == "worker-1"
        lm.release(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claimed_by FROM campaigns WHERE id = %s", (campaign_id,))
            assert cur.fetchone()["claimed_by"] is None

    def test_acquire_fails_if_already_claimed(self, db, campaign_id):
        lm1 = LeaseManager(db, worker_id="worker-1")
        lm2 = LeaseManager(db, worker_id="worker-2")
        lm1.acquire(campaign_id)
        with pytest.raises(LeaseError):
            lm2.acquire(campaign_id)
        lm1.release(campaign_id)

    def test_acquire_succeeds_if_lease_expired(self, db, campaign_id):
        lm1 = LeaseManager(db, worker_id="worker-1")
        lm1.acquire(campaign_id)
        # Manually expire the lease
        with db.connection() as conn:
            conn.execute(
                "UPDATE campaigns SET claim_expires_at = now() - interval '1 minute' WHERE id = %s",
                (campaign_id,),
            )
        lm2 = LeaseManager(db, worker_id="worker-2")
        lm2.acquire(campaign_id)  # Should succeed
        with db.connection() as conn:
            cur = conn.execute("SELECT claimed_by FROM campaigns WHERE id = %s", (campaign_id,))
            assert cur.fetchone()["claimed_by"] == "worker-2"
        lm2.release(campaign_id)

    def test_refresh_extends_lease(self, db, campaign_id):
        lm = LeaseManager(db, worker_id="worker-1")
        lm.acquire(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claim_expires_at FROM campaigns WHERE id = %s", (campaign_id,))
            first_expiry = cur.fetchone()["claim_expires_at"]
        lm.refresh(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claim_expires_at FROM campaigns WHERE id = %s", (campaign_id,))
            second_expiry = cur.fetchone()["claim_expires_at"]
        assert second_expiry > first_expiry
        lm.release(campaign_id)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_locking.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement lease manager**

```python
# src/agent_hpo/core/locking.py
"""Lease-based concurrency control for campaign execution."""

from __future__ import annotations

import uuid
from datetime import timedelta

from agent_hpo.core.db import Database

LEASE_DURATION = timedelta(minutes=15)
REFRESH_INTERVAL = timedelta(minutes=5)


class LeaseError(Exception):
    pass


class LeaseManager:
    def __init__(self, db: Database, worker_id: str | None = None) -> None:
        self._db = db
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

    def acquire(self, campaign_id: int) -> None:
        with self._db.connection() as conn:
            cur = conn.execute(
                "UPDATE campaigns SET claimed_by = %s, "
                "claim_expires_at = now() + %s "
                "WHERE id = %s AND (claimed_by IS NULL OR claim_expires_at < now())",
                (self.worker_id, LEASE_DURATION, campaign_id),
            )
            if cur.rowcount == 0:
                raise LeaseError(
                    f"Campaign {campaign_id} is already claimed by another worker"
                )

    def refresh(self, campaign_id: int) -> None:
        with self._db.connection() as conn:
            cur = conn.execute(
                "UPDATE campaigns SET claim_expires_at = now() + %s "
                "WHERE id = %s AND claimed_by = %s",
                (LEASE_DURATION, campaign_id, self.worker_id),
            )
            if cur.rowcount == 0:
                raise LeaseError(
                    f"Cannot refresh lease: campaign {campaign_id} not claimed by {self.worker_id}"
                )

    def release(self, campaign_id: int) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE campaigns SET claimed_by = NULL, claim_expires_at = NULL "
                "WHERE id = %s AND claimed_by = %s",
                (campaign_id, self.worker_id),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_locking.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/core/locking.py tests/test_locking.py
git commit -m "feat: lease-based concurrency control for campaign execution"
```

---

### Task 6: Campaign Service (Core)

**Files:**
- Create: `src/agent_hpo/core/campaign.py`
- Create: `tests/test_campaign.py`

This is the largest task. It covers: campaign CRUD, round creation (including system-generated round 1), proposal validation (guardrails, cooldown), and state transitions.

- [ ] **Step 1: Write tests for campaign creation and round 1 bootstrap**


```python
# tests/test_campaign.py
import json
import pytest
from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, ActionProposal,
)
from agent_hpo.core.state import CampaignState, RoundState


@pytest.fixture
def db(test_db_url):
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


@pytest.fixture
def service(db):
    return CampaignService(db)


@pytest.fixture
def sample_config():
    return CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=[
            ParamSpec(name="max_depth", type="int", low=1, high=15),
            ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
        ],
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=10, patience_rounds=3),
        trials_per_round=50,
    )


class TestCampaignCreation:
    def test_create_campaign(self, service, sample_config):
        campaign = service.create_campaign("test-campaign", sample_config)
        assert campaign["name"] == "test-campaign"
        assert campaign["state"] == "CREATED"

    def test_create_campaign_creates_round_1(self, service, sample_config):
        campaign = service.create_campaign("test-campaign", sample_config)
        rounds = service.get_rounds(campaign["id"])
        assert len(rounds) == 1
        assert rounds[0]["round_number"] == 1
        assert rounds[0]["state"] == "PROPOSED"
        assert rounds[0]["trial_offset"] == 0

    def test_duplicate_name_rejected(self, service, sample_config):
        service.create_campaign("test-campaign", sample_config)
        with pytest.raises(Exception):
            service.create_campaign("test-campaign", sample_config)


class TestStateTransitions:
    def test_transition_campaign_state(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        updated = service.get_campaign(campaign["id"])
        assert updated["state"] == "RUNNING"

    def test_invalid_transition_rejected(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        with pytest.raises(Exception):
            service.transition_campaign(campaign["id"], CampaignState.COMPLETED)

    def test_transition_round_state(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        rounds = service.get_rounds(campaign["id"])
        service.transition_round(rounds[0]["id"], RoundState.RUNNING)
        updated_round = service.get_round(rounds[0]["id"])
        assert updated_round["state"] == "RUNNING"


class TestProposalValidation:
    def test_continue_creates_round_reusing_study(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        # Simulate round 1 completing
        rounds = service.get_rounds(campaign["id"])
        r1 = rounds[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_round(r1["id"], RoundState.RUNNING)
        service.complete_round_execution(r1["id"], trial_end=50)
        service.transition_round(r1["id"], RoundState.SUMMARIZING)
        service.write_summary(r1["id"], {"schema_version": 1})
        service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

        proposal = ActionProposal(
            action="continue",
            justification="Score still improving",
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["round_number"] == 2
        assert r2["optuna_study_name"] == r1["optuna_study_name"]
        assert r2["trial_offset"] == 50

    def test_narrow_creates_new_study(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        rounds = service.get_rounds(campaign["id"])
        r1 = rounds[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_round(r1["id"], RoundState.RUNNING)
        service.complete_round_execution(r1["id"], trial_end=50)
        service.transition_round(r1["id"], RoundState.SUMMARIZING)
        service.write_summary(r1["id"], {"schema_version": 1})
        service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

        proposal = ActionProposal(
            action="narrow_search",
            justification="Focus on learning_rate",
            proposed_search_space=[
                {"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": True},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["optuna_study_name"] != r1["optuna_study_name"]
        assert r2["parent_round_id"] == r1["id"]
        assert r2["trial_offset"] == 0

    def test_widen_creates_new_study(self, service, sample_config):
        campaign = service.create_campaign("test-widen", sample_config)
        rounds = service.get_rounds(campaign["id"])
        r1 = rounds[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_round(r1["id"], RoundState.RUNNING)
        service.complete_round_execution(r1["id"], trial_end=50)
        service.transition_round(r1["id"], RoundState.SUMMARIZING)
        service.write_summary(r1["id"], {"schema_version": 1})
        service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

        proposal = ActionProposal(
            action="widen_search",
            justification="Explore broader range for max_depth",
            proposed_search_space=[
                {"name": "max_depth", "type": "int", "low": 1, "high": 20},
                {"name": "learning_rate", "type": "float", "low": 0.0001, "high": 2.0, "log": True},
            ],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["optuna_study_name"] != r1["optuna_study_name"]
        assert r2["parent_round_id"] == r1["id"]
        assert r2["trial_offset"] == 0

    def test_increase_budget_reuses_study(self, service, sample_config):
        campaign = service.create_campaign("test-budget", sample_config)
        rounds = service.get_rounds(campaign["id"])
        r1 = rounds[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_round(r1["id"], RoundState.RUNNING)
        service.complete_round_execution(r1["id"], trial_end=50)
        service.transition_round(r1["id"], RoundState.SUMMARIZING)
        service.write_summary(r1["id"], {"schema_version": 1})
        service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

        proposal = ActionProposal(
            action="increase_budget",
            justification="Need more trials, still converging",
            proposed_budget=100,
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        r2 = rounds[1]
        assert r2["optuna_study_name"] == r1["optuna_study_name"]
        assert r2["budget"] == 100
        assert r2["trial_offset"] == 50

    def test_stop_does_not_create_round(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        rounds = service.get_rounds(campaign["id"])
        r1 = rounds[0]
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)
        service.transition_round(r1["id"], RoundState.RUNNING)
        service.complete_round_execution(r1["id"], trial_end=50)
        service.transition_round(r1["id"], RoundState.SUMMARIZING)
        service.write_summary(r1["id"], {"schema_version": 1})
        service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

        proposal = ActionProposal(
            action="stop",
            justification="Diminishing returns",
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], proposal)
        assert result["accepted"] is True

        rounds = service.get_rounds(campaign["id"])
        assert len(rounds) == 1  # no new round

        updated = service.get_campaign(campaign["id"])
        assert updated["state"] == "COMPLETED"

    def test_cooldown_rejects_immediate_reversal(self, service, sample_config):
        campaign = service.create_campaign("test", sample_config)
        service.transition_campaign(campaign["id"], CampaignState.RUNNING)

        # Complete round 1
        r1 = service.get_rounds(campaign["id"])[0]
        service.transition_round(r1["id"], RoundState.RUNNING)
        service.complete_round_execution(r1["id"], trial_end=50)
        service.transition_round(r1["id"], RoundState.SUMMARIZING)
        service.write_summary(r1["id"], {"schema_version": 1})
        service.transition_round(r1["id"], RoundState.AWAITING_AGENT)

        # Narrow in round 2
        narrow_proposal = ActionProposal(
            action="narrow_search",
            justification="Focus",
            proposed_search_space=[{"name": "max_depth", "type": "int", "low": 3, "high": 10}],
            reference_round_ids=[r1["id"]],
        )
        result = service.submit_proposal(campaign["id"], narrow_proposal)
        assert result["accepted"] is True
        # submit_proposal already resolved r1 — verify
        r1_updated = service.get_round(r1["id"])
        assert r1_updated["state"] == "RESOLVED"

        # Complete round 2
        r2 = service.get_rounds(campaign["id"])[1]
        service.transition_round(r2["id"], RoundState.RUNNING)
        service.complete_round_execution(r2["id"], trial_end=50)
        service.transition_round(r2["id"], RoundState.SUMMARIZING)
        service.write_summary(r2["id"], {"schema_version": 1})
        service.transition_round(r2["id"], RoundState.AWAITING_AGENT)

        # Immediately try to widen — should be rejected (cooldown)
        widen_proposal = ActionProposal(
            action="widen_search",
            justification="Expand",
            proposed_search_space=[{"name": "max_depth", "type": "int", "low": 1, "high": 20}],
            reference_round_ids=[r2["id"]],
        )
        result = service.submit_proposal(campaign["id"], widen_proposal)
        assert result["accepted"] is False
        assert "cooldown" in result["rejection_reason"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_campaign.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CampaignService**


```python
# src/agent_hpo/core/campaign.py
"""Campaign and round management — the core service layer."""

from __future__ import annotations

import json
from typing import Any

from psycopg.rows import dict_row

from agent_hpo.core.db import Database
from agent_hpo.core.models import (
    ActionProposal,
    CampaignConfig,
    ParamSpec,
)
from agent_hpo.core.state import (
    CampaignState,
    RoundState,
    validate_campaign_transition,
    validate_round_transition,
)

COOLDOWN_ROUNDS = 2
STRUCTURAL_ACTIONS = {"narrow_search", "widen_search"}
OPPOSITE_ACTIONS = {"narrow_search": "widen_search", "widen_search": "narrow_search"}


class CampaignService:
    def __init__(self, db: Database) -> None:
        self._db = db

    # --- Campaign CRUD ---

    def create_campaign(self, name: str, config: CampaignConfig) -> dict:
        study_name = f"{name}_round_1"
        search_space_dicts = [p.to_dict() for p in config.initial_search_space]

        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO campaigns "
                "(name, metric_name, objective_direction, backend, sampler_config, "
                "initial_search_space, improvement_criteria, stop_conditions, trials_per_round) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    name,
                    config.metric_name,
                    config.objective_direction,
                    config.backend,
                    json.dumps(config.sampler_config),
                    json.dumps(search_space_dicts),
                    json.dumps(config.improvement_criteria.to_dict()),
                    json.dumps(config.stop_conditions.to_dict()),
                    config.trials_per_round,
                ),
            )
            campaign = cur.fetchone()

            # Create system-generated round 1
            conn.execute(
                "INSERT INTO study_rounds "
                "(campaign_id, round_number, optuna_study_name, search_space, budget, trial_offset) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    campaign["id"],
                    1,
                    study_name,
                    json.dumps(search_space_dicts),
                    config.trials_per_round,
                    0,
                ),
            )
        return campaign

    def get_campaign(self, campaign_id: int) -> dict:
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM campaigns WHERE id = %s", (campaign_id,)
            )
            return cur.fetchone()

    def get_rounds(self, campaign_id: int) -> list[dict]:
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM study_rounds WHERE campaign_id = %s ORDER BY round_number",
                (campaign_id,),
            )
            return cur.fetchall()

    def get_round(self, round_id: int) -> dict:
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM study_rounds WHERE id = %s", (round_id,)
            )
            return cur.fetchone()

    # --- State transitions ---

    def transition_campaign(self, campaign_id: int, to_state: CampaignState) -> None:
        campaign = self.get_campaign(campaign_id)
        from_state = CampaignState(campaign["state"])
        validate_campaign_transition(from_state, to_state)
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE campaigns SET state = %s, updated_at = now() WHERE id = %s",
                (to_state.value, campaign_id),
            )

    def transition_round(self, round_id: int, to_state: RoundState, failed_from: str | None = None) -> None:
        round_row = self.get_round(round_id)
        from_state = RoundState(round_row["state"])
        validate_round_transition(from_state, to_state)
        with self._db.connection() as conn:
            if to_state == RoundState.FAILED:
                conn.execute(
                    "UPDATE study_rounds SET state = %s, failed_from = %s, updated_at = now() WHERE id = %s",
                    (to_state.value, failed_from or from_state.value, round_id),
                )
            elif to_state == RoundState.RETRYING:
                conn.execute(
                    "UPDATE study_rounds SET state = %s, retry_count = retry_count + 1, updated_at = now() WHERE id = %s",
                    (to_state.value, round_id),
                )
            else:
                conn.execute(
                    "UPDATE study_rounds SET state = %s, updated_at = now() WHERE id = %s",
                    (to_state.value, round_id),
                )

    # --- Round execution helpers ---

    def complete_round_execution(self, round_id: int, trial_end: int) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE study_rounds SET trial_end = %s, updated_at = now() WHERE id = %s",
                (trial_end, round_id),
            )

    def write_summary(self, round_id: int, summary: dict) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE study_rounds SET summary = %s, summary_schema_version = %s, updated_at = now() WHERE id = %s",
                (json.dumps(summary), summary.get("schema_version", 1), round_id),
            )

    # --- Proposal validation and execution ---

    def submit_proposal(self, campaign_id: int, proposal: ActionProposal) -> dict:
        # Gate: campaign must be RUNNING and latest round must be AWAITING_AGENT
        campaign = self.get_campaign(campaign_id)
        if campaign["state"] != "RUNNING":
            return self._record_decision(
                campaign_id, proposal, accepted=False,
                reason=f"Campaign is {campaign['state']}, proposals only accepted when RUNNING",
            )
        rounds = self.get_rounds(campaign_id)
        if not rounds or rounds[-1]["state"] != "AWAITING_AGENT":
            current_state = rounds[-1]["state"] if rounds else "no rounds"
            return self._record_decision(
                campaign_id, proposal, accepted=False,
                reason=f"Latest round is {current_state}, proposals only accepted when AWAITING_AGENT",
            )

        try:
            proposal.validate()
        except ValueError as e:
            return self._record_decision(campaign_id, proposal, accepted=False, reason=str(e))

        # Validate search space against backend param definitions
        if proposal.action in ("narrow_search", "widen_search") and proposal.proposed_search_space:
            campaign = self.get_campaign(campaign_id)
            from agent_hpo.backends import get_backend
            backend_cls = get_backend(campaign["backend"])
            backend = backend_cls()
            valid_names = {p.name for p in backend.param_definitions()}
            proposed_names = {p["name"] for p in proposal.proposed_search_space}
            invalid = proposed_names - valid_names
            if invalid:
                return self._record_decision(
                    campaign_id, proposal, accepted=False,
                    reason=f"Unknown params in proposed_search_space: {invalid}",
                )

        # Cooldown check
        if proposal.action in STRUCTURAL_ACTIONS:
            rejection = self._check_cooldown(campaign_id, proposal.action)
            if rejection:
                return self._record_decision(campaign_id, proposal, accepted=False, reason=rejection)

        # Get the current round (the one the agent is responding to)
        rounds = self.get_rounds(campaign_id)
        current_round = rounds[-1]

        if proposal.action == "stop":
            decision = self._record_decision(campaign_id, proposal, accepted=True)
            self.transition_round(current_round["id"], RoundState.RESOLVED)
            self.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return decision

        # Determine new round params
        if proposal.action in ("continue", "increase_budget"):
            new_study_name = current_round["optuna_study_name"]
            new_search_space = current_round["search_space"]
            trial_offset = current_round.get("trial_end", 0) or 0
            parent_id = None
            budget = proposal.proposed_budget if proposal.action == "increase_budget" else current_round["budget"]
        else:
            # narrow or widen — new Optuna study
            campaign = self.get_campaign(campaign_id)
            new_round_number = len(rounds) + 1
            new_study_name = f"{campaign['name']}_round_{new_round_number}"
            new_search_space = json.dumps(proposal.proposed_search_space)
            trial_offset = 0
            parent_id = current_round["id"]
            budget = current_round["budget"]

        # Mark current round as resolved
        self.transition_round(current_round["id"], RoundState.RESOLVED)

        # Create new round
        new_round_number = len(rounds) + 1
        with self._db.connection() as conn:
            search_space_val = new_search_space if isinstance(new_search_space, str) else json.dumps(new_search_space)
            conn.execute(
                "INSERT INTO study_rounds "
                "(campaign_id, round_number, optuna_study_name, search_space, budget, trial_offset, parent_round_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (campaign_id, new_round_number, new_study_name, search_space_val, budget, trial_offset, parent_id),
            )

        return self._record_decision(campaign_id, proposal, accepted=True)

    def _check_cooldown(self, campaign_id: int, action: str) -> str | None:
        """Check if a structural action violates the cooldown rule."""
        opposite = OPPOSITE_ACTIONS.get(action)
        if not opposite:
            return None

        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT action, round_id FROM agent_decisions "
                "WHERE campaign_id = %s AND accepted = true AND action IN %s "
                "ORDER BY created_at DESC LIMIT 1",
                (campaign_id, tuple(STRUCTURAL_ACTIONS)),
            )
            last_structural = cur.fetchone()

        if not last_structural:
            return None

        if last_structural["action"] != opposite:
            return None

        # Count rounds since the round CREATED BY the structural action (one after the decision round)
        rounds = self.get_rounds(campaign_id)
        decision_round_idx = None
        for i, r in enumerate(rounds):
            if r["id"] == last_structural["round_id"]:
                decision_round_idx = i
                break

        if decision_round_idx is None:
            return None

        # The structural action created the round AFTER the decision round
        created_round_idx = decision_round_idx + 1
        rounds_since = len(rounds) - 1 - created_round_idx
        if rounds_since < COOLDOWN_ROUNDS:
            return (
                f"Cooldown violation: {opposite} was applied {rounds_since} round(s) ago, "
                f"must wait {COOLDOWN_ROUNDS} rounds before reversing with {action}"
            )
        return None

    def _record_decision(
        self, campaign_id: int, proposal: ActionProposal, accepted: bool, reason: str | None = None
    ) -> dict:
        rounds = self.get_rounds(campaign_id)
        current_round_id = rounds[-1]["id"]
        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO agent_decisions "
                "(campaign_id, round_id, action, justification, proposed_search_space, "
                "proposed_budget, reference_round_ids, accepted, rejection_reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (
                    campaign_id,
                    current_round_id,
                    proposal.action,
                    proposal.justification,
                    json.dumps(proposal.proposed_search_space) if proposal.proposed_search_space else None,
                    proposal.proposed_budget,
                    json.dumps(proposal.reference_round_ids),
                    accepted,
                    reason,
                ),
            )
            return cur.fetchone()

    # --- Query helpers for MCP/CLI ---

    def get_campaign_history(self, campaign_id: int) -> dict:
        rounds = self.get_rounds(campaign_id)
        with self._db.connection() as conn:
            cur = conn.execute(
                "SELECT * FROM agent_decisions WHERE campaign_id = %s ORDER BY created_at",
                (campaign_id,),
            )
            decisions = cur.fetchall()
        return {"rounds": rounds, "decisions": decisions}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_campaign.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/core/campaign.py tests/test_campaign.py
git commit -m "feat: campaign service with CRUD, round management, and proposal validation"
```

---

### Task 7: Backend Interface + XGBoost

**Files:**
- Create: `src/agent_hpo/backends/base.py`
- Modify: `src/agent_hpo/backends/__init__.py`
- Create: `src/agent_hpo/backends/xgboost.py`
- Create: `tests/test_backend_xgboost.py`

- [ ] **Step 1: Write tests for XGBoost backend**

```python
# tests/test_backend_xgboost.py
import pytest
import numpy as np
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from agent_hpo.backends.base import suggest_from_param_spec
from agent_hpo.backends.xgboost import XGBoostBackend
from agent_hpo.core.models import ParamSpec, DatasetSplit


@pytest.fixture
def dataset():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)


class TestSuggestFromParamSpec:
    def test_float_param(self):
        spec = ParamSpec(name="lr", type="float", low=0.01, high=1.0, log=True)
        study = optuna.create_study()
        trial = study.ask()
        val = suggest_from_param_spec(trial, spec)
        assert 0.01 <= val <= 1.0

    def test_int_param(self):
        spec = ParamSpec(name="depth", type="int", low=1, high=15)
        study = optuna.create_study()
        trial = study.ask()
        val = suggest_from_param_spec(trial, spec)
        assert 1 <= val <= 15
        assert isinstance(val, int)

    def test_categorical_param(self):
        spec = ParamSpec(name="booster", type="categorical", choices=["gbtree", "dart"])
        study = optuna.create_study()
        trial = study.ask()
        val = suggest_from_param_spec(trial, spec)
        assert val in ["gbtree", "dart"]


class TestXGBoostBackend:
    def test_default_search_space(self):
        backend = XGBoostBackend()
        space = backend.default_search_space()
        assert len(space) > 0
        names = {p.name for p in space}
        assert "max_depth" in names
        assert "learning_rate" in names

    def test_create_objective_returns_callable(self, dataset):
        backend = XGBoostBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        assert callable(objective)

    def test_objective_runs_and_returns_float(self, dataset):
        backend = XGBoostBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2, show_progress_bar=False)
        assert len(study.trials) == 2
        assert all(t.value is not None for t in study.trials)

    def test_objective_logs_train_metric(self, dataset):
        backend = XGBoostBackend()
        space = backend.default_search_space()
        objective = backend.create_objective(dataset, "accuracy", space)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        assert "train_accuracy" in study.trials[0].user_attrs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backend_xgboost.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base.py**

```python
# src/agent_hpo/backends/base.py
"""Backend protocol and shared helpers."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import optuna

from agent_hpo.core.models import DatasetSplit, ParamSpec


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

    def param_definitions(self) -> list[ParamSpec]: ...
```

- [ ] **Step 4: Implement xgboost.py**

```python
# src/agent_hpo/backends/xgboost.py
"""XGBoost backend for agent-hpo."""

from __future__ import annotations

from typing import Callable

import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from agent_hpo.backends.base import suggest_from_param_spec
from agent_hpo.core.models import DatasetSplit, ParamSpec

METRICS = {
    "accuracy": (accuracy_score, False),       # (fn, needs_proba)
    "rmse": (mean_squared_error, False),
    "log_loss": (log_loss, True),
}


class XGBoostBackend:
    def create_objective(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        search_space: list[ParamSpec],
    ) -> Callable[[optuna.Trial], float]:
        metric_fn, needs_proba = METRICS[metric_name]

        def objective(trial: optuna.Trial) -> float:
            params = {spec.name: suggest_from_param_spec(trial, spec) for spec in search_space}
            params["verbosity"] = 0
            params["nthread"] = 1

            model = xgb.XGBClassifier(**params) if metric_name != "rmse" else xgb.XGBRegressor(**params)
            model.fit(dataset.X_train, dataset.y_train, verbose=False)

            # Validation score
            if needs_proba:
                y_pred_val = model.predict_proba(dataset.X_val)
                y_pred_train = model.predict_proba(dataset.X_train)
            else:
                y_pred_val = model.predict(dataset.X_val)
                y_pred_train = model.predict(dataset.X_train)

            val_score = metric_fn(dataset.y_val, y_pred_val)
            train_score = metric_fn(dataset.y_train, y_pred_train)

            if metric_name == "rmse":
                val_score = val_score ** 0.5
                train_score = train_score ** 0.5

            # Log train metric for generalization gap (same scale as val_score)
            trial.set_user_attr(f"train_{metric_name}", float(train_score))

            return float(val_score)

        return objective

    def default_search_space(self) -> list[ParamSpec]:
        return self._param_defs()

    def param_definitions(self) -> list[ParamSpec]:
        return self._param_defs()

    def _param_defs(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="max_depth", type="int", low=1, high=15),
            ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
            ParamSpec(name="n_estimators", type="int", low=50, high=500),
            ParamSpec(name="min_child_weight", type="float", low=1.0, high=10.0),
            ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ParamSpec(name="colsample_bytree", type="float", low=0.5, high=1.0),
            ParamSpec(name="gamma", type="float", low=0.0, high=5.0),
            ParamSpec(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True),
        ]
```

- [ ] **Step 5: Register the backend**

Update `src/agent_hpo/backends/__init__.py` to add:
```python
from agent_hpo.backends.xgboost import XGBoostBackend
register_backend("xgboost", XGBoostBackend)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_backend_xgboost.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/agent_hpo/backends/ tests/test_backend_xgboost.py
git commit -m "feat: XGBoost backend with search space support and train metric logging"
```

---

### Task 8: Summarizer

**Files:**
- Create: `src/agent_hpo/summarizer.py`
- Create: `tests/test_summarizer.py`

- [ ] **Step 1: Write tests for summarizer**

```python
# tests/test_summarizer.py
import pytest
import optuna
from agent_hpo.summarizer import RoundSummarizer
from agent_hpo.core.models import RoundSummary


def _make_study_with_trials(n_trials=10, direction="maximize"):
    """Create an Optuna study with completed trials for testing."""
    study = optuna.create_study(direction=direction)

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 10.0)
        y = trial.suggest_int("y", 1, 5)
        trial.set_user_attr("train_accuracy", x * y * 0.02)
        return x * y * 0.01

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


class TestRoundSummarizer:
    def test_basic_summary(self):
        study = _make_study_with_trials(10)
        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=1,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=0,
            trial_end=10,
            prev_best_score=None,
            parent_round_id=None,
            optuna_study_name="test_study",
            action_that_created="init",
            cumulative_wall_time=10.0,
        )
        assert isinstance(summary, RoundSummary)
        assert summary.trials_added == 10
        assert summary.best_score is not None
        assert summary.completed_trials > 0
        assert summary.schema_version == 1

    def test_summary_with_trial_boundaries(self):
        study = _make_study_with_trials(20)
        summarizer = RoundSummarizer()
        # Summarize only trials [10, 20)
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=2,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=10,
            trial_end=20,
            prev_best_score=0.1,
            parent_round_id=1,
            optuna_study_name="test_study",
            action_that_created="continue",
            cumulative_wall_time=20.0,
        )
        assert summary.trials_added == 10
        assert summary.total_trials == 20
        assert summary.delta_from_prev is not None

    def test_summary_with_no_completed_trials(self):
        study = optuna.create_study(direction="maximize")
        # Add only pruned trials
        for i in range(5):
            trial = study.ask()
            trial.suggest_float("x", 0.0, 10.0)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)

        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=1,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=0,
            trial_end=5,
            prev_best_score=None,
            parent_round_id=None,
            optuna_study_name="test",
            action_that_created="init",
            cumulative_wall_time=5.0,
        )
        assert summary.round_completed_trials == 0
        assert summary.round_best_score is None
        assert summary.new_best_in_round is False
        assert summary.pruned_rate == 1.0

    def test_convergence_curve_is_round_local(self):
        study = _make_study_with_trials(20)
        summarizer = RoundSummarizer()
        summary = summarizer.summarize(
            study=study,
            campaign_id=1,
            round_id=2,
            metric_name="accuracy",
            objective_direction="maximize",
            trial_offset=10,
            trial_end=20,
            prev_best_score=0.0,
            parent_round_id=None,
            optuna_study_name="test",
            action_that_created="continue",
            cumulative_wall_time=20.0,
        )
        # Convergence curve indices should be round-local (0-based)
        if summary.convergence_curve:
            assert summary.convergence_curve[0][0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_summarizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement summarizer**

```python
# src/agent_hpo/summarizer.py
"""Produces immutable, schema-versioned round summaries from Optuna study data."""

from __future__ import annotations

import time
from typing import Any

import optuna
from optuna.trial import TrialState

from agent_hpo.core.models import RoundSummary


class RoundSummarizer:
    def summarize(
        self,
        study: optuna.Study,
        campaign_id: int,
        round_id: int,
        metric_name: str,
        objective_direction: str,
        trial_offset: int,
        trial_end: int,
        prev_best_score: float | None,
        parent_round_id: int | None,
        optuna_study_name: str,
        action_that_created: str,
        cumulative_wall_time: float,
    ) -> RoundSummary:
        all_trials = study.trials
        round_trials = all_trials[trial_offset:trial_end]
        cumulative_trials = all_trials[:trial_end]

        # Classify round trials
        round_complete = [t for t in round_trials if t.state == TrialState.COMPLETE]
        round_failed = [t for t in round_trials if t.state == TrialState.FAIL]
        round_pruned = [t for t in round_trials if t.state == TrialState.PRUNED]
        cum_complete = [t for t in cumulative_trials if t.state == TrialState.COMPLETE]

        trials_added = len(round_trials)
        round_completed = len(round_complete)
        total_trials = len(cumulative_trials)
        completed_trials = len(cum_complete)

        # Rates
        failure_rate = len(round_failed) / trials_added if trials_added > 0 else 0.0
        pruned_rate = len(round_pruned) / trials_added if trials_added > 0 else 0.0

        # Best scores
        is_maximize = objective_direction == "maximize"

        def _best(trials):
            if not trials:
                return None, None
            if is_maximize:
                best = max(trials, key=lambda t: t.value)
            else:
                best = min(trials, key=lambda t: t.value)
            return best.value, best.params

        cum_best_score, cum_best_params = _best(cum_complete)
        round_best_score, _ = _best(round_complete)

        # Delta from prev
        delta = None
        if cum_best_score is not None and prev_best_score is not None:
            delta = cum_best_score - prev_best_score

        # New best in round
        new_best = False
        if round_best_score is not None and prev_best_score is not None:
            if is_maximize:
                new_best = round_best_score > prev_best_score
            else:
                new_best = round_best_score < prev_best_score
        elif round_best_score is not None and prev_best_score is None:
            new_best = True

        # Convergence curve (round-local)
        convergence = []
        running_best = None
        for i, t in enumerate(round_trials):
            if t.state != TrialState.COMPLETE:
                continue
            if running_best is None:
                running_best = t.value
            elif is_maximize:
                running_best = max(running_best, t.value)
            else:
                running_best = min(running_best, t.value)
            convergence.append((i, running_best))

        # Plateau signal
        plateau = False
        if round_completed > 0 and len(convergence) >= 3:
            cutoff = int(len(convergence) * 0.7)
            late_values = [v for _, v in convergence[cutoff:]]
            if late_values and late_values[0] == late_values[-1]:
                plateau = True

        # Parameter importance (best effort)
        param_importance = {}
        try:
            if completed_trials >= 4:
                importance = optuna.importance.get_param_importances(study)
                param_importance = dict(importance)
        except Exception:
            pass

        # Parameter ranges used in this round
        param_ranges = {}
        for t in round_complete:
            for name, val in t.params.items():
                if name not in param_ranges:
                    param_ranges[name] = (val, val)
                else:
                    lo, hi = param_ranges[name]
                    param_ranges[name] = (min(lo, val), max(hi, val))

        # Generalization gap
        gen_gap = None
        if round_complete:
            train_key = f"train_{metric_name}"
            gaps = []
            for t in round_complete:
                if train_key in t.user_attrs:
                    gaps.append(abs(t.value - t.user_attrs[train_key]))
            if gaps:
                gen_gap = sum(gaps) / len(gaps)

        # Wall time for this round
        round_wall = 0.0
        if round_trials:
            start_times = [t.datetime_start for t in round_trials if t.datetime_start]
            end_times = [t.datetime_complete for t in round_trials if t.datetime_complete]
            if start_times and end_times:
                round_wall = (max(end_times) - min(start_times)).total_seconds()

        return RoundSummary(
            schema_version=RoundSummary.CURRENT_SCHEMA_VERSION,
            round_id=round_id,
            campaign_id=campaign_id,
            metric_name=metric_name,
            objective_direction=objective_direction,
            best_score=cum_best_score,
            best_params=cum_best_params,
            delta_from_prev=delta,
            total_trials=total_trials,
            completed_trials=completed_trials,
            trials_added=trials_added,
            round_completed_trials=round_completed,
            new_best_in_round=new_best,
            round_best_score=round_best_score,
            convergence_curve=convergence,
            plateau_signal=plateau,
            param_importance=param_importance,
            param_ranges_used=param_ranges,
            generalization_gap=gen_gap,
            failure_rate=failure_rate,
            pruned_rate=pruned_rate,
            round_wall_time_seconds=round_wall,
            total_wall_time_seconds=cumulative_wall_time + round_wall,
            parent_round_id=parent_round_id,
            optuna_study_name=optuna_study_name,
            action_that_created_this_round=action_that_created,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_summarizer.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/summarizer.py tests/test_summarizer.py
git commit -m "feat: round summarizer with trial boundaries and nullable fields"
```

---

### Task 9: Scheduler

**Files:**
- Create: `src/agent_hpo/scheduler.py`
- Create: `tests/test_scheduler.py`

The scheduler orchestrates one round: budget clipping, Optuna execution, summarization, stop condition checks. Returns control at `AWAITING_AGENT`, `COMPLETED`, or `FAILED`.

- [ ] **Step 1: Write tests for scheduler**

```python
# tests/test_scheduler.py
import pytest
import optuna

from agent_hpo.scheduler import Scheduler
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, RoundSummary,
)


class TestBudgetClipping:
    def test_clips_to_remaining_trials(self):
        sc = StopConditions(max_total_trials=100, patience_rounds=3)
        effective = Scheduler.clip_budget(budget=50, cumulative_trials=80, stop_conditions=sc)
        assert effective == 20

    def test_returns_zero_when_budget_exhausted(self):
        sc = StopConditions(max_total_trials=100, patience_rounds=3)
        effective = Scheduler.clip_budget(budget=50, cumulative_trials=100, stop_conditions=sc)
        assert effective == 0

    def test_no_clip_when_no_cap(self):
        sc = StopConditions(patience_rounds=3)
        effective = Scheduler.clip_budget(budget=50, cumulative_trials=500, stop_conditions=sc)
        assert effective == 50


class TestStopConditionChecks:
    def test_target_score_fires_maximize(self):
        sc = StopConditions(target_score=0.95, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.96, direction="maximize",
                                         total_trials=10, wall_time=10.0) == "target_score"

    def test_target_score_fires_minimize(self):
        sc = StopConditions(target_score=0.05, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.04, direction="minimize",
                                         total_trials=10, wall_time=10.0) == "target_score"

    def test_target_score_not_met(self):
        sc = StopConditions(target_score=0.95, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.90, direction="maximize",
                                         total_trials=10, wall_time=10.0) is None

    def test_max_trials_fires(self):
        sc = StopConditions(max_total_trials=100, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.9, direction="maximize",
                                         total_trials=100, wall_time=10.0) == "max_total_trials"

    def test_max_wall_time_fires(self):
        sc = StopConditions(max_wall_time_seconds=3600.0, patience_rounds=3)
        assert Scheduler.check_hard_stop(sc, best_score=0.9, direction="maximize",
                                         total_trials=10, wall_time=3601.0) == "max_wall_time"

    def test_max_rounds_fires_after_completion(self):
        sc = StopConditions(max_rounds=5, patience_rounds=3)
        assert Scheduler.check_rounds_stop(sc, completed_rounds=5) == "max_rounds"

    def test_max_rounds_not_met(self):
        sc = StopConditions(max_rounds=5, patience_rounds=3)
        assert Scheduler.check_rounds_stop(sc, completed_rounds=4) is None

    def test_max_rounds_allows_last_round_to_execute(self):
        sc = StopConditions(max_rounds=5, patience_rounds=3)
        # Round 5 is executing, 4 completed — should not stop yet
        assert Scheduler.check_rounds_stop(sc, completed_rounds=4) is None
        # After round 5 completes — should stop
        assert Scheduler.check_rounds_stop(sc, completed_rounds=5) == "max_rounds"


class TestPatienceCheck:
    def test_patience_fires(self):
        ic = ImprovementCriteria(mode="strict_better")
        # 3 rounds, none improved
        summaries = [
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.90, round_completed_trials=10),
        ]
        assert Scheduler.check_patience(summaries, ic, "maximize", patience=3) is True

    def test_patience_not_fired(self):
        ic = ImprovementCriteria(mode="strict_better")
        summaries = [
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.91, round_completed_trials=10),
            RoundSummary(best_score=0.91, round_completed_trials=10),
        ]
        assert Scheduler.check_patience(summaries, ic, "maximize", patience=3) is False

    def test_patience_counts_zero_completed_as_no_improvement(self):
        ic = ImprovementCriteria(mode="strict_better")
        summaries = [
            RoundSummary(best_score=0.90, round_completed_trials=10),
            RoundSummary(best_score=0.90, round_completed_trials=0),
            RoundSummary(best_score=0.90, round_completed_trials=0),
        ]
        assert Scheduler.check_patience(summaries, ic, "maximize", patience=2) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scheduler**


```python
# src/agent_hpo/scheduler.py
"""Round orchestration: budget clipping, stop conditions, execution control."""

from __future__ import annotations

from agent_hpo.core.models import (
    ImprovementCriteria,
    RoundSummary,
    StopConditions,
)


class Scheduler:
    @staticmethod
    def clip_budget(
        budget: int,
        cumulative_trials: int,
        stop_conditions: StopConditions,
    ) -> int:
        if stop_conditions.max_total_trials is not None:
            remaining = stop_conditions.max_total_trials - cumulative_trials
            return max(0, min(budget, remaining))
        return budget

    @staticmethod
    def check_hard_stop(
        sc: StopConditions,
        best_score: float | None,
        direction: str,
        total_trials: int,
        wall_time: float,
    ) -> str | None:
        if sc.max_total_trials is not None and total_trials >= sc.max_total_trials:
            return "max_total_trials"
        if sc.max_wall_time_seconds is not None and wall_time >= sc.max_wall_time_seconds:
            return "max_wall_time"
        if sc.target_score is not None and best_score is not None:
            if direction == "maximize" and best_score >= sc.target_score:
                return "target_score"
            if direction == "minimize" and best_score <= sc.target_score:
                return "target_score"
        return None

    @staticmethod
    def check_rounds_stop(sc: StopConditions, completed_rounds: int) -> str | None:
        """Check if max_rounds has been reached. completed_rounds is the number of
        rounds that have finished executing (not the current round number).
        Called AFTER a round completes, so the just-finished round is included."""
        if sc.max_rounds is not None and completed_rounds >= sc.max_rounds:
            return "max_rounds"
        return None

    @staticmethod
    def check_patience(
        summaries: list[RoundSummary],
        improvement_criteria: ImprovementCriteria,
        direction: str,
        patience: int,
    ) -> bool:
        if len(summaries) < patience:
            return False

        recent = summaries[-patience:]
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]

            if curr.round_completed_trials == 0:
                continue  # counts as no improvement

            if prev.best_score is None or curr.best_score is None:
                continue

            if improvement_criteria.is_improvement(curr.best_score, prev.best_score, direction):
                return False

        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/scheduler.py tests/test_scheduler.py
git commit -m "feat: scheduler with budget clipping, stop conditions, and patience check"
```

---

### Task 10: CLI

**Files:**
- Create: `src/agent_hpo/cli.py`
- Create: `tests/test_cli.py`

Thin Click wrapper over the core service layer. Each command reads config, calls core, prints output.

- [ ] **Step 1: Write tests for CLI commands**

```python
# tests/test_cli.py
import json
import pytest
from click.testing import CliRunner
from agent_hpo.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCliInit:
    def test_init_creates_campaign(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENT_HPO_DB_URL", test_db_url)
        result = runner.invoke(cli, [
            "init", "my-campaign",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
            "--trials-per-round", "20",
            "--max-rounds", "5",
            "--patience", "3",
        ])
        assert result.exit_code == 0, result.output
        assert "my-campaign" in result.output
        assert "CREATED" in result.output

    def test_init_duplicate_fails(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENT_HPO_DB_URL", test_db_url)
        runner.invoke(cli, [
            "init", "dup-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
        ])
        result = runner.invoke(cli, [
            "init", "dup-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
        ])
        assert result.exit_code != 0


class TestCliStatus:
    def test_status_shows_campaign(self, runner, test_db_url, monkeypatch):
        monkeypatch.setenv("AGENT_HPO_DB_URL", test_db_url)
        runner.invoke(cli, [
            "init", "status-test",
            "--backend", "xgboost",
            "--metric", "accuracy",
            "--direction", "maximize",
        ])
        result = runner.invoke(cli, ["status", "status-test"])
        assert result.exit_code == 0
        assert "CREATED" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CLI**


```python
# src/agent_hpo/cli.py
"""CLI: thin Click wrapper over the core service layer."""

from __future__ import annotations

import json
import os
import sys

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
    """Manually stop a campaign.

    If no round is currently executing, transitions immediately to STOPPED.
    If a round is RUNNING, rejects — use pause instead, or wait for the round to finish.
    """
    db = _get_db()
    service = CampaignService(db)
    try:
        with db.connection() as conn:
            cur = conn.execute("SELECT * FROM campaigns WHERE name = %s", (name,))
            campaign = cur.fetchone()
        if not campaign:
            click.echo(f"Campaign '{name}' not found", err=True)
            sys.exit(1)

        # Check if any round is currently executing
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


@cli.command()
@click.argument("name")
def export(name):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/cli.py tests/test_cli.py
git commit -m "feat: CLI with init, status, pause, resume, stop, history, export commands"
```

---

### Task 10b: CLI `run` and `baseline` Commands (Core Orchestration)

**Files:**
- Modify: `src/agent_hpo/cli.py`
- Create: `src/agent_hpo/runner.py` (round orchestration logic, reusable by CLI and tests)
- Create: `tests/test_runner.py`

This is the core orchestration — wires together scheduler, backend, summarizer, and Optuna with persistent RDBStorage.

- [ ] **Step 1: Write tests for runner**

```python
# tests/test_runner.py
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from agent_hpo.runner import RoundRunner, RunResult
from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec, DatasetSplit,
)
from agent_hpo.core.state import CampaignState, RoundState


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


@pytest.fixture
def campaign(db):
    service = CampaignService(db)
    from agent_hpo.backends.xgboost import XGBoostBackend
    backend = XGBoostBackend()
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=backend.default_search_space(),
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=3, patience_rounds=2, max_total_trials=15),
        trials_per_round=5,
    )
    return service.create_campaign("runner-test", config)


class TestRoundRunner:
    def test_run_first_round(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        result = runner.run_next_round(campaign["id"])
        assert result.status in ("AWAITING_AGENT", "COMPLETED", "FAILED")
        assert result.round_number == 1

        # Verify round was summarized
        service = CampaignService(db)
        r1 = service.get_rounds(campaign["id"])[0]
        assert r1["summary"] is not None
        assert r1["trial_end"] is not None
        assert r1["trial_end"] > 0

    def test_run_respects_budget_clipping(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        # max_total_trials=15, trials_per_round=5 → 3 rounds max
        result1 = runner.run_next_round(campaign["id"])
        assert result1.status == "AWAITING_AGENT"

    def test_run_uses_persistent_optuna_storage(self, db, dataset, campaign):
        runner = RoundRunner(db, dataset)
        result = runner.run_next_round(campaign["id"])

        # Verify Optuna study exists in storage and can be loaded
        import optuna
        storage = optuna.storages.RDBStorage(db.optuna_storage_url)
        service = CampaignService(db)
        r1 = service.get_rounds(campaign["id"])[0]
        study = optuna.load_study(study_name=r1["optuna_study_name"], storage=storage)
        assert len(study.trials) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement runner.py**


```python
# src/agent_hpo/runner.py
"""Round orchestration: executes one study round end-to-end."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import optuna

from agent_hpo.backends import get_backend
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.db import Database
from agent_hpo.core.locking import LeaseManager
from agent_hpo.core.models import (
    ImprovementCriteria,
    ParamSpec,
    RoundSummary,
    StopConditions,
    DatasetSplit,
)
from agent_hpo.core.state import CampaignState, RoundState
from agent_hpo.scheduler import Scheduler
from agent_hpo.summarizer import RoundSummarizer


@dataclass
class RunResult:
    status: str  # AWAITING_AGENT, COMPLETED, FAILED
    round_number: int
    stop_reason: str | None = None


class RoundRunner:
    def __init__(self, db: Database, dataset: DatasetSplit) -> None:
        self._db = db
        self._dataset = dataset
        self._service = CampaignService(db)
        self._summarizer = RoundSummarizer()

    def _cumulative_wall_time(self, rounds: list[dict]) -> float:
        total = 0.0
        for r in rounds:
            s = r.get("summary")
            if s:
                if isinstance(s, str):
                    s = json.loads(s)
                total = s.get("total_wall_time_seconds", total)
        return total

    def run_next_round(self, campaign_id: int) -> RunResult:
        lease = LeaseManager(self._db)
        lease.acquire(campaign_id)
        try:
            return self._execute(campaign_id, lease)
        except Exception:
            lease.release(campaign_id)
            raise
        finally:
            try:
                lease.release(campaign_id)
            except Exception:
                pass

    def _execute(self, campaign_id: int, lease: LeaseManager) -> RunResult:
        campaign = self._service.get_campaign(campaign_id)
        stop_cond = StopConditions.from_dict(campaign["stop_conditions"])
        improvement = ImprovementCriteria.from_dict(campaign["improvement_criteria"])

        # Transition campaign to RUNNING if CREATED
        if campaign["state"] == "CREATED":
            self._service.transition_campaign(campaign_id, CampaignState.RUNNING)

        rounds = self._service.get_rounds(campaign_id)
        current_round = rounds[-1]

        if current_round["state"] != "PROPOSED":
            raise RuntimeError(f"Expected PROPOSED round, got {current_round['state']}")

        # Check if campaign was stopped while we were waiting
        if campaign["state"] in ("STOPPED", "COMPLETED", "FAILED"):
            raise RuntimeError(f"Campaign is {campaign['state']}, cannot run")

        round_number = current_round["round_number"]

        # Pre-round budget clipping (trial cap)
        cumulative_trials = sum(r.get("trial_end", 0) or 0 for r in rounds[:-1])
        effective_budget = Scheduler.clip_budget(
            current_round["budget"], cumulative_trials, stop_cond
        )
        if effective_budget <= 0:
            self._service.transition_round(current_round["id"], RoundState.RUNNING)
            self._service.transition_round(current_round["id"], RoundState.SUMMARIZING)
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, "max_total_trials")

        # Pre-round wall time check
        cumulative_wall = self._cumulative_wall_time(rounds[:-1])
        if stop_cond.max_wall_time_seconds and cumulative_wall >= stop_cond.max_wall_time_seconds:
            self._service.transition_round(current_round["id"], RoundState.RUNNING)
            self._service.transition_round(current_round["id"], RoundState.SUMMARIZING)
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, "max_wall_time")

        # Compute Optuna timeout from remaining wall time
        optuna_timeout = None
        if stop_cond.max_wall_time_seconds:
            optuna_timeout = max(1.0, stop_cond.max_wall_time_seconds - cumulative_wall)

        # Setup Optuna study with persistent storage
        storage = optuna.storages.RDBStorage(self._db.optuna_storage_url)
        sampler_config = campaign["sampler_config"]
        if isinstance(sampler_config, str):
            sampler_config = json.loads(sampler_config)
        sampler = optuna.samplers.TPESampler(seed=sampler_config.get("seed", 42))

        study_name = current_round["optuna_study_name"]
        try:
            study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=campaign["objective_direction"],
                sampler=sampler,
            )

        # Create backend and objective
        backend_cls = get_backend(campaign["backend"])
        backend = backend_cls()
        search_space_raw = current_round["search_space"]
        if isinstance(search_space_raw, str):
            search_space_raw = json.loads(search_space_raw)
        search_space = [ParamSpec.from_dict(s) for s in search_space_raw]
        objective = backend.create_objective(self._dataset, campaign["metric_name"], search_space)

        # Run
        self._service.transition_round(current_round["id"], RoundState.RUNNING)
        lease.refresh(campaign_id)

        trial_offset = current_round["trial_offset"]
        study.optimize(objective, n_trials=effective_budget, timeout=optuna_timeout, show_progress_bar=False)
        trial_end = len(study.trials)

        self._service.complete_round_execution(current_round["id"], trial_end=trial_end)

        # Summarize first, then check stops using accurate cumulative wall time
        self._service.transition_round(current_round["id"], RoundState.SUMMARIZING)
        prev_best = None
        for r in reversed(rounds[:-1]):
            if r.get("summary"):
                s = r["summary"]
                if isinstance(s, str):
                    s = json.loads(s)
                if s.get("best_score") is not None:
                    prev_best = s["best_score"]
                    break

        summary = self._summarizer.summarize(
            study=study,
            campaign_id=campaign_id,
            round_id=current_round["id"],
            metric_name=campaign["metric_name"],
            objective_direction=campaign["objective_direction"],
            trial_offset=trial_offset,
            trial_end=trial_end,
            prev_best_score=prev_best,
            parent_round_id=current_round.get("parent_round_id"),
            optuna_study_name=study_name,
            action_that_created=current_round.get("action_that_created", "init") if round_number == 1 else "agent",
            cumulative_wall_time=cumulative_wall,
        )
        self._service.write_summary(current_round["id"], summary.to_dict())

        # Post-round hard stop check (uses summary's cumulative wall time)
        hard_stop = Scheduler.check_hard_stop(
            stop_cond, summary.best_score, campaign["objective_direction"],
            summary.total_trials, summary.total_wall_time_seconds,
        )
        # Also check max_rounds (now that this round has completed)
        completed_rounds = round_number  # this round just finished
        if not hard_stop:
            rounds_stop = Scheduler.check_rounds_stop(stop_cond, completed_rounds)
            if rounds_stop:
                hard_stop = rounds_stop

        if hard_stop:
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, hard_stop)

        # Patience check
        all_summaries = []
        for r in rounds:
            s = r.get("summary")
            if s:
                if isinstance(s, str):
                    s = json.loads(s)
                all_summaries.append(RoundSummary.from_dict(s))
        all_summaries.append(summary)

        if Scheduler.check_patience(all_summaries, improvement, campaign["objective_direction"], stop_cond.patience_rounds):
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_round(current_round["id"], RoundState.CLOSED)
            self._service.transition_campaign(campaign_id, CampaignState.COMPLETED)
            return RunResult("COMPLETED", round_number, "patience")

        # Check pause requested
        campaign = self._service.get_campaign(campaign_id)
        if campaign.get("pause_requested") or campaign["state"] == "PAUSE_REQUESTED":
            self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
            self._service.transition_campaign(campaign_id, CampaignState.PAUSED)
            with self._db.connection() as conn:
                conn.execute(
                    "UPDATE campaigns SET pause_requested = false WHERE id = %s", (campaign_id,)
                )
            return RunResult("AWAITING_AGENT", round_number)

        self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
        return RunResult("AWAITING_AGENT", round_number)
```

- [ ] **Step 4: Add `run` and `baseline` commands to cli.py**

Add to `src/agent_hpo/cli.py`:

```python
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
```

Also add `import time` at the top of `cli.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/agent_hpo/runner.py src/agent_hpo/cli.py tests/test_runner.py
git commit -m "feat: round runner with full orchestration, run and baseline CLI commands"
```

---

### Task 11: MCP Server

**Files:**
- Create: `src/agent_hpo/mcp_server.py`
- Create: `tests/test_mcp_server.py`

5 MCP tools over the core service layer.

- [ ] **Step 1: Write tests for MCP tools**

```python
# tests/test_mcp_server.py
import json
import pytest
from agent_hpo.mcp_server import (
    handle_list_campaigns,
    handle_get_campaign_status,
    handle_get_round_summary,
    handle_get_campaign_history,
    handle_submit_action_proposal,
)
from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec,
)


@pytest.fixture
def db(test_db_url):
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


@pytest.fixture
def service(db):
    return CampaignService(db)


@pytest.fixture
def campaign_with_round(service):
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=[ParamSpec(name="max_depth", type="int", low=1, high=15)],
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(patience_rounds=3),
        trials_per_round=50,
    )
    return service.create_campaign("mcp-test", config)


class TestMcpHandlers:
    def test_list_campaigns(self, db, campaign_with_round):
        result = handle_list_campaigns(db)
        assert len(result) >= 1
        assert any(c["name"] == "mcp-test" for c in result)

    def test_get_campaign_status(self, db, campaign_with_round):
        result = handle_get_campaign_status(db, "mcp-test")
        assert result["name"] == "mcp-test"
        assert result["state"] == "CREATED"

    def test_get_campaign_status_not_found(self, db):
        with pytest.raises(ValueError, match="not found"):
            handle_get_campaign_status(db, "nonexistent")

    def test_get_campaign_history(self, db, campaign_with_round):
        result = handle_get_campaign_history(db, "mcp-test")
        assert "rounds" in result
        assert len(result["rounds"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement MCP server**

```python
# src/agent_hpo/mcp_server.py
"""MCP server: agent-facing control plane with 5 tools."""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import ActionProposal


# --- Handler functions (testable without MCP transport) ---

def handle_list_campaigns(db: Database) -> list[dict]:
    with db.connection() as conn:
        cur = conn.execute("SELECT id, name, state, metric_name, objective_direction FROM campaigns ORDER BY id")
        return cur.fetchall()


def handle_get_campaign_status(db: Database, campaign_name: str) -> dict:
    service = CampaignService(db)
    with db.connection() as conn:
        cur = conn.execute("SELECT * FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    rounds = service.get_rounds(campaign["id"])
    latest_round = rounds[-1] if rounds else None
    return {
        **campaign,
        "total_rounds": len(rounds),
        "latest_round": latest_round,
    }


def handle_get_round_summary(db: Database, campaign_name: str, round_number: int | None = None) -> dict:
    with db.connection() as conn:
        cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    service = CampaignService(db)
    rounds = service.get_rounds(campaign["id"])

    if round_number is not None:
        target = [r for r in rounds if r["round_number"] == round_number]
        if not target:
            raise ValueError(f"Round {round_number} not found")
        return target[0]
    else:
        if not rounds:
            raise ValueError("No rounds found")
        return rounds[-1]


def handle_get_campaign_history(db: Database, campaign_name: str) -> dict:
    with db.connection() as conn:
        cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    service = CampaignService(db)
    return service.get_campaign_history(campaign["id"])


def handle_submit_action_proposal(db: Database, campaign_name: str, proposal_dict: dict) -> dict:
    with db.connection() as conn:
        cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    proposal = ActionProposal.from_dict(proposal_dict)
    service = CampaignService(db)
    return service.submit_proposal(campaign["id"], proposal)


# --- MCP Server setup ---

def create_server() -> Server:
    server = Server("agent-hpo")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="list_campaigns",
                description="List all optimization campaigns",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="get_campaign_status",
                description="Get current campaign state, active round, and config",
                inputSchema={
                    "type": "object",
                    "properties": {"campaign_name": {"type": "string"}},
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="get_round_summary",
                description="Get summary for a specific round or the latest round",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "campaign_name": {"type": "string"},
                        "round_number": {"type": "integer"},
                    },
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="get_campaign_history",
                description="Get all rounds and agent decisions for a campaign",
                inputSchema={
                    "type": "object",
                    "properties": {"campaign_name": {"type": "string"}},
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="submit_action_proposal",
                description="Propose the next action for a campaign (continue, narrow_search, widen_search, increase_budget, stop)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "campaign_name": {"type": "string"},
                        "proposal": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["continue", "narrow_search", "widen_search", "increase_budget", "stop"]},
                                "justification": {"type": "string"},
                                "proposed_search_space": {"type": "array"},
                                "proposed_budget": {"type": "integer"},
                                "reference_round_ids": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["action", "justification", "reference_round_ids"],
                        },
                    },
                    "required": ["campaign_name", "proposal"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        db = _get_db()
        try:
            if name == "list_campaigns":
                result = handle_list_campaigns(db)
            elif name == "get_campaign_status":
                result = handle_get_campaign_status(db, arguments["campaign_name"])
            elif name == "get_round_summary":
                result = handle_get_round_summary(db, arguments["campaign_name"], arguments.get("round_number"))
            elif name == "get_campaign_history":
                result = handle_get_campaign_history(db, arguments["campaign_name"])
            elif name == "submit_action_proposal":
                result = handle_submit_action_proposal(db, arguments["campaign_name"], arguments["proposal"])
            else:
                result = {"error": f"Unknown tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        finally:
            db.close()

    return server


def _get_db() -> Database:
    url = os.environ.get("AGENT_HPO_DB_URL", "postgresql://localhost:5432/agent_hpo")
    db = Database(url)
    db.setup_schema()
    return db


async def main():
    server = create_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcp_server.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent_hpo/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: MCP server with 5 agent-facing tools over core service layer"
```

---

### Task 12: Agent Skills

**Files:**
- Create: `skills/hpo-overview.md`
- Create: `skills/hpo-interpret-summary.md`
- Create: `skills/hpo-action-guidelines.md`

- [ ] **Step 1: Write hpo-overview skill**

```markdown
# src/agent_hpo/skills/hpo-overview.md
---
name: hpo-overview
description: System model for agent-driven hyperparameter optimization
---

# Agent-HPO Overview

You are orchestrating hyperparameter optimization campaigns. You operate **between** bounded optimization rounds — you never control what happens inside a round.

## Terminology

- **Campaign**: A long-lived optimization goal (e.g., "tune XGBoost for accuracy on breast cancer dataset")
- **Study round**: One optimization run with a fixed search space and trial budget
- **Trial**: One model training + evaluation within a round

## Your Role

You decide what happens **between** rounds:
- Review round summaries
- Decide the next action: continue, narrow_search, widen_search, increase_budget, or stop
- Justify every decision by referencing prior round history

You do NOT:
- Control individual trials
- Access raw Optuna data
- Modify the sampler
- See test set metrics (only validation)

## The Loop

1. A round runs (you wait)
2. You receive a summary
3. You decide the next action via `submit_action_proposal`
4. If accepted, the next round runs

## Available MCP Tools

- `list_campaigns` — see all campaigns
- `get_campaign_status` — current state and config
- `get_round_summary` — summary for a round
- `get_campaign_history` — all rounds and decisions
- `submit_action_proposal` — propose your next action

## CLI Commands (for execution)

- `agent-hpo run <name>` — execute the next round
- `agent-hpo status <name>` — check campaign state
```

- [ ] **Step 2: Write hpo-interpret-summary skill**

```markdown
# src/agent_hpo/skills/hpo-interpret-summary.md
---
name: hpo-interpret-summary
description: How to read and interpret round summary fields
---

# Interpreting Round Summaries

## Key Fields

| Field | What it tells you |
|---|---|
| `best_score` | Best metric value across ALL rounds (cumulative) |
| `delta_from_prev` | How much the cumulative best improved this round |
| `round_best_score` | Best in THIS round only (None if all trials failed/pruned) |
| `new_best_in_round` | Did this round find a new overall best? |
| `completed_trials` / `round_completed_trials` | How many trials actually produced results |
| `plateau_signal` | No improvement in the last 30% of this round's trials |
| `param_importance` | Which parameters matter most (fANOVA) |
| `generalization_gap` | Average |train_score - val_score| — overfitting signal |
| `failure_rate` | Fraction of errored trials (something is wrong if > 0.1) |
| `pruned_rate` | Fraction of pruned trials (healthy if moderate, concerning if > 0.5) |

## Red Flags

- **failure_rate > 0.1**: Something is broken in the search space or data
- **generalization_gap > 0.1**: Model is overfitting — consider narrowing or regularization params
- **plateau_signal = true AND new_best_in_round = false**: This search space may be exhausted
- **round_completed_trials = 0**: All trials failed or were pruned — urgent issue
- **pruned_rate > 0.5**: Search space may include many bad regions — consider narrowing

## What Null Values Mean

- `best_score = None`: No completed trial in the entire campaign yet
- `round_best_score = None`: No completed trial in this specific round
- `delta_from_prev = None`: First round, or no previous best to compare against
```

- [ ] **Step 3: Write hpo-action-guidelines skill**

```markdown
# src/agent_hpo/skills/hpo-action-guidelines.md
---
name: hpo-action-guidelines
description: Decision framework for choosing next actions in HPO campaigns
---

# Action Guidelines

## Decision Framework

After each round, choose exactly one action:

### continue
**When:** Score is still improving, search space seems right, no red flags.
**Effect:** Adds more trials to the same Optuna study with the same search space.

### narrow_search
**When:** `param_importance` shows some params dominate, `param_ranges_used` shows the best trials cluster in a subregion, or you want to exploit a promising area.
**Effect:** Creates a new Optuna study with tighter parameter ranges. Use `param_importance` and `param_ranges_used` to decide which params to narrow and by how much.

### widen_search
**When:** `plateau_signal` is true, exploration seems insufficient, or the best params are hitting range boundaries.
**Effect:** Creates a new Optuna study with broader parameter ranges.

### increase_budget
**When:** The round showed progress but may not have converged — you want more trials with the same space.
**Effect:** Adds more trials (higher budget) to the same study.

### stop
**When:** Target metric reached, plateau with no improvement across multiple rounds, or diminishing returns.

## Rules You Must Follow

1. **One structural change per round.** narrow and widen are structural. Don't combine with other structural changes.
2. **Cooldown on reversals.** If you just narrowed, you cannot widen for 2 rounds (and vice versa). This prevents oscillation.
3. **Reference history.** Every justification must cite specific round IDs and their results. "It's not improving" is not enough — cite the scores.
4. **No dataset heuristics.** Your decisions should be based on summary signals, not assumptions about specific datasets.

## Justification Template

> "Based on rounds [X, Y, Z]: best_score improved from A to B (+C) over the last N rounds.
> param_importance shows [param] dominates at D%. The convergence curve shows [pattern].
> Action: [action] because [reasoning]."
```

- [ ] **Step 4: Commit**

```bash
git add src/agent_hpo/skills/
git commit -m "feat: agent skills for HPO overview, summary interpretation, and action guidelines"
```

---

### Task 13: Benchmark Datasets and Runner

**Files:**
- Create: `benchmarks/datasets.py`
- Create: `benchmarks/run_benchmark.py`

- [ ] **Step 1: Write dataset loader (packaged under agent_hpo)**

```python
# src/agent_hpo/datasets.py
"""Dataset loading with consistent train/val/test splits."""

from __future__ import annotations

from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split

from agent_hpo.core.models import DatasetSplit

DATASETS = {
    "breast_cancer": {"loader": load_breast_cancer, "metric": "accuracy", "direction": "maximize"},
    "california_housing": {"loader": fetch_california_housing, "metric": "rmse", "direction": "minimize"},
    "digits": {"loader": load_digits, "metric": "accuracy", "direction": "maximize"},
}


def load_dataset(name: str, seed: int = 42) -> tuple[DatasetSplit, dict]:
    """Load a dataset with consistent splits. Returns (split, metadata)."""
    info = DATASETS[name]
    X, y = info["loader"](return_X_y=True)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    split = DatasetSplit(X_train, y_train, X_val, y_val, X_test, y_test)
    return split, {"metric": info["metric"], "direction": info["direction"]}
```

- [ ] **Step 2: Write benchmark runner**

```python
# benchmarks/run_benchmark.py
"""Benchmark: agent-driven campaign vs plain Optuna baseline."""

from __future__ import annotations

import json
import time

import click
import optuna

from agent_hpo.datasets import load_dataset, DATASETS
from agent_hpo.backends.xgboost import XGBoostBackend
from agent_hpo.core.models import ParamSpec


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
```

- [ ] **Step 3: Test the benchmark runs**

Run: `cd /Users/huijokim/personal/agent_param_optimization && python -m benchmarks.run_benchmark --dataset breast_cancer --total-trials 10 --seeds 42`
Expected: Completes with printed results

- [ ] **Step 4: Commit**

```bash
git add benchmarks/
git commit -m "feat: benchmark datasets and baseline runner for agent vs plain Optuna comparison"
```

---

### Task 14: Integration Test — End-to-End Campaign

**Files:**
- Create: `tests/test_integration.py`

One test that exercises the full flow: create campaign, run round 1, produce summary, simulate agent decision, verify state.

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end test: full campaign lifecycle without a real agent."""

import pytest
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

    search_space = [ParamSpec.from_dict(s) for s in r1["search_space"]]
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
    assert r2["optuna_study_name"] == r1["optuna_study_name"]  # reuses study
    assert r2["trial_offset"] == r1["trial_end"]

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
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for full campaign lifecycle"
```

---

### Task 15: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: address any issues found in full test suite run"
```
