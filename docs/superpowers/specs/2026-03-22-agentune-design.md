# Agent-Driven Hyperparameter Optimization — Design Spec

## Overview

A pip-installable Python library that combines Optuna-based hyperparameter optimization with an LLM agent layer (via Claude Code). The agent operates **between** bounded optimization rounds — never inside them. Traditional optimization stays deterministic; the agent reasons about results and decides what to try next.

The system exposes a **CLI** for execution, **MCP tools** as the agent control plane, and **Claude Code skills** to teach the agent the reasoning framework.

## Terminology

| Term | Meaning |
|---|---|
| **Campaign** | A long-lived optimization goal (e.g., "tune XGBoost for California Housing RMSE") |
| **Study round** | One agent decision + one bounded Optuna optimization run |
| **Trial** | One objective function evaluation within a study round |

## Architecture

### Layer Diagram

```
┌─────────┐  ┌────────────┐
│   CLI   │  │ MCP Server │
└────┬────┘  └─────┬──────┘
     │             │
     └──────┬──────┘
            │
   ┌────────▼────────┐
   │  Core Service   │  ← validation, state transitions, locking
   │     Layer       │
   └───┬────┬────┬───┘
       │    │    │
       ▼    ▼    ▼
  Campaign  Optuna   Summarizer
   Schema  Storage   (internal)
 (Postgres)(Postgres)
```

### Component Ownership

| Component | Responsibility | Talks to |
|---|---|---|
| **CLI** (`agentune`) | Thin user-facing commands over core | Core service layer |
| **Scheduler** (internal module) | Launches rounds, applies stop conditions, returns control at checkpoints | Core service layer |
| **Summarizer** (internal module) | Produces machine-readable round summary from Optuna data | Optuna storage (read), campaign schema (write) |
| **MCP Server** | Agent-facing control plane only | Core service layer |
| **Agent Skills** | Teach Claude how to orchestrate | MCP tools (via Claude Code) |
| **Core Service Layer** | Validation, state transitions, locking, campaign/round management | Campaign schema, Optuna storage |

### What Each Component Does NOT Do

- CLI does not reason about search spaces — that's the agent's job
- Scheduler does not decide actions — it enforces stop conditions and passes control
- Summarizer does not interpret — it computes fixed fields
- MCP does not touch Optuna internals — it reads curated views from core
- Skills encode process and guardrails, not dataset-specific heuristics or hidden optimization policies

### Storage Split

- **Optuna native Postgres storage:** Trial-level truth — params, metrics, intermediate values, pruning, sampler state
- **Campaign schema (Postgres, 3 tables):** Agent control state — campaigns, study_rounds, agent_decisions
- **Local filesystem:** Model artifacts, large outputs

The agent never sees raw Optuna tables. It reads curated summaries through MCP tools.

## Core Loop

1. A study spec exists (search space, budget). For round 1, `agentune init` creates it automatically from campaign defaults (`initial_search_space`, `trials_per_round`). For subsequent rounds, the agent proposes it via `submit_action_proposal`.
2. CLI executes the study round via Optuna — deterministic, no agent involvement mid-study
3. Scheduler checks stop conditions (max_total_trials, max_wall_time, target_score). If a hard stop fires, campaign ends — agent is not consulted.
4. Summarizer produces a compact, immutable, schema-versioned report
5. Scheduler checks patience-based stop conditions (no improvement for N rounds). If fired, campaign ends.
6. Agent reads the report + full campaign history and picks one action

### Execution Model (v0)

`agentune run` drives one round and returns at `AWAITING_AGENT`, `COMPLETED`, or `FAILED`. Scheduler is an internal module — no long-running daemon process in v0.

## State Machines

### Campaign States

```
CREATED → RUNNING → PAUSE_REQUESTED → PAUSED → RUNNING → COMPLETED
                  ↘ FAILED                              ↗
                  ↘ STOPPED (manual)
```

**Transitions:**
- `CREATED → RUNNING`: Triggered by `agentune run` when the first round begins
- `RUNNING → PAUSE_REQUESTED`: Triggered by `agentune pause`. Sets a persisted flag on the campaign row (`pause_requested = true`). The scheduler checks this flag after each round completes and transitions to `PAUSED` instead of continuing.
- `PAUSE_REQUESTED → PAUSED`: Applied by scheduler when the active round reaches `AWAITING_AGENT` or `CLOSED`. The flag is cleared.
- `PAUSED → RUNNING`: Triggered by `agentune resume`
- `RUNNING → COMPLETED`: Triggered when a stop condition fires or the agent proposes `stop`
- `RUNNING → FAILED`: Triggered when a study round fails and exceeds max retries (3), indicating an unrecoverable problem
- `RUNNING → STOPPED`: Triggered by `agentune stop` (manual operator intervention). Terminal state — cannot be resumed. Differs from `COMPLETED` in intent: `STOPPED` means "aborted by operator," `COMPLETED` means "goal met or stop condition fired."

### Study Round States

```
PROPOSED → RUNNING → SUMMARIZING → AWAITING_AGENT → RESOLVED
              ↓ ↑        ↓ ↑                ↓
            FAILED     FAILED             CLOSED
              ↓          ↓
           RETRYING   RETRYING
```

- `PROPOSED`: Round created, not yet executing. Round 1 is system-generated by `agentune init` from campaign defaults (no agent proposal needed, `references` is empty). Subsequent rounds are created from accepted agent proposals.
- `RUNNING`: Optuna study executing
- `SUMMARIZING`: Round complete, summarizer producing report
- `AWAITING_AGENT`: Summary ready, waiting for agent's next action
- `RESOLVED`: Agent has acted on this round, round is sealed (immutable)
- `CLOSED`: Round sealed without agent action — a hard stop condition (max_total_trials, max_wall_time, target_score) fired after this round completed. Campaign transitions to `COMPLETED`. Summary is still written if possible.
- `FAILED`: Execution or summarization error. Source state is preserved (`failed_from` field on the row).
- `RETRYING`: Explicit transition from `FAILED`. Scheduler increments `retry_count`, sets state to `RETRYING`, then transitions to the `failed_from` state (back to `RUNNING` or `SUMMARIZING`). Max 3 retries, then escalates to campaign `FAILED`.

**Round closing has exactly two terminal states:** `RESOLVED` (agent acted) and `CLOSED` (stop condition fired without agent). This eliminates ambiguity about whether the agent needs to be consulted.

## Database Schema

Three tables in the campaign schema. Optuna manages its own tables separately via `RDBStorage`.

```sql
CREATE TABLE campaigns (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    state           TEXT NOT NULL DEFAULT 'CREATED',  -- CREATED, RUNNING, PAUSE_REQUESTED, PAUSED, COMPLETED, FAILED, STOPPED
    pause_requested BOOLEAN NOT NULL DEFAULT false,   -- persisted flag, checked by scheduler after each round
    metric_name     TEXT NOT NULL,
    objective_direction TEXT NOT NULL,                 -- 'minimize' or 'maximize'
    backend         TEXT NOT NULL,                     -- e.g., 'xgboost'
    sampler_config  JSONB NOT NULL,                    -- frozen at creation
    initial_search_space JSONB NOT NULL,               -- list of ParamSpec as JSON
    improvement_criteria JSONB NOT NULL,               -- {mode, threshold}
    stop_conditions JSONB NOT NULL,                    -- {max_rounds, max_total_trials, ...}
    trials_per_round INT NOT NULL,
    claimed_by      TEXT,                              -- worker id holding the lease (NULL = unclaimed)
    claim_expires_at TIMESTAMPTZ,                      -- lease expiry (NULL = unclaimed)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE study_rounds (
    id              SERIAL PRIMARY KEY,
    campaign_id     INT NOT NULL REFERENCES campaigns(id),
    round_number    INT NOT NULL,                      -- 1-indexed sequential within campaign
    state           TEXT NOT NULL DEFAULT 'PROPOSED',   -- PROPOSED, RUNNING, SUMMARIZING, AWAITING_AGENT, RESOLVED, CLOSED, FAILED, RETRYING
    failed_from     TEXT,                              -- state before FAILED (RUNNING or SUMMARIZING), used for retry
    parent_round_id INT REFERENCES study_rounds(id),   -- lineage for narrow/widen
    optuna_study_name TEXT NOT NULL,                    -- reference to Optuna study
    search_space    JSONB NOT NULL,                     -- active search space for this round
    budget          INT NOT NULL,                       -- n_trials allocated for this round
    trial_offset    INT NOT NULL,                       -- first trial number in the Optuna study belonging to this round
    trial_end       INT,                                -- last trial number (exclusive), written when round completes
    summary         JSONB,                              -- RoundSummary as JSON, immutable once written
    summary_schema_version INT,                         -- version of summary schema
    retry_count     INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(campaign_id, round_number)
);

CREATE TABLE agent_decisions (
    id              SERIAL PRIMARY KEY,
    campaign_id     INT NOT NULL REFERENCES campaigns(id),
    round_id        INT NOT NULL REFERENCES study_rounds(id),  -- the round this decision was made AFTER
    action          TEXT NOT NULL,                      -- continue, narrow_search, widen_search, increase_budget, stop
    justification   TEXT NOT NULL,
    proposed_search_space JSONB,                        -- for narrow/widen
    proposed_budget INT,                                -- for increase_budget
    reference_round_ids JSONB NOT NULL,                  -- list of round_ids
    accepted        BOOLEAN NOT NULL,                   -- whether core validated and accepted
    rejection_reason TEXT,                              -- if not accepted
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Nested types (`dict`, `list[ParamSpec]`) are stored as JSONB columns. No normalization needed for v0 — these are opaque to SQL queries; the application layer deserializes them.

## Concurrency Control

Advisory locks are wrong here — Optuna rounds are long-running and cannot be protected by holding a transaction open. Instead, use an explicit lease/claim model.

**Lease model:**
- The `campaigns` table has `claimed_by TEXT` (worker id) and `claim_expires_at TIMESTAMPTZ` columns.
- Before starting work, a worker attempts to claim the campaign: `UPDATE campaigns SET claimed_by = :worker_id, claim_expires_at = now() + interval '15 minutes' WHERE id = :id AND (claimed_by IS NULL OR claim_expires_at < now())`. If 0 rows affected, another worker holds the lease.
- The worker refreshes the lease periodically (every 5 minutes) during long-running rounds.
- On completion or failure, the worker releases the claim: `SET claimed_by = NULL, claim_expires_at = NULL`.
- If a worker crashes, the lease expires and another worker can claim it. The round will be in `RUNNING` or `SUMMARIZING` state and the new worker handles it as a retry.

**Short-lived mutations** (state transitions, writing summaries, accepting proposals) still use `pg_advisory_xact_lock` within their individual transactions for atomicity.

## Data Models

### Campaign Configuration

```python
@dataclass
class CampaignConfig:
    metric_name: str                    # e.g., "accuracy", "rmse"
    objective_direction: Literal["minimize", "maximize"]
    backend: str                        # e.g., "xgboost"
    sampler_config: dict                # frozen at creation, agent cannot modify
    initial_search_space: list[ParamSpec]
    improvement_criteria: ImprovementCriteria
    stop_conditions: StopConditions
    trials_per_round: int               # default budget per round
```

### Stop Conditions

```python
@dataclass
class StopConditions:
    max_rounds: int | None              # hard cap on total study rounds
    max_total_trials: int | None        # hard cap on cumulative trials across all rounds
    max_wall_time_seconds: float | None # hard cap on cumulative wall time
    patience_rounds: int                # stop if no improvement (per ImprovementCriteria) for N consecutive rounds
    target_score: float | None          # stop if best_score meets this threshold (direction-aware)
```

**Evaluation:** Hard caps (`max_total_trials`, `max_wall_time_seconds`, `target_score`) are checked both before and after each round. Patience (`patience_rounds`) is checked after summarization. First condition met wins.

**Pre-round budget clipping:** Before a round starts, the scheduler clips its budget so it cannot exceed hard caps:
- If `max_total_trials` is set: `effective_budget = min(budget, max_total_trials - cumulative_trials_so_far)`. If `effective_budget <= 0`, the campaign ends without running the round.
- If `max_wall_time_seconds` is set and cumulative wall time already exceeds it, the campaign ends without running the round. (Wall time cannot be precisely pre-clipped since trial duration varies, but the check prevents starting a round that is already over budget.)

This ensures the campaign never overshoots its hard caps and baseline comparison is fair.

### Improvement Criteria

```python
@dataclass
class ImprovementCriteria:
    mode: Literal["strict_better", "min_absolute_delta", "min_relative_delta"]
    threshold: float                    # 0.0 for strict_better, delta value otherwise
```

**Exact definitions:**
- `strict_better`: `new_best > prev_best` (maximize) or `new_best < prev_best` (minimize). Any improvement counts.
- `min_absolute_delta`: `|new_best - prev_best| >= threshold` in the favorable direction. E.g., threshold=0.01 means accuracy must improve by at least 0.01.
- `min_relative_delta`: `|new_best - prev_best| / |prev_best| >= threshold` in the favorable direction. E.g., threshold=0.05 means a 5% relative improvement. If `prev_best == 0`, falls back to `min_absolute_delta` with the same threshold to avoid division by zero.

Rounds with `round_completed_trials == 0` are counted as "no improvement" for patience purposes.

### Round Summary (immutable, schema-versioned)

```python
@dataclass
class RoundSummary:
    schema_version: int
    round_id: int
    campaign_id: int

    # Campaign context
    metric_name: str
    objective_direction: str

    # Performance (cumulative — from all completed trials across all rounds)
    best_score: float | None            # None if no completed trial exists in campaign yet
    best_params: dict | None            # None if no completed trial exists in campaign yet
    delta_from_prev: float | None       # None for first round or if best_score is None
    total_trials: int
    completed_trials: int               # trials with COMPLETE status (excludes pruned/failed)

    # Performance (round-local)
    trials_added: int
    round_completed_trials: int         # completed trials in this round specifically
    new_best_in_round: bool             # False if round_completed_trials == 0
    round_best_score: float | None      # None if round_completed_trials == 0

    # Convergence (round-local: trial indices are 0 to trials_added-1)
    convergence_curve: list[tuple[int, float]]  # (round_local_trial_index, best_score_at_point)
    plateau_signal: bool                # True if no improvement in last 30% of this round's trials

    # Parameter analysis
    param_importance: dict[str, float]  # fANOVA importance
    param_ranges_used: dict[str, tuple]

    # Health
    generalization_gap: float | None    # train vs validation delta
    failure_rate: float                 # fraction of errored trials
    pruned_rate: float                  # fraction of pruned trials (separate — pruning is often healthy)

    # Cost
    round_wall_time_seconds: float
    total_wall_time_seconds: float

    # Lineage
    parent_round_id: int | None
    optuna_study_name: str
    action_that_created_this_round: str
```

**Critical rule:** The agent never sees held-out test metrics. Only validation-time signals drive actions.

### Action Proposal

```python
@dataclass
class ActionProposal:
    action: Literal["continue", "narrow_search", "widen_search", "increase_budget", "stop"]
    justification: str
    proposed_search_space: dict | None  # required for narrow/widen
    proposed_budget: int | None         # required for increase_budget
    reference_round_ids: list[int]      # round_ids this decision is based on
```

## Agent Action Space

| Action | Meaning | Optuna Effect |
|---|---|---|
| `continue` | Same search space, add another chunk of trials | Creates new study_round row, but reuses the same Optuna `Study` object (appends trials) |
| `narrow_search` | Tighten param ranges based on importance | Creates new study_round row AND new Optuna `Study` (with lineage to previous round) |
| `widen_search` | Expand param ranges to explore more | Creates new study_round row AND new Optuna `Study` (with lineage to previous round) |
| `increase_budget` | Raise per-round trial count | Creates new study_round row, reuses same Optuna `Study`, runs with higher n_trials |
| `stop` | Campaign goal met or diminishing returns | No new round. Records decision in `agent_decisions`, transitions campaign to COMPLETED. |

**Clarification on round vs Optuna study lifecycle:** Every action *except `stop`* creates a new `study_rounds` row in the campaign schema. `stop` is a terminal decision — it is recorded in `agent_decisions` only and closes the campaign. The distinction is whether it also creates a new Optuna `Study`. `continue` and `increase_budget` reuse the existing Optuna study (appending trials preserves the sampler's learned distribution). `narrow_search` and `widen_search` create a fresh Optuna study because the search space has changed.

### Trial Boundaries and Budget Enforcement

When multiple rounds share the same Optuna study (`continue`, `increase_budget`), the system must track which trials belong to which round. Without this, round-local summaries, cooldown logic, and baseline comparison are not reconstructable.

**Before a round starts:**
1. Read the current trial count from the Optuna study: `current_count = len(study.trials)`
2. Write `trial_offset = current_count` to the `study_rounds` row
3. Run exactly `budget` trials via `study.optimize(objective, n_trials=budget)`
4. After completion, write `trial_end = trial_offset + actual_completed_trials` to the row

**Invariant:** For rounds sharing an Optuna study, `round[N].trial_offset == round[N-1].trial_end`. No gaps, no overlaps.

**Summarizer uses these boundaries:** When computing round-local stats (`trials_added`, `round_best_score`, `convergence_curve`, `failure_rate`, `pruned_rate`), the summarizer filters Optuna trials to `[trial_offset, trial_end)`. Cumulative stats use all trials `[0, trial_end)`.

**For new Optuna studies** (`narrow_search`, `widen_search`): `trial_offset` is always 0 and `trial_end` equals the total trials in that study. The boundary tracking is trivially correct but still persisted for uniform handling.

### Guardrails (enforced by core, not agent)

- One structural change (narrow/widen) per round
- Cooldown: cannot reverse a narrow with a widen (or vice versa) within the next 2 rounds after the structural change. "Reverse" means any action in the opposite direction on any parameter, not just the same parameter.
- `references` must include at least the most recent round (not applicable to system-generated round 1)
- `proposed_search_space` validated against backend's parameter definitions
- Sampler config is frozen at campaign creation — agent cannot modify it

## MCP Tools (Agent Control Plane)

| Tool | Purpose |
|---|---|
| `get_campaign_status` | Current state, active round, improvement criteria, stop conditions |
| `get_round_summary` | Summary for a specific round (or latest) |
| `get_campaign_history` | All rounds + agent decisions, ordered |
| `submit_action_proposal` | Propose next action — validated by core before applying |
| `list_campaigns` | Overview of all campaigns |

Small surface area. No raw Optuna access, no direct writes to storage.

## Agent Skills (3 files)

1. **`hpo-overview`** — System model: campaign/round/trial, the core loop, what the agent controls vs what it doesn't
2. **`hpo-interpret-summary`** — How to read each summary field, what signals matter, red flags (high failure rate, large generalization gap, plateau with no improvement)
3. **`hpo-action-guidelines`** — Decision framework: when to narrow vs widen, cooldown awareness, how to write justifications that reference history, when to stop

Skills encode process and guardrails. No dataset-specific heuristics.

## CLI Commands

| Command | What it does |
|---|---|
| `agentune init` | Create a campaign (metric, direction, backend, improvement criteria, sampler, stop conditions) |
| `agentune run` | Execute current/next study round — returns at AWAITING_AGENT, COMPLETED, or FAILED |
| `agentune status` | Print campaign state + latest round summary |
| `agentune history` | Print all rounds and decisions |
| `agentune pause` | Transition campaign to PAUSED |
| `agentune resume` | Transition campaign from PAUSED to RUNNING |
| `agentune stop` | Manually stop a campaign |
| `agentune export` | Export best params, model artifact path |
| `agentune baseline` | Run a single plain Optuna study with same total budget — the comparison target |

All thin wrappers over the core service layer.

## Model Backend Interface

```python
@dataclass
class ParamSpec:
    """Defines one hyperparameter's search space."""
    name: str
    type: Literal["float", "int", "categorical"]
    low: float | None = None            # for float/int
    high: float | None = None           # for float/int
    log: bool = False                   # log-scale sampling
    choices: list | None = None         # for categorical

@dataclass
class DatasetSplit:
    """Train/validation/test arrays, pre-split."""
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any   # never exposed to agent; used only in final benchmark report
    y_test: Any

class ObjectiveBackend(Protocol):
    def create_objective(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        search_space: list[ParamSpec],
    ) -> Callable[[optuna.Trial], float]:
        """Return an Optuna objective function.

        The returned callable takes an optuna.Trial, suggests params from the
        provided search_space, trains a model on dataset.X_train/y_train,
        evaluates on dataset.X_val/y_val, and returns the metric value.
        Must also log train metric for generalization_gap.

        search_space is the active round's search space — this is how the
        agent's narrow_search/widen_search decisions flow into trial execution.
        The objective uses these ParamSpecs to call trial.suggest_*() methods.
        """
        ...

    def default_search_space(self) -> list[ParamSpec]:
        """Return default parameter search space as structured specs."""
        ...

    def param_definitions(self) -> list[ParamSpec]:
        """Return all valid parameter definitions (used for validating proposed_search_space)."""
        ...
```

v0 implementation: `XGBoostBackend`. Interface exists for generality but we ship one backend. Backend registration is a simple dict mapping in `backends/__init__.py` (`{"xgboost": XGBoostBackend}`); no plugin system in v0.

## Validation & Benchmarking

### Datasets

| Dataset | Task | Why |
|---|---|---|
| Breast Cancer | Binary classification | Small, fast iterations, easy to show convergence |
| California Housing | Regression | Different metric (RMSE), different objective direction (minimize) |
| Digits | Multi-class classification | Higher-dimensional, tests whether narrow/widen decisions help |

### Evaluation Protocol

**Fairness constraints:**
- Same initial search space
- Same sampler family and config
- Same dataset split seeds
- Same compute budget (wall time, not just trial count)

**Comparison:**
1. Run `agentune baseline` — plain Optuna, same total budget
2. Run agent-driven campaign — same budget
3. Compare: final metric, trials to reach baseline's best, stability across 3 seeds
4. Report held-out test set performance for final results (never seen by agent during campaign)

**Success criterion:** The agent layer is only justified if it matches or beats baseline under the same budget.

## Package Structure

```
agent_param_optimization/
├── pyproject.toml
├── src/
│   └── agentune/
│       ├── core/
│       │   ├── campaign.py        # campaign + round management
│       │   ├── state.py           # state machines, transitions
│       │   ├── locking.py         # pg_advisory_xact_lock
│       │   └── models.py          # dataclasses: RoundSummary, ActionProposal, etc.
│       ├── backends/
│       │   ├── base.py            # ObjectiveBackend protocol
│       │   └── xgboost.py         # XGBoostBackend
│       ├── summarizer.py          # Optuna data → RoundSummary
│       ├── scheduler.py           # round orchestration, stop conditions
│       ├── cli.py                 # click/typer CLI, thin over core
│       └── mcp_server.py          # MCP tool definitions
├── skills/
│   ├── hpo-overview.md
│   ├── hpo-interpret-summary.md
│   └── hpo-action-guidelines.md
├── benchmarks/
│   ├── run_benchmark.py           # agent vs baseline comparison
│   └── datasets.py                # sklearn dataset loaders
└── tests/
```

## Key Design Decisions

1. Agent decides between studies, not inside them
2. Optuna owns optimization state; campaign schema owns agent control state
3. MCP is agent control plane only, not runtime data path
4. Round summaries are immutable and schema-versioned
5. Sampler config frozen at campaign creation
6. Agent never sees test metrics
7. `continue`/`increase_budget` append to same Optuna study; `narrow`/`widen` create new study with lineage
8. Anti-oscillation: one structural change per round + cooldown on reversals + history-referenced justifications
9. Transaction-scoped advisory locks prevent concurrent mutations
10. Benchmark fairness measured by wall time, not trial count
