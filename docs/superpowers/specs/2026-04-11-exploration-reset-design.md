# Exploration Reset: Auto-restart on Plateau in Strong-Exploration Mode

## Problem

In strong-exploration mode with `--max-wall-time 86400` (24 hours), the campaign terminates when patience triggers — no improvement for N consecutive rounds. The agent stops, wasting remaining time budget.

Users want: "keep searching different parameter strategies until the clock runs out, report the best result."

## Solution

When patience triggers and `mode == "strong-exploration"`, the runner **auto-resets** instead of completing the campaign:

1. Selects a new param subset from the backend's full catalog (coverage-based — prioritizes untried params)
2. Creates a new round with a fresh Optuna study
3. Returns `AWAITING_AGENT` so the agent can override the auto-selected params or accept them

Only `max_wall_time` (and other hard stops) can terminate the campaign. Patience resets with each exploration reset, giving each param subset a fair chance.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent involvement | Hybrid — system auto-selects, agent can override | Runs unattended but agent retains control |
| Param selection | Coverage-based — prioritize untried params | Systematic exploration of the full catalog |
| Reset limits | Unlimited — only wall time stops | "Run until time runs out" use case |
| Global best tracking | Campaign-level (existing) | Summarizer already tracks best across all rounds |

## Core Mechanism

### 1. Runner patience check modification (`src/agentune/runner.py`)

Current behavior (line ~363):
```python
if Scheduler.check_patience(all_summaries, ...):
    return self._complete_after_summary(campaign_id, ..., "patience")
```

New behavior:
```python
if Scheduler.check_patience(all_summaries, ...):
    if campaign.get("mode") == "strong-exploration":
        # Auto-reset: pick new params, create new round, continue
        new_params = select_exploration_params(backend, rounds)
        reset_number = current_round.get("reset_number", 0) + 1
        # Create new round with new_params, reset_number, fresh study
        # Return AWAITING_AGENT
    else:
        return self._complete_after_summary(campaign_id, ..., "patience")
```

### 2. Patience scoping by reset boundary

Add `reset_number` column to `study_rounds` table. Patience only considers rounds with the same `reset_number` as the current round:

```python
current_reset = current_round.get("reset_number", 0)
reset_rounds = [r for r in rounds if r.get("reset_number", 0) == current_reset]
all_summaries = self._round_summaries(reset_rounds)
```

Hard stops (`max_wall_time`, `max_total_trials`, `target_score`) still look at all rounds across all resets.

### 3. Coverage-based param selection (`src/agentune/exploration.py`)

```python
def select_exploration_params(backend, rounds) -> list[ParamSpec]:
    catalog = backend.available_params()
    
    # Count how many resets each param appeared in
    coverage = {p.name: 0 for p in catalog}
    seen_resets = {}  # param_name -> set of reset_numbers
    for r in rounds:
        search_space = r.get("search_space", [])
        reset_num = r.get("reset_number", 0)
        for p in search_space:
            seen_resets.setdefault(p["name"], set()).add(reset_num)
    for name, resets in seen_resets.items():
        coverage[name] = len(resets)
    
    # Always include core params
    core = {"learning_rate", "n_estimators"}
    
    # Score remaining params: priority = 1 / (times_used + 1)
    remaining = [p for p in catalog if p.name not in core]
    remaining.sort(key=lambda p: coverage.get(p.name, 0))
    
    # Pick 7 more (least-used first), total = 9
    selected_names = core | {p.name for p in remaining[:7]}
    return [p for p in catalog if p.name in selected_names]
```

## DB Changes

### `study_rounds` table

Add column:
```sql
reset_number INT NOT NULL DEFAULT 0
```

Migration:
```sql
IF NOT EXISTS (SELECT 1 FROM information_schema.columns
    WHERE table_name = 'study_rounds' AND column_name = 'reset_number') THEN
    ALTER TABLE study_rounds ADD COLUMN reset_number INT NOT NULL DEFAULT 0;
END IF;
```

## Files Changed

| File | Change |
|---|---|
| `src/agentune/core/db.py` | Add `reset_number` column + migration |
| `src/agentune/runner.py` | Scope patience by reset_number; add auto-reset on patience in strong-exploration |
| `src/agentune/exploration.py` | New file: `select_exploration_params()` |
| `tests/test_exploration.py` | Tests for param selection coverage logic |
| `tests/test_runner.py` or `tests/test_campaign.py` | Tests for reset behavior: patience scoping, auto-reset trigger, agent override |

## Files NOT Changed

- `models.py` — no new dataclass fields
- `campaign.py` — proposal validation unchanged
- `mcp_server.py` — MCP tools unchanged
- `cli.py` — no new flags (reset is automatic in strong-exploration)
- `scheduler.py` — patience logic unchanged, just called with scoped data

## What the Agent Sees

After a reset, `run_next_round` returns `AWAITING_AGENT` with a new round containing auto-selected params. The agent can:
- Call `run_next_round` to accept the auto-selected params
- Call `revise_search` to override with its own param selection
- Call `narrow_search`/`widen_search` to adjust ranges

No new MCP tools needed. The reset is transparent — just another round with different params.
