# Exploration Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When patience triggers in `strong-exploration` mode, auto-reset with a new coverage-based param subset instead of terminating the campaign.

**Architecture:** Add `reset_number` column to `study_rounds` to scope patience per-reset. Add `src/agentune/exploration.py` with coverage-based param selection. Modify runner's patience check to auto-reset in strong-exploration mode by creating a new round with auto-selected params.

**Tech Stack:** Python 3.11, psycopg, pytest, PostgreSQL

---

### Task 1: Create `select_exploration_params` utility

**Files:**
- Create: `src/agentune/exploration.py`
- Test: `tests/test_exploration.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_exploration.py`:

```python
import pytest
from agentune.core.models import ParamSpec
from agentune.exploration import select_exploration_params


class FakeBackend:
    """Minimal backend for testing param selection."""
    def available_params(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
            ParamSpec(name="n_estimators", type="int", low=50, high=500),
            ParamSpec(name="max_depth", type="int", low=1, high=15),
            ParamSpec(name="min_child_weight", type="float", low=1.0, high=10.0),
            ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ParamSpec(name="colsample_bytree", type="float", low=0.5, high=1.0),
            ParamSpec(name="gamma", type="float", low=0.0, high=5.0),
            ParamSpec(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="max_leaves", type="int", low=0, high=256),
            ParamSpec(name="max_bin", type="int", low=32, high=1024),
        ]


class TestSelectExplorationParams:
    def test_returns_9_params(self):
        backend = FakeBackend()
        result = select_exploration_params(backend, [])
        assert len(result) == 9

    def test_always_includes_learning_rate(self):
        backend = FakeBackend()
        result = select_exploration_params(backend, [])
        names = {p.name for p in result}
        assert "learning_rate" in names

    def test_prioritizes_untried_params(self):
        backend = FakeBackend()
        # Simulate rounds where first 9 params were used in reset 0
        rounds = [
            {
                "search_space": [
                    {"name": "learning_rate"}, {"name": "n_estimators"},
                    {"name": "max_depth"}, {"name": "min_child_weight"},
                    {"name": "subsample"}, {"name": "colsample_bytree"},
                    {"name": "gamma"}, {"name": "reg_alpha"}, {"name": "reg_lambda"},
                ],
                "reset_number": 0,
            },
        ]
        result = select_exploration_params(backend, rounds)
        names = {p.name for p in result}
        # Untried params (max_leaves, max_bin) should be included
        assert "max_leaves" in names
        assert "max_bin" in names

    def test_returns_param_specs_from_catalog(self):
        """Returned ParamSpecs should have proper ranges from the backend catalog."""
        backend = FakeBackend()
        result = select_exploration_params(backend, [])
        lr = next(p for p in result if p.name == "learning_rate")
        assert lr.low == 0.001
        assert lr.high == 1.0
        assert lr.log is True

    def test_small_catalog_returns_all(self):
        """If catalog has <= 9 params, return all of them."""
        class SmallBackend:
            def available_params(self):
                return [
                    ParamSpec(name="learning_rate", type="float", low=0.01, high=1.0, log=True),
                    ParamSpec(name="n_estimators", type="int", low=50, high=500),
                    ParamSpec(name="max_depth", type="int", low=1, high=15),
                ]
        result = select_exploration_params(SmallBackend(), [])
        assert len(result) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exploration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentune.exploration'`

- [ ] **Step 3: Implement `select_exploration_params`**

Create `src/agentune/exploration.py`:

```python
"""Coverage-based param selection for exploration resets."""

from __future__ import annotations

from agentune.core.models import ParamSpec

# Params that every backend needs — always included in selections
CORE_PARAM_NAMES = {"learning_rate"}

# Target number of params per exploration reset
TARGET_PARAM_COUNT = 9


def select_exploration_params(backend, rounds: list[dict]) -> list[ParamSpec]:
    """Select params for an exploration reset, prioritizing untried params.

    Always includes core params (learning_rate). Fills remaining slots
    with least-used params from the backend's full catalog.
    """
    catalog = backend.available_params()

    if len(catalog) <= TARGET_PARAM_COUNT:
        return list(catalog)

    # Count how many distinct resets each param appeared in
    seen_resets: dict[str, set[int]] = {}
    for r in rounds:
        search_space = r.get("search_space", [])
        if isinstance(search_space, str):
            import json
            search_space = json.loads(search_space)
        reset_num = r.get("reset_number", 0)
        for p in search_space:
            seen_resets.setdefault(p["name"], set()).add(reset_num)

    usage_count = {p.name: len(seen_resets.get(p.name, set())) for p in catalog}

    # Separate core params from the rest
    core = [p for p in catalog if p.name in CORE_PARAM_NAMES]
    remaining = [p for p in catalog if p.name not in CORE_PARAM_NAMES]

    # Sort by usage count (ascending) — least-used first
    remaining.sort(key=lambda p: usage_count.get(p.name, 0))

    # Fill remaining slots
    slots = TARGET_PARAM_COUNT - len(core)
    selected = core + remaining[:slots]
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exploration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentune/exploration.py tests/test_exploration.py
git commit -m "feat: add coverage-based param selection for exploration resets"
```

---

### Task 2: Add `reset_number` column to `study_rounds`

**Files:**
- Modify: `src/agentune/core/db.py:37-55` (study_rounds CREATE TABLE)
- Modify: `src/agentune/core/db.py:57-77` (migration block)
- Test: `tests/test_campaign.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_campaign.py`, add to `TestCampaignCreation`:

```python
    def test_round_has_reset_number(self, service, sample_config):
        campaign = service.create_campaign("test-reset-num", sample_config)
        rounds = service.get_rounds(campaign["id"])
        assert rounds[0]["reset_number"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_campaign.py::TestCampaignCreation::test_round_has_reset_number -v`
Expected: FAIL — `KeyError: 'reset_number'`

- [ ] **Step 3: Add `reset_number` column to schema and migration**

In `src/agentune/core/db.py`, add to the `study_rounds` CREATE TABLE (after `retry_count INT NOT NULL DEFAULT 0,`):

```sql
    reset_number    INT NOT NULL DEFAULT 0,
```

And add a migration in the `DO $$ BEGIN ... END $$;` block (before the closing `END $$;`):

```sql
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'study_rounds' AND column_name = 'reset_number') THEN
        ALTER TABLE study_rounds ADD COLUMN reset_number INT NOT NULL DEFAULT 0;
    END IF;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_campaign.py::TestCampaignCreation::test_round_has_reset_number -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentune/core/db.py tests/test_campaign.py
git commit -m "feat: add reset_number column to study_rounds"
```

---

### Task 3: Scope patience by `reset_number` in the runner

**Files:**
- Modify: `src/agentune/runner.py:363-368` (patience check)
- Test: `tests/test_runner.py` (new file or add to existing)

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_reset.py`:

```python
"""Tests for exploration reset behavior in the runner."""

import pytest
from agentune.core.db import Database
from agentune.core.campaign import CampaignService
from agentune.core.models import (
    CampaignConfig, ImprovementCriteria, StopConditions, ParamSpec,
)
from agentune.core.state import CampaignState, RoundState
from agentune.datasets import load_dataset
from agentune.runner import RoundRunner


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
def explore_campaign(service):
    """Create a strong-exploration campaign with patience=2."""
    config = CampaignConfig(
        metric_name="accuracy",
        objective_direction="maximize",
        backend="xgboost",
        sampler_config={"name": "TPESampler", "seed": 42},
        initial_search_space=[
            ParamSpec(name="max_depth", type="int", low=1, high=10),
            ParamSpec(name="learning_rate", type="float", low=0.01, high=0.5, log=True),
        ],
        improvement_criteria=ImprovementCriteria(mode="strict_better"),
        stop_conditions=StopConditions(max_rounds=20, patience_rounds=2),
        trials_per_round=5,
        dataset="breast_cancer",
        mode="strong-exploration",
    )
    return service.create_campaign("test-reset", config)


class TestExplorationReset:
    def test_patience_triggers_reset_not_completion(self, db, service, explore_campaign):
        """In strong-exploration mode, patience should auto-reset instead of completing."""
        split, _ = load_dataset("breast_cancer", seed=42)
        runner = RoundRunner(db, split)

        # Run rounds until patience would trigger (2 rounds without improvement)
        # Round 1
        result = runner.run_next_round(explore_campaign["id"])
        assert result.status == "AWAITING_AGENT"

        # Continue without changes for round 2
        from agentune.core.models import ActionProposal
        r1 = service.get_rounds(explore_campaign["id"])[-1]
        service.submit_proposal(explore_campaign["id"], ActionProposal(
            action="continue", justification="keep going", reference_round_ids=[r1["id"]],
        ))
        result = runner.run_next_round(explore_campaign["id"])
        assert result.status == "AWAITING_AGENT"

        # Continue for round 3 — patience should trigger (2 rounds, small budget likely no improvement)
        r2 = service.get_rounds(explore_campaign["id"])[-1]
        service.submit_proposal(explore_campaign["id"], ActionProposal(
            action="continue", justification="keep going", reference_round_ids=[r2["id"]],
        ))
        result = runner.run_next_round(explore_campaign["id"])

        # Should NOT be COMPLETED — should be AWAITING_AGENT with a new round (reset)
        # OR still AWAITING_AGENT (if score improved, patience didn't trigger)
        # The key test: campaign is NOT completed
        campaign = service.get_campaign(explore_campaign["id"])
        assert campaign["state"] != "COMPLETED"

    def test_reset_creates_round_with_incremented_reset_number(self, db, service, explore_campaign):
        """After a reset, the new round should have reset_number incremented."""
        split, _ = load_dataset("breast_cancer", seed=42)
        runner = RoundRunner(db, split)

        # Run enough rounds to trigger patience
        for i in range(3):
            result = runner.run_next_round(explore_campaign["id"])
            if result.status == "COMPLETED":
                break
            if result.status == "AWAITING_AGENT":
                rounds = service.get_rounds(explore_campaign["id"])
                latest = rounds[-1]
                # If reset happened, new round will have reset_number > 0
                if latest.get("reset_number", 0) > 0:
                    assert True
                    return
                # Otherwise, continue
                service.submit_proposal(explore_campaign["id"], ActionProposal(
                    action="continue", justification="keep going",
                    reference_round_ids=[latest["id"]],
                ))

        # If we reach here, check that campaign is still running (reset happened)
        campaign = service.get_campaign(explore_campaign["id"])
        # Campaign should not be COMPLETED if strong-exploration with resets
        assert campaign["state"] in ("RUNNING",)

    def test_standard_mode_still_completes_on_patience(self, db, service):
        """Standard mode should still terminate on patience."""
        config = CampaignConfig(
            metric_name="accuracy",
            objective_direction="maximize",
            backend="xgboost",
            sampler_config={"name": "TPESampler", "seed": 42},
            initial_search_space=[
                ParamSpec(name="max_depth", type="int", low=1, high=10),
                ParamSpec(name="learning_rate", type="float", low=0.01, high=0.5, log=True),
            ],
            improvement_criteria=ImprovementCriteria(mode="strict_better"),
            stop_conditions=StopConditions(max_rounds=20, patience_rounds=2),
            trials_per_round=5,
            dataset="breast_cancer",
            mode="standard",
        )
        campaign = service.create_campaign("test-standard-patience", config)
        split, _ = load_dataset("breast_cancer", seed=42)
        runner = RoundRunner(db, split)

        from agentune.core.models import ActionProposal

        for i in range(10):
            result = runner.run_next_round(campaign["id"])
            if result.status == "COMPLETED":
                assert result.stop_reason == "patience"
                return
            rounds = service.get_rounds(campaign["id"])
            latest = rounds[-1]
            service.submit_proposal(campaign["id"], ActionProposal(
                action="continue", justification="keep going",
                reference_round_ids=[latest["id"]],
            ))

        # Should have completed via patience
        c = service.get_campaign(campaign["id"])
        assert c["state"] == "COMPLETED"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner_reset.py -v`
Expected: FAIL — patience still completes the campaign in strong-exploration mode

- [ ] **Step 3: Implement exploration reset in the runner**

In `src/agentune/runner.py`, modify the patience check section (lines 363-368). Replace:

```python
        # Patience check
        all_summaries = self._round_summaries(rounds)
        all_summaries.append(summary)

        if Scheduler.check_patience(all_summaries, improvement, campaign["objective_direction"], stop_cond.patience_rounds):
            return self._complete_after_summary(campaign_id, current_round["id"], round_number, "patience")
```

With:

```python
        # Patience check — scoped by reset_number
        current_reset = current_round.get("reset_number", 0)
        reset_rounds = [r for r in rounds if r.get("reset_number", 0) == current_reset]
        all_summaries = self._round_summaries(reset_rounds)
        all_summaries.append(summary)

        if Scheduler.check_patience(all_summaries, improvement, campaign["objective_direction"], stop_cond.patience_rounds):
            if campaign.get("mode") == "strong-exploration":
                # Auto-reset: select new params, create new round, continue
                return self._exploration_reset(
                    campaign_id, campaign, current_round, round_number, current_reset,
                )
            return self._complete_after_summary(campaign_id, current_round["id"], round_number, "patience")
```

Then add the `_exploration_reset` method to `RoundRunner`:

```python
    def _exploration_reset(
        self,
        campaign_id: int,
        campaign: dict,
        current_round: dict,
        round_number: int,
        current_reset: int,
    ) -> RunResult:
        """Auto-reset: pick new params, create a new round, return AWAITING_AGENT."""
        from agentune.exploration import select_exploration_params

        backend_cls = get_backend(campaign["backend"])
        backend = backend_cls()
        rounds = self._service.get_rounds(campaign_id)

        new_params = select_exploration_params(backend, rounds)
        new_search_space = [p.to_dict() for p in new_params]
        new_reset = current_reset + 1
        new_round_number = len(rounds) + 1
        new_study_name = f"{campaign['name']}_round_{new_round_number}"

        # Close current round
        self._service.transition_round(current_round["id"], RoundState.AWAITING_AGENT)
        self._service.transition_round(current_round["id"], RoundState.RESOLVED)

        # Create new round with incremented reset_number
        import json
        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO study_rounds "
                "(campaign_id, round_number, optuna_study_name, search_space, budget, "
                "trial_offset, parent_round_id, reset_number) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    campaign_id, new_round_number, new_study_name,
                    json.dumps(new_search_space), campaign["trials_per_round"],
                    0, current_round["id"], new_reset,
                ),
            )

        report_path = self._auto_generate_report(campaign_id)
        return RunResult("AWAITING_AGENT", round_number, report_path=report_path)
```

Also add `import json` at the top of `runner.py` if not already there (it is — line 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner_reset.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `uv run pytest tests/ -v`
Expected: All PASS (except pre-existing `test_campaign_requires_dataset` if still present)

- [ ] **Step 6: Commit**

```bash
git add src/agentune/runner.py tests/test_runner_reset.py
git commit -m "feat: auto-reset on patience in strong-exploration mode"
```

---

### Task 4: Update CLAUDE.md and README with exploration reset docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`, update the `### Strong-exploration mode` section to mention the auto-reset:

After the existing bullet points, add:

```markdown
- **Auto-reset on patience** — when patience triggers, the system automatically selects a new param subset from the full catalog (coverage-based, prioritizing untried params) and creates a new round. The agent can override via `revise_search` or accept the auto-selected params.
```

Also update the `### When to stop` section to add:

```markdown
- In strong-exploration mode, patience triggers a reset (not a stop). Only hard stops (`max_wall_time`, `max_total_trials`, `target_score`, `max_rounds`) terminate the campaign.
```

- [ ] **Step 2: Update README**

In `README.md`, update the `### Strong-Exploration Mode` section. After the existing bullets about guardrails, add:

```markdown
- **Auto-reset on plateau** -- when the agent runs out of improvements, the system automatically picks a new param subset from the full catalog and keeps going. Only the wall-time cap stops the campaign.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add exploration reset documentation"
```

---

### Task 5: Full test suite and smoke test

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run a quick smoke test**

```bash
export AGENTUNE_DB_URL="postgresql://agentune:agentune@localhost:5432/agentune"
uv run agentune init reset-smoke --backend xgboost --dataset breast_cancer \
  --trials-per-round 5 --max-rounds 20 --patience 2 --mode strong-exploration
```

Then run 5-6 rounds via MCP or CLI and verify:
- Patience triggers a reset (not campaign completion)
- New round has different search space
- `reset_number` increments

- [ ] **Step 3: Commit any fixes**

---

## Self-Review Checklist

1. **Spec coverage:** All spec requirements implemented: `reset_number` column (Task 2), patience scoping (Task 3), coverage-based param selection (Task 1), auto-reset in runner (Task 3), docs (Task 4).

2. **Placeholder scan:** No TBDs, TODOs, or "similar to Task N" references. All code is complete.

3. **Type consistency:** `select_exploration_params` takes `(backend, rounds: list[dict])` consistently. `reset_number` is `int` everywhere. `ParamSpec` used for both input and output.
