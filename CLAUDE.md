# Agentune: Claude Code as the optimization agent

This project uses Claude Code as the LLM agent that drives hyperparameter optimization campaigns. You have MCP tools to read campaign state and propose actions.

## Setup

Postgres and MLflow must be running: `docker compose up -d`

MLflow UI is at http://localhost:5001. To enable MLflow tracking, set `MLFLOW_TRACKING_URI=http://localhost:5001`.

## Your MCP tools

You have these tools via the `agentune` MCP server:

- `mcp__agentune__list_campaigns` — see all campaigns
- `mcp__agentune__get_campaign_status` — current state, config, latest round
- `mcp__agentune__get_round_summary` — round summary (scores, param importance, convergence)
- `mcp__agentune__get_campaign_history` — all rounds + past decisions
- `mcp__agentune__submit_action_proposal` — propose your next action
- `mcp__agentune__run_next_round` — execute the next PROPOSED round (runs trials, generates summary, checks stops)
- `mcp__agentune__get_tuning_guide` — backend-specific tuning knowledge (param roles, interactions, diagnostic patterns)
- `mcp__agentune__generate_report` — generate an HTML report for a campaign

## How to run a campaign

### 1. Create a campaign (CLI)

```bash
uv run agentune init <name> --backend <backend> --metric <metric> --direction <min/max> --dataset <dataset> --trials-per-round 40 --max-rounds 6 --patience 5
```

Available backends: `xgboost`, `lightgbm`, `catboost`
Available datasets: `breast_cancer`, `california_housing`, `digits`, `covertype`, `credit_g`, `phoneme`, `store_sales`, `rossmann`

Available modes:
- `standard` (default): Conservative guardrails — 2-round cooldown, max 3 param swaps per revise, revise only when plateau/no improvement
- `strong-exploration`: Relaxed guardrails — no cooldown, unlimited param swaps, revise allowed anytime. Use when you want aggressive exploration of the full parameter catalog.

### 2. Run the campaign (single trigger)

When the user says "run campaign X" or "optimize X", run the **entire campaign autonomously** using this loop. Do not wait for user input between rounds.

**Setup (once):**
1. Call `get_campaign_status` to get the backend name and current state
2. Call `get_tuning_guide` with the backend — read it to understand param roles, interactions, and diagnostic patterns
3. Tell the user: "Running campaign. Progress visible at `reports/<name>-report.html` — refresh anytime."

**Loop (repeat until terminal):**
1. Call `run_next_round` — this executes trials, summarizes, checks stops, **and auto-updates the HTML report**
2. If `status: "COMPLETED"` → campaign hit a hard stop. Tell the user the final result and report path. Done.
3. If `status: "FAILED"` → tell the user what failed. Done.
4. If `status: "AWAITING_AGENT"` → continue to step 5
5. Call `get_round_summary` to read results
6. **Diagnose** using the tuning guide: match the summary signals (param importance, plateau, generalization gap) to the guide's diagnostic patterns
7. Call `submit_action_proposal` with your decision
8. Go to step 1

**Important:** Keep output minimal. After each round, print **one status line**: `Round N/max: metric=score (delta)`. Don't narrate your reasoning — the HTML report and decision log capture everything. Only elaborate at the end (final result) or if something unexpected happens (failure, rejection).

The `run_next_round` response includes a `report_path` field — this is the auto-generated HTML report updated after every round.

## Decision framework

### How to diagnose

Before deciding, match the round summary signals to the tuning guide's diagnostic patterns:

1. **Check generalization gap** — large gap (train >> val) means overfitting. Look at which params the guide says control regularization.
2. **Check param importance** — if a param dominates, the guide tells you what it controls and what interacts with it.
3. **Check plateau signal** — the guide has specific patterns for "plateau + dominant param" vs "plateau + no dominant param".
4. **Check boundary hits** — if best params are near range boundaries, the guide tells you whether to widen or whether the param is naturally bounded.

### When to narrow_search
- A parameter dominates importance (>30%) — check the tuning guide for what it controls
- Best params cluster in a subregion of the search space
- Plateau detected but score could still improve with focused exploration
- Provide `proposed_search_space`: tighten ranges around best values, more aggressively for high-importance params

### When to continue
- Score improved this round
- No plateau — optimization still making progress

### When to increase_budget
- Improving but plateau in late trials — more data might help TPE
- Provide `proposed_budget` (higher than current)

### When to widen_search
- Best params hitting range boundaries
- Search space too narrow, missing promising regions
- Provide `proposed_search_space` with broader ranges

### When to revise_search
- **Plateau after multiple rounds** with no dominant parameter (all params <15% importance)
- Best params hitting range boundaries on multiple params simultaneously
- The current parameter set is fundamentally wrong — you need different params, not different ranges
- Provide `proposed_search_space` selecting from the backend's full catalog (call `get_tuning_guide` for available params)
- Must add or drop at least one parameter vs the current round's space
- Max 3 parameter swaps (adds + drops) per round — keep changes attributable
- After revise_search, narrow/widen cooldown resets — you can immediately narrow the new space

**Strategy for revise_search:**
1. Look at `param_importance` — drop params with <5% importance across multiple rounds
2. Consult the tuning guide's diagnostics — add params that address the diagnosis (e.g., regularization params if overfitting, tree structure params if underfitting)
3. Keep at least 2-3 params from the previous space for continuity

### Strong-exploration mode

In `strong-exploration` mode, guardrails are relaxed:
- `revise_search` is allowed **every round**, even when improving — you don't need plateau/no-improvement signals
- **No churn limit** — you can swap the entire parameter set in one round
- **No cooldown** between narrow↔widen reversals

Use this mode when:
- The default parameter set may not contain the right params for this dataset
- You want the agent to aggressively explore different parameter subsets from the full catalog
- You have enough trial budget (6+ rounds) to absorb exploration cost

The agent should still cite specific signals in justifications and track which param sets worked vs. didn't.

- **Auto-reset on patience** — when patience triggers, the system automatically selects a new param subset from the full catalog (coverage-based, prioritizing untried params) and creates a new round. The agent can override via `revise_search` or accept the auto-selected params.

### When to stop
- No improvement for 2+ consecutive rounds
- High failure rate (>10%)
- Target score reached
- In strong-exploration mode, patience triggers a reset (not a stop). Only hard stops (`max_wall_time`, `max_total_trials`, `target_score`, `max_rounds`) terminate the campaign.

## Rules

1. One structural change (narrow/widen) per round
2. 2-round cooldown before reversing narrow↔widen
3. Every justification must cite specific round IDs and scores
4. Never assume — read the summary data first
5. `proposed_search_space` must only use params from the backend's catalog (call `get_tuning_guide`). `narrow_search`/`widen_search` use current params; `revise_search` can use the full catalog.
6. Before proposing `stop`, consider whether `revise_search` could explore a fundamentally different parameter set. But do not force a `revise_search` when the campaign is clearly exhausted.
7. Always call `get_tuning_guide` at the start of a campaign to understand the backend's param roles and interactions.

## Param spec format

Each param in `proposed_search_space` is a dict:
```json
{"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": true}
{"name": "max_depth", "type": "int", "low": 1, "high": 10}
{"name": "grow_policy", "type": "categorical", "choices": ["depthwise", "lossguide"]}
```

Types: `float`, `int`, `categorical`. Float/int require `low` and `high`. Optional: `log` (bool), `choices` (for categorical).
