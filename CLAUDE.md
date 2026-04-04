# Agent-HPO: Claude Code as the optimization agent

This project uses Claude Code as the LLM agent that drives hyperparameter optimization campaigns. You have MCP tools to read campaign state and propose actions.

## Setup

Postgres must be running: `docker compose up -d`

## Your MCP tools

You have these tools via the `agent-hpo` MCP server:

- `mcp__agent-hpo__list_campaigns` — see all campaigns
- `mcp__agent-hpo__get_campaign_status` — current state, config, latest round
- `mcp__agent-hpo__get_round_summary` — round summary (scores, param importance, convergence)
- `mcp__agent-hpo__get_campaign_history` — all rounds + past decisions
- `mcp__agent-hpo__submit_action_proposal` — propose your next action

## How to run a campaign

### 1. Create a campaign (CLI)

```bash
uv run agent-hpo init <name> --backend xgboost --metric <metric> --direction <min/max> --trials-per-round 40 --max-rounds 6 --patience 3
```

### 2. Run a round (CLI)

```bash
uv run agent-hpo run <name> --dataset <dataset>
```

Available datasets: `breast_cancer`, `california_housing`, `digits`

### 3. Read the summary (MCP)

After a round completes with `AWAITING_AGENT`, use `get_round_summary` to read results.

### 4. Decide and propose (MCP)

Use `submit_action_proposal` with:

```json
{
  "campaign_name": "<name>",
  "proposal": {
    "action": "continue | narrow_search | widen_search | increase_budget | stop",
    "justification": "cite specific round data: scores, param importance, deltas",
    "reference_round_ids": [1, 2],
    "proposed_search_space": [],
    "proposed_budget": null
  }
}
```

### 5. Repeat from step 2

## Decision framework

### When to narrow_search
- A parameter dominates importance (>30%)
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

### When to stop
- No improvement for 2+ consecutive rounds
- High failure rate (>10%)
- Target score reached

## Rules

1. One structural change (narrow/widen) per round
2. 2-round cooldown before reversing narrow↔widen
3. Every justification must cite specific round IDs and scores
4. Never assume — read the summary data first
5. `proposed_search_space` must only use params from the backend (for xgboost: max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda)

## XGBoost param specs format

Each param in `proposed_search_space` is a dict:
```json
{"name": "learning_rate", "type": "float", "low": 0.01, "high": 0.5, "log": true}
{"name": "max_depth", "type": "int", "low": 1, "high": 10}
```

Types: `float`, `int`, `categorical`. Float/int require `low` and `high`. Optional: `log` (bool), `choices` (for categorical).
