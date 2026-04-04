# agent-hpo

Agent-driven hyperparameter optimization with Optuna. Claude Code acts as the LLM agent — reading round summaries via MCP tools, reasoning about the results, and proposing search space changes — while Optuna runs the optimization deterministically within each round.

## Install & Run

```bash
docker compose up -d   # start Postgres
uv sync                # install
```

### Use with Claude Code (recommended)

Open Claude Code in this directory and ask:

> "Create an HPO campaign for california_housing and optimize it"

Claude reads `CLAUDE.md`, discovers the MCP tools via `.mcp.json`, and drives the full campaign:

```
You                          Claude Code                      agent-hpo
 │                               │                               │
 │  "optimize california_housing"│                               │
 │──────────────────────────────>│                               │
 │                               │  uv run agent-hpo init ...   │
 │                               │──────────────────────────────>│  create campaign
 │                               │  uv run agent-hpo run ...    │
 │                               │──────────────────────────────>│  run 40 trials
 │                               │                               │
 │                               │  get_round_summary (MCP)     │
 │                               │<──────────────────────────────│  RMSE=0.46, learning_rate 44%
 │                               │                               │
 │                               │  (reasons about the data)    │
 │                               │                               │
 │                               │  submit_action_proposal (MCP)│
 │                               │──────────────────────────────>│  narrow_search: focus lr, depth
 │                               │                               │
 │                               │  uv run agent-hpo run ...    │
 │                               │──────────────────────────────>│  run 40 more trials
 │                               │          ... repeats ...      │
 │                               │                               │
 │  "Round 3: RMSE improved to  │                               │
 │   0.448. Stopping — patience" │                               │
 │<──────────────────────────────│                               │
```

No API key needed — Claude Code itself is the agent. The MCP server is registered in `.mcp.json` and auto-approved.

### Or run the scripted demo

```bash
uv run python demo.py
```

Runs a California Housing campaign end-to-end with the rule-based `AgentReasoner`, printing the full observe → diagnose → decide chain for each round.

### Or step by step via CLI

```bash
export AGENT_HPO_DB_URL=postgresql://agent_hpo:agent_hpo@localhost:5432/agent_hpo

# Create campaign
uv run agent-hpo init my-campaign \
  --backend xgboost --metric rmse --direction minimize \
  --trials-per-round 40 --max-rounds 6 --patience 3

# Run rounds (repeat after each agent decision)
uv run agent-hpo run my-campaign --dataset california_housing

# Check results
uv run agent-hpo status my-campaign
uv run agent-hpo decisions my-campaign   # full reasoning history from Postgres
uv run agent-hpo export my-campaign      # best params as JSON

# Compare against plain Optuna
uv run agent-hpo baseline my-baseline --dataset california_housing --total-trials 120
```

## How It Works

```
  Optuna runs N trials   →   Summarizer extracts signals   →   Agent decides next action
  (deterministic)              (immutable report)               (via MCP or AgentReasoner)
```

### Agent actions

| Action | When | Effect |
|---|---|---|
| `narrow_search` | Dominant param found, can focus | New Optuna study with tighter ranges |
| `continue` | Still improving | More trials in same study |
| `increase_budget` | Improving but plateau in late trials | More trials per round |
| `widen_search` | Best params at range boundaries | New study with broader ranges |
| `stop` | No improvement for N rounds | Campaign ends |

### Agent reasoning (stored in Postgres)

Every decision is persisted with structured reasoning in the `agent_decisions.reasoning` JSONB column:

```
OBSERVED:                           DIAGNOSIS:                        ACTION:
  rmse: 0.4605                        - Mild overfitting (gap=0.086)    narrow_search
  learning_rate: 44% importance       - Dominant: learning_rate (44%)
  plateau: false                      - Boundary: reg_alpha, reg_lambda
  gen gap: 0.086
                                    SEARCH SPACE CHANGES:
                                      learning_rate  [0.001, 1.0] → [0.001, 0.27]  (focusing tightly)
                                      n_estimators   [50, 500]    → [191, 461]      (moderate narrowing)
                                      max_depth      [1, 15]      → [2, 11]         (moderate narrowing)
```

Query anytime: `uv run agent-hpo decisions <campaign> [--round N]`

### Example: California Housing

| Round | RMSE | Delta | Agent Decision | Why |
|---|---|---|---|---|
| 1 | 0.4605 | — | `narrow_search` | `learning_rate` dominates at 44%, mild overfitting |
| 2 | 0.4514 | -0.009 | `continue` | Significant improvement (2.0%), narrowing worked |
| 3 | 0.4486 | -0.004 | `continue` | Still improving |
| 4 | 0.4480 | -0.001 | `increase_budget` | Small improvement + plateau detected |
| 5 | — | — | (max_total_trials) | Budget exhausted |

**Result**: Agent RMSE **0.448** vs plain Optuna baseline **0.450** (same 120 trial budget).

### Guardrails

- One structural change (narrow/widen) per round
- 2-round cooldown before reversing narrow↔widen
- Every decision must reference specific round IDs
- Sampler frozen at campaign creation
- Agent never sees test metrics

## Architecture

```
┌─────────────────┐  ┌────────────────────┐
│   CLI (Bash)    │  │ MCP Server (stdio) │
│  init, run,     │  │  5 tools for       │
│  status, export │  │  Claude Code       │
└────────┬────────┘  └─────────┬──────────┘
         └───────────┬─────────┘
            ┌────────▼────────┐
            │  Core Service   │  ← validation, state machines, locking, cooldown
            └──┬─────┬─────┬──┘
               ▼     ▼     ▼
          Campaign  Optuna   Agent
           Schema  Storage   Reasoning
         (Postgres)(Postgres)(Postgres)
```

### CLI commands

| Command | Description |
|---|---|
| `init <name>` | Create campaign |
| `run <name>` | Execute next round |
| `status <name>` | Campaign state + latest round |
| `decisions <name>` | Full reasoning history from Postgres |
| `history <name>` | Rounds and decisions summary |
| `export <name>` | Best params as JSON |
| `pause / resume / stop` | Campaign lifecycle control |
| `baseline <name>` | Plain Optuna run for comparison |

### MCP tools (Claude Code agent interface)

Registered in `.mcp.json`, auto-discovered by Claude Code:

| Tool | Purpose |
|---|---|
| `get_campaign_status` | Current state, config, active round |
| `get_round_summary` | Scores, param importance, convergence, generalization gap |
| `get_campaign_history` | All rounds + past decisions |
| `submit_action_proposal` | Propose next action (validated by core before applying) |
| `list_campaigns` | All campaigns overview |

## Development

```bash
docker compose up -d
uv sync
uv run pytest tests/ -v     # 92 tests
```

### Datasets

| Dataset | Task | Metric |
|---|---|---|
| `breast_cancer` | Binary classification | accuracy (maximize) |
| `california_housing` | Regression | RMSE (minimize) |
| `digits` | Multi-class classification | accuracy (maximize) |
