# agentune

Agent-driven hyperparameter optimization with Optuna. Claude Code acts as the LLM agent — reading round summaries via MCP tools, reasoning about the results, and proposing search space changes — while Optuna runs the optimization deterministically within each round.

## Install & Run

```bash
docker compose up -d   # start Postgres
uv sync                # install
```

### Use with Claude Code (recommended)

Open Claude Code in this directory and ask:

> "Run an HPO campaign on california_housing with 40 trials per round"

Claude reads `CLAUDE.md`, discovers the MCP tools via `.mcp.json`, and drives the full campaign:

```mermaid
sequenceDiagram
    participant You
    participant Claude as Claude Code
    participant HPO as agentune

    You->>Claude: "Run an HPO campaign on california_housing"

    Claude->>HPO: agentune init (CLI)
    HPO-->>Claude: Campaign created

    loop Each round
        Claude->>HPO: agentune run (CLI)
        HPO-->>Claude: Round complete

        Claude->>HPO: get_round_summary (MCP)
        HPO-->>Claude: scores, param importance, convergence

        Note over Claude: Observe → Diagnose → Decide

        Claude->>HPO: submit_action_proposal (MCP)
        HPO-->>Claude: Accepted
    end

    Claude->>You: Best RMSE 0.4466. Stopping.
```

No API key needed — Claude Code itself is the agent. The MCP server is registered in `.mcp.json` and auto-approved.

### Or run the scripted demo

```bash
uv run python demo.py
```

### Or step by step via CLI

```bash
export AGENTUNE_DB_URL=postgresql://agentune:agentune@localhost:5432/agentune

uv run agentune init my-campaign \
  --backend xgboost --metric rmse --direction minimize \
  --trials-per-round 40 --max-rounds 6 --patience 3

uv run agentune run my-campaign --dataset california_housing   # repeat after each decision
uv run agentune decisions my-campaign                          # reasoning history
uv run agentune export my-campaign                             # best params as JSON
```

## How It Works

Each round: Optuna runs N trials → Summarizer extracts signals → Agent decides next action → repeat or stop.

| Action | When | Effect |
|---|---|---|
| `continue` | Still improving | More trials in same study |
| `narrow_search` | Dominant param found | New study with tighter ranges |
| `widen_search` | Best params at range boundaries | New study with broader ranges |
| `increase_budget` | Plateau in late trials | More trials per round |
| `stop` | No improvement for N rounds | Campaign ends |

Every decision is persisted with reasoning in Postgres (`agent_decisions.reasoning` JSONB). Query with `uv run agentune decisions <campaign>`.

### Example: California Housing (XGBoost, RMSE, 40 trials/round)

| Round | RMSE | Delta | Decision | Key Reasoning |
|---|---|---|---|---|
| 1 | 0.4605 | — | `narrow_search` | `learning_rate` dominates at 43%, cluster around 0.05-0.07 |
| 2 | 0.4496 | -0.011 | `continue` | Solid improvement; importance shifted to `gamma` (88%) |
| 3 | 0.4496 | 0.000 | `narrow_search` | Plateau; tighten `gamma` to [0, 0.4] |
| 4 | 0.4476 | -0.002 | `continue` | Modest gain; `colsample_bytree` now dominates (50%) |
| 5 | **0.4466** | -0.001 | `stop` | Diminishing returns, plateau signal, gen gap 0.26 |

Best RMSE **0.4466** in 5 rounds (200 trials). The agent progressively narrowed dominant params as they were identified.

### Guardrails

- One structural change (narrow/widen) per round
- 2-round cooldown before reversing narrow↔widen
- Every decision must reference specific round IDs
- Agent never sees test metrics

## Reference

### CLI commands

| Command | Description |
|---|---|
| `init` | Create campaign |
| `run` | Execute next round |
| `status` | Campaign state + latest round |
| `decisions` | Full reasoning history |
| `history` | Rounds and decisions summary |
| `export` | Best params as JSON |
| `pause / resume / stop` | Lifecycle control |
| `baseline` | Plain Optuna run for comparison |

### MCP tools

| Tool | Purpose |
|---|---|
| `get_campaign_status` | Current state, config, active round |
| `get_round_summary` | Scores, param importance, convergence |
| `get_campaign_history` | All rounds + past decisions |
| `submit_action_proposal` | Propose next action |
| `list_campaigns` | All campaigns overview |

### Datasets

| Dataset | Task | Metric |
|---|---|---|
| `breast_cancer` | Binary classification | accuracy (maximize) |
| `california_housing` | Regression | RMSE (minimize) |
| `digits` | Multi-class classification | accuracy (maximize) |

## Development

```bash
docker compose up -d && uv sync
uv run pytest tests/ -v
```
