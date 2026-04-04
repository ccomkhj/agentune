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

Claude reads `CLAUDE.md`, discovers the MCP tools via `.mcp.json`, and drives the full campaign autonomously:

```mermaid
sequenceDiagram
    participant You
    participant Claude as Claude Code
    participant HPO as agentune

    You->>Claude: "Run an HPO campaign on california_housing"

    Claude->>HPO: agentune init (CLI)
    HPO-->>Claude: Campaign created

    loop Autonomous loop
        Claude->>HPO: run_next_round (MCP)
        HPO-->>Claude: Round complete

        Claude->>HPO: get_round_summary (MCP)
        HPO-->>Claude: scores, param importance, convergence

        Note over Claude: Observe → Diagnose → Decide

        Claude->>HPO: submit_action_proposal (MCP)
        HPO-->>Claude: Accepted
    end

    Claude->>HPO: generate_report (MCP)
    Claude->>You: Best RMSE 0.4466. Report saved.
```

No API key needed — Claude Code itself is the agent. The MCP server is registered in `.mcp.json` and auto-approved.

### Or run the scripted demo

```bash
uv run python scripts/run_hard_scenarios.py
```

This runs 3 hard datasets (covertype, credit-g, phoneme) end-to-end with a dynamic rule-based agent, then generates HTML reports in `reports/`.

### Or step by step via CLI

```bash
export AGENTUNE_DB_URL=postgresql://agentune:agentune@localhost:5432/agentune

uv run agentune init my-campaign \
  --backend xgboost --metric rmse --direction minimize \
  --dataset california_housing \
  --trials-per-round 40 --max-rounds 6 --patience 3

uv run agentune run my-campaign --dataset california_housing   # repeat after each decision
uv run agentune decisions my-campaign                          # reasoning history
uv run agentune report my-campaign                             # HTML report
uv run agentune export my-campaign                             # best params as JSON
```

## How It Works

Each round: Optuna runs N trials → Summarizer extracts signals → Agent decides next action → repeat or stop.

| Action | When | Effect |
|---|---|---|
| `continue` | Still improving | More trials in same study |
| `narrow_search` | Dominant param found | New study with tighter ranges |
| `widen_search` | Best params at range boundaries | New study with broader ranges |
| `revise_search` | Plateau + no dominant param | New study with different params from the extended catalog |
| `increase_budget` | Plateau in late trials | More trials per round |
| `stop` | No improvement for N rounds | Campaign ends |

Every decision is persisted with reasoning in Postgres (`agent_decisions.reasoning` JSONB). Query with `uv run agentune decisions <campaign>`.

### Guardrails

- One structural change (narrow/widen) per round
- 2-round cooldown before reversing narrow↔widen
- `revise_search` must add or drop at least one param, max 3 swaps per round
- `revise_search` resets the narrow/widen cooldown
- Every decision must reference specific round IDs
- Agent never sees test metrics (test_score is computed post-hoc for reports)

### Example: California Housing (XGBoost, RMSE, 40 trials/round)

| Round | RMSE | Delta | Decision | Key Reasoning |
|---|---|---|---|---|
| 1 | 0.4605 | — | `narrow_search` | `learning_rate` dominates at 43%, cluster around 0.05-0.07 |
| 2 | 0.4496 | -0.011 | `continue` | Solid improvement; importance shifted to `gamma` (88%) |
| 3 | 0.4496 | 0.000 | `narrow_search` | Plateau; tighten `gamma` to [0, 0.4] |
| 4 | 0.4476 | -0.002 | `continue` | Modest gain; `colsample_bytree` now dominates (50%) |
| 5 | **0.4466** | -0.001 | `stop` | Diminishing returns, plateau signal, gen gap 0.26 |

Best RMSE **0.4466** in 5 rounds (200 trials). The agent progressively narrowed dominant params as they were identified.

## Reports

Generate self-contained HTML reports for any campaign:

```bash
uv run agentune report my-campaign                         # writes my-campaign-report.html
uv run agentune report my-campaign -o custom-path.html     # custom output path
```

Reports include:
- Overview cards (state, metric, best val/test scores, rounds, trials, wall time, termination reason)
- Score progression chart (validation + test)
- Round details table (scores, delta, timing, plateau, param importance, decisions)
- Search space evolution across rounds
- Best hyperparameters
- Full decision log with justifications

## Reference

### CLI commands

| Command | Description |
|---|---|
| `init` | Create campaign (requires `--dataset`) |
| `run` | Execute next round |
| `status` | Campaign state + latest round |
| `report` | Generate HTML report |
| `decisions` | Full reasoning history |
| `history` | Rounds and decisions summary |
| `export` | Best params as JSON |
| `pause / resume / stop` | Lifecycle control |
| `baseline` | Plain Optuna run for comparison |

### MCP tools

| Tool | Purpose |
|---|---|
| `run_next_round` | Execute next round (runs trials, summarizes, checks stops) |
| `get_campaign_status` | Current state, config, active round |
| `get_round_summary` | Scores, param importance, convergence |
| `get_campaign_history` | All rounds + past decisions |
| `submit_action_proposal` | Propose next action |
| `generate_report` | Generate HTML report |
| `list_campaigns` | All campaigns overview |

### Datasets

| Dataset | Task | Metric | Size |
|---|---|---|---|
| `breast_cancer` | Binary classification | accuracy (maximize) | 569 |
| `california_housing` | Regression | RMSE (minimize) | 20,640 |
| `digits` | Multi-class classification | accuracy (maximize) | 1,797 |
| `covertype` | 7-class classification | accuracy (maximize) | 20,000 (subsampled) |
| `credit_g` | Imbalanced binary classification | accuracy (maximize) | 1,000 |
| `phoneme` | Noisy binary classification | accuracy (maximize) | 5,404 |

### XGBoost param catalog

**Default search space** (used at campaign init):
`max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`

**Extended catalog** (available for `revise_search`):
`max_leaves`, `max_bin`, `colsample_bylevel`, `colsample_bynode`, `scale_pos_weight`, `grow_policy`

## Development

```bash
docker compose up -d && uv sync
uv run pytest tests/ -v    # 109 tests
```
