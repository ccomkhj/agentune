<p align="center">
  <img src="assets/logo.png" alt="agentune logo" width="300">
</p>

<h1 align="center">agentune</h1>

<p align="center">
  <a href="https://github.com/huijokim/agentune/actions/workflows/ci.yml"><img src="https://github.com/huijokim/agentune/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

Agent-driven hyperparameter optimization with Optuna. Claude Code acts as the LLM agent -- reading round summaries via MCP tools, diagnosing signals like parameter importance and convergence plateaus, and proposing search space changes -- while Optuna runs the optimization deterministically within each round.

Unlike traditional HPO that runs a fixed algorithm, agentune lets an LLM agent adapt the search strategy itself: narrowing around promising regions, widening when hitting boundaries, revising the parameter set entirely, or stopping early when gains plateau.

## Quick Start

```bash
git clone https://github.com/huijokim/agentune.git
cd agentune
docker compose up -d   # starts Postgres + MLflow
uv sync                # install dependencies
cp .mcp.json.example .mcp.json  # MCP server config for Claude Code
```

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

        Note over Claude: Observe / Diagnose / Decide

        Claude->>HPO: submit_action_proposal (MCP)
        HPO-->>Claude: Accepted
    end

    Claude->>You: Best RMSE 0.4466. Report at reports/my-campaign-report.html
```

No API key needed -- Claude Code itself is the agent. The MCP server is registered in `.mcp.json` and auto-approved.

## Architecture

```mermaid
flowchart TB
    Agent["Claude Code<br/>(LLM Agent)"]

    subgraph MCP["MCP Server (8 tools)"]
        run_next["run_next_round"]
        get_summary["get_round_summary"]
        submit["submit_action_proposal"]
        tuning_guide["get_tuning_guide"]
        other_tools["list / status / history / report"]
    end

    subgraph Core["Core"]
        CampaignService["Campaign Service"]
        Scheduler["Scheduler"]
        StateMachine["FSM<br/>(campaign + round)"]
    end

    subgraph Runner["RoundRunner"]
        Execute["Execute trials"]
        Summarize["Extract signals"]
        Report["Auto-generate HTML report"]
    end

    subgraph Backends["Backends"]
        XGB["XGBoost<br/>9 default / 19 total params"]
        LGB["LightGBM<br/>10 default / 17 total params"]
        CB["CatBoost<br/>7 default / 14 total params"]
    end

    Optuna["Optuna (TPE)"]
    Postgres[("PostgreSQL")]
    MLflow["MLflow<br/>(optional)"]

    Agent <-->|MCP tools| MCP
    MCP --> Runner
    MCP --> Core
    Runner --> Optuna
    Runner --> Backends
    Runner -.-> MLflow
    Core --> Postgres
    Optuna --> Postgres
```

Each round: Optuna runs N trials with the backend's objective function -> Summarizer extracts signals (param importance, generalization gap, plateau detection) -> Agent diagnoses and decides the next action -> repeat or stop.

## Agent Actions

| Action | When | Effect |
|---|---|---|
| `continue` | Still improving | More trials in same Optuna study |
| `narrow_search` | Dominant param found | New study with tighter ranges |
| `widen_search` | Best params at range boundaries | New study with broader ranges |
| `revise_search` | Plateau + no dominant param | New study with different params from the extended catalog |
| `increase_budget` | Plateau in late trials | More trials per round |
| `stop` | No improvement for N rounds | Campaign ends |

### Guardrails

- One structural change (narrow/widen) per round
- 2-round cooldown before reversing narrow <-> widen
- `revise_search` must add or drop at least one param, max 3 swaps
- Agent never sees test metrics during active campaigns (stripped from MCP responses; only in final report)
- Every decision must reference specific round IDs

### Strong-Exploration Mode

When you have a time budget and want the agent to keep searching and exploring until the clock runs out, use `--mode strong-exploration` with `--max-wall-time`:

```bash
# Explore for 24 hours, report the best params at the end
uv run agentune init my-campaign --backend xgboost --dataset covertype \
  --trials-per-round 40 --max-wall-time 86400 --mode strong-exploration
```

Then ask Claude Code:

> "Run campaign my-campaign -- keep exploring until time runs out and report the best parameters"

This relaxes three guardrails so the agent can freely explore the full parameter catalog:
- `revise_search` allowed **every round**, even when improving -- no plateau signal required
- **No churn limit** -- the agent can swap the entire parameter set in one round
- **No cooldown** between narrow/widen reversals

In standard mode (default), the agent is conservative: it only revises parameters when it detects a plateau. Strong-exploration mode removes those constraints, letting the agent radically swap parameter sets each round for the duration of the time budget.

## Datasets

| Dataset | Task | Size | Metric | Direction |
|---|---|---|---|---|
| `breast_cancer` | Binary classification | 569 | accuracy | maximize |
| `california_housing` | Regression | 20,640 | rmse | minimize |
| `digits` | Multi-class (10 classes) | 1,797 | accuracy | maximize |
| `covertype` | Multi-class (7 classes) | 20,000 | accuracy | maximize |
| `credit_g` | Imbalanced binary | 1,000 | accuracy | maximize |
| `phoneme` | Noisy binary | 5,404 | accuracy | maximize |
| `store_sales` | Time-series regression | ~8,400 | rmse | minimize |
| `rossmann` | Time-series regression | ~4,700 | rmse | minimize |

Time-series datasets (`store_sales`, `rossmann`) use [mlforecast](https://github.com/Nixtla/mlforecast) for feature engineering (lags, rolling means, date features) with temporal train/val/test splits and 28-day warm-up periods to prevent data leakage. Non-temporal datasets use random 60/20/20 splits.

## Example: Store Sales (XGBoost, RMSE, 40 trials/round)

| Round | RMSE | Delta | Decision | Key Signal |
|---|---|---|---|---|
| 1 | 205.79 | -- | `continue` | `learning_rate` dominates (52%), no plateau |
| 2 | 205.78 | -0.01 | `narrow_search` | Plateau; tighten lr + n_estimators ranges |
| 3 | 205.42 | -0.36 | `continue` | Importance redistributed, gen gap halved |
| 4 | 205.32 | -0.10 | `continue` | n_estimators hitting upper bound |
| 5 | 205.24 | -0.08 | `continue` | Cooldown prevents widen |
| 6 | **205.24** | 0.00 | (max_rounds) | Converged. Test RMSE: 207.44 |

Best RMSE **205.24** in 6 rounds (240 trials, 39 seconds). The agent narrowed after detecting a plateau in round 2, which was the most impactful structural change.

## MLflow Integration

MLflow tracking is opt-in. When `MLFLOW_TRACKING_URI` is set, every trial and round-level metric is logged:

- Campaign = MLflow experiment
- Round = parent run (best_score, test_score, param importance)
- Trial = nested run (params, value, train metrics)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
```

MLflow UI is at http://localhost:5001 after `docker compose up -d`.

## CLI Usage

```bash
export AGENTUNE_DB_URL=postgresql://agentune:agentune@localhost:5432/agentune

# Create a campaign (metric/direction auto-inferred from dataset)
uv run agentune init my-campaign --backend xgboost --dataset california_housing \
  --trials-per-round 40 --max-rounds 6 --patience 3

# Run rounds manually
uv run agentune run my-campaign --dataset california_housing

# Inspect
uv run agentune status my-campaign
uv run agentune decisions my-campaign     # full reasoning chain
uv run agentune history my-campaign
uv run agentune export my-campaign        # best params as JSON
uv run agentune report my-campaign        # HTML report
```

### All Commands

| Command | Description |
|---|---|
| `init` | Create campaign (`--metric` and `--direction` default from dataset) |
| `run` | Execute next round |
| `status` | Campaign state + latest round |
| `report` | Generate HTML report |
| `decisions` | Full observe/diagnose/decide reasoning history |
| `history` | Rounds and decisions summary |
| `export` | Best params as JSON |
| `pause` / `resume` / `stop` | Lifecycle control |
| `baseline` | Plain Optuna run for comparison |
| `benchmark` | 3-way comparison: all-params vs top-5 vs default-9 |

### MCP Tools

| Tool | Purpose |
|---|---|
| `run_next_round` | Execute next round (trials + summarize + stop checks + auto-report) |
| `get_campaign_status` | Current state, config, active round |
| `get_round_summary` | Scores, param importance, convergence signals |
| `get_campaign_history` | All rounds + past decisions |
| `submit_action_proposal` | Propose next action with justification |
| `get_tuning_guide` | Backend-specific param knowledge and diagnostic patterns |
| `generate_report` | Generate HTML report |
| `list_campaigns` | All campaigns overview |

## Reports

Self-contained HTML reports auto-generated after every round:

- Status banner and progress timeline
- Score progression chart (validation + test)
- Round details table (scores, delta, plateau, param importance, generalization gap)
- Search space evolution across rounds
- Best hyperparameters
- Decision log with full agent reasoning context

```bash
uv run agentune report my-campaign -o custom-path.html
```

## Development

```bash
docker compose up -d
uv sync
uv run pytest tests/ -v
```

## License

MIT
