---
name: run-campaign
description: Use whenever the user asks to run, start, kick off, or trigger an agentune HPO campaign in natural language (e.g. "run HPO with lgbm 40 trials/round", "tune xgboost on housing", "optimize california_housing"). Enforces parameter elicitation, init, and the autonomous round loop.
---

# Running an Agentune Campaign

This skill fires whenever the user wants to run a hyperparameter optimization campaign. It overrides any tendency to guess defaults or skip the loop discipline. Follow it exactly.

## Step 1 — Parse the request

Extract whatever the user *did* specify from their message. Map common phrasings:

| User says | Maps to |
|-----------|---------|
| `lgbm`, `lightgbm` | `--backend lightgbm` |
| `xgb`, `xgboost` | `--backend xgboost` |
| `catboost` | `--backend catboost` |
| `N trials per round`, `N trials/round` | `--trials-per-round N` |
| `M rounds`, `max M rounds` | `--max-rounds M` |
| `housing`, `california housing` | `--dataset california_housing` |
| `aggressive`, `explore widely` | `--mode strong-exploration` |

## Step 2 — Elicit missing required args (do NOT silently default)

`agentune init` requires: `name`, `backend`, `metric`, `direction`, `dataset`, `trials-per-round`, `max-rounds`, `patience`, `mode`.

For **every** missing arg, ask the user via `AskUserQuestion`. **Batch all missing questions into a single tool call** so the user answers them in one round-trip, not one-by-one.

Each question MUST include a one-line explanation of what the parameter controls. Use these templates:

- **`name`** — A short identifier for this campaign. Used in MCP calls and the report path (`reports/<name>-report.html`). Suggest one based on `<backend>-<dataset>` if obvious.
- **`backend`** — Which gradient boosting library to tune. Choices: `xgboost`, `lightgbm`, `catboost`.
- **`dataset`** — Which dataset to optimize on. Choices: `breast_cancer`, `california_housing`, `digits`, `covertype`, `credit_g`, `phoneme`, `store_sales`, `rossmann`.
- **`metric` + `direction`** — The validation score the agent optimizes and whether to minimize or maximize. Common pairings: `rmse`/`min` and `mae`/`min` for regression; `auc`/`max`, `accuracy`/`max`, `logloss`/`min` for classification. Suggest a default based on the dataset (regression vs classification) but still confirm.
- **`trials-per-round`** — How many model trainings the TPE sampler runs before handing back to the agent. Higher = better signal per round but slower. Typical: 30–60.
- **`max-rounds`** — Hard cap on how many agent decisions the campaign will make. Typical: 5–10.
- **`patience`** — How many consecutive rounds without improvement before the campaign stops itself. Typical: 3–5.
- **`mode`** — `standard` (conservative guardrails: 2-round cooldown on narrow↔widen reversals, max 3 param swaps per revise, revise only on plateau) vs `strong-exploration` (no cooldown, unlimited param swaps, revise allowed any round — use when you want aggressive exploration of the full parameter catalog).

If the user provided a value but it's ambiguous (e.g. "use auc" but the dataset is regression), ask to confirm rather than overriding silently.

## Step 3 — Confirm and init

Before running `agentune init`, echo back the resolved config in one block so the user can spot mistakes:

```
Campaign: housing-demo
  backend: lightgbm
  dataset: california_housing
  metric: rmse (minimize)
  trials/round: 40
  max rounds: 6
  patience: 5
  mode: standard
```

Then run:

```bash
uv run agentune init <name> \
  --backend <backend> \
  --metric <metric> --direction <direction> \
  --dataset <dataset> \
  --trials-per-round <N> \
  --max-rounds <M> \
  --patience <P> \
  --mode <mode>
```

## Step 4 — Run the autonomous loop

Follow CLAUDE.md's loop exactly. In summary:

**Setup (once):**
1. Call `mcp__agentune__get_campaign_status` for backend confirmation.
2. Call `mcp__agentune__get_tuning_guide` with the backend — read it before deciding anything.
3. Tell the user once: "Running campaign. Progress visible at `reports/<name>-report.html` — refresh anytime."

**Loop (until terminal):**
1. `mcp__agentune__run_next_round`
2. If `status: COMPLETED` → print final result + report path. Done.
3. If `status: FAILED` → print what failed. Done.
4. If `status: AWAITING_AGENT` → continue.
5. `mcp__agentune__get_round_summary`
6. **Diagnose in this order** (cite specific numbers in the proposal justification):
   - generalization gap (train vs val)
   - param importance (any param > 30%? all < 15%?)
   - plateau signal
   - boundary hits
7. `mcp__agentune__submit_action_proposal` with the chosen action.
8. Goto 1.

## Step 5 — Output discipline

- Per round, print **one line only**: `Round N/max: metric=score (delta)`.
- Do NOT narrate reasoning between rounds — the HTML report and the decision log capture it.
- Elaborate only at: final result, failure, or rejected proposal.
- If a proposal is rejected, read the rejection reason, adapt, and resubmit — don't re-explain to the user.

## Red flags — STOP and re-read this skill

| Thought | Reality |
|---------|---------|
| "I'll just default `--patience` to 5" | No. Ask. |
| "The user said lgbm, obvious dataset is housing" | No. Ask. |
| "I'll narrate what the agent decided this round" | No. One status line. |
| "I'll skip `get_tuning_guide`, I remember xgboost" | No. Always read it at the start of a campaign. |
| "I'll batch ask the user round-by-round" | No. Loop is autonomous; only stop on terminal state. |
