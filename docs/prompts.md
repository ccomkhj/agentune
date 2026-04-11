# Sample Prompts for Claude Code

## Quick Campaign (5-10 minutes)

> Run an HPO campaign on breast_cancer with xgboost, 40 trials per round, 6 rounds max

## Time-Bounded Exploration (hours)

> Run an HPO campaign on covertype with xgboost, 40 trials per round, max wall time 1 hour, using strong-exploration mode. Keep exploring until time runs out and report the best parameters.

```bash
uv run agentune init my-campaign --backend xgboost --dataset covertype \
  --trials-per-round 40 --max-wall-time 3600 --mode strong-exploration
```

## Overnight Run (24 hours)

> Create a campaign on california_housing with lightgbm, 50 trials per round, 24-hour wall time, strong-exploration mode. Run it -- keep searching different parameter strategies until the clock runs out.

```bash
uv run agentune init overnight --backend lightgbm --dataset california_housing \
  --trials-per-round 50 --max-wall-time 86400 --mode strong-exploration
```

## Compare Backends

> Run three campaigns on credit_g -- one with xgboost, one with lightgbm, one with catboost. Use 40 trials per round, 6 rounds each. Compare the results.

## Custom Dataset

> Run an HPO campaign on my data at ./data/sales.csv with xgboost. The target column is "revenue", metric is rmse, direction minimize. Use 30 trials per round.

```bash
uv run agentune init sales-opt --backend xgboost --dataset ./data/sales.csv \
  --target revenue --metric rmse --direction minimize --trials-per-round 30
```

## Resume and Inspect

> What's the status of campaign my-campaign? Show me the decision history.

> Resume campaign my-campaign and run 3 more rounds.
