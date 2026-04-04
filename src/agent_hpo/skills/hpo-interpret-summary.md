---
name: hpo-interpret-summary
description: How to read and interpret round summary fields
---

# Interpreting Round Summaries

## Key Fields

| Field | What it tells you |
|---|---|
| `best_score` | Best metric value across ALL rounds (cumulative) |
| `delta_from_prev` | How much the cumulative best improved this round |
| `round_best_score` | Best in THIS round only (None if all trials failed/pruned) |
| `new_best_in_round` | Did this round find a new overall best? |
| `completed_trials` / `round_completed_trials` | How many trials actually produced results |
| `plateau_signal` | No improvement in the last 30% of this round's trials |
| `param_importance` | Which parameters matter most (fANOVA) |
| `generalization_gap` | Average |train_score - val_score| — overfitting signal |
| `failure_rate` | Fraction of errored trials (something is wrong if > 0.1) |
| `pruned_rate` | Fraction of pruned trials (healthy if moderate, concerning if > 0.5) |

## Red Flags

- **failure_rate > 0.1**: Something is broken in the search space or data
- **generalization_gap > 0.1**: Model is overfitting — consider narrowing or regularization params
- **plateau_signal = true AND new_best_in_round = false**: This search space may be exhausted
- **round_completed_trials = 0**: All trials failed or were pruned — urgent issue
- **pruned_rate > 0.5**: Search space may include many bad regions — consider narrowing

## What Null Values Mean

- `best_score = None`: No completed trial in the entire campaign yet
- `round_best_score = None`: No completed trial in this specific round
- `delta_from_prev = None`: First round, or no previous best to compare against
