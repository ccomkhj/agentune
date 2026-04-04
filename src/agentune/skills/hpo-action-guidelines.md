---
name: hpo-action-guidelines
description: Decision framework for choosing next actions in HPO campaigns
---

# Action Guidelines

## Decision Framework

After each round, choose exactly one action:

### continue
**When:** Score is still improving, search space seems right, no red flags.
**Effect:** Adds more trials to the same Optuna study with the same search space.

### narrow_search
**When:** `param_importance` shows some params dominate, `param_ranges_used` shows the best trials cluster in a subregion, or you want to exploit a promising area.
**Effect:** Creates a new Optuna study with tighter parameter ranges. Use `param_importance` and `param_ranges_used` to decide which params to narrow and by how much.

### widen_search
**When:** `plateau_signal` is true, exploration seems insufficient, or the best params are hitting range boundaries.
**Effect:** Creates a new Optuna study with broader parameter ranges.

### increase_budget
**When:** The round showed progress but may not have converged — you want more trials with the same space.
**Effect:** Adds more trials (higher budget) to the same study.

### stop
**When:** Target metric reached, plateau with no improvement across multiple rounds, or diminishing returns.

## Rules You Must Follow

1. **One structural change per round.** narrow and widen are structural. Don't combine with other structural changes.
2. **Cooldown on reversals.** If you just narrowed, you cannot widen for 2 rounds (and vice versa). This prevents oscillation.
3. **Reference history.** Every justification must cite specific round IDs and their results. "It's not improving" is not enough — cite the scores.
4. **No dataset heuristics.** Your decisions should be based on summary signals, not assumptions about specific datasets.

## Justification Template

> "Based on rounds [X, Y, Z]: best_score improved from A to B (+C) over the last N rounds.
> param_importance shows [param] dominates at D%. The convergence curve shows [pattern].
> Action: [action] because [reasoning]."
