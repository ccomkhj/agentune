# Presentation Improvements Spec

Three targeted improvements to make agentune presentation-ready for PyCon.

## 1. CLI Demo Mode (`--demo` flag)

**File:** `src/agentune/cli.py`

Add a `--demo` flag to `agentune run` that prints structured narration after each round step, making the terminal output followable for a live audience.

**Behavior when `--demo` is active:**

After each round completes, print a formatted block:

```
══════════════════════════════════════════════════════
  Round 2/6 complete
  Score: 0.8420 (+0.0015)
  Signals: max_depth dominates (68.9%), plateau detected
  Agent decided: narrow_search
  Reason: "Tightening lr + n_estimators ranges around best region"
══════════════════════════════════════════════════════
```

**Implementation:**
- Add `--demo` flag to the `run` CLI command
- After `runner.run_next_round()`, if demo mode: fetch the round summary and latest decision, format and print the narration block
- No changes to runner, MCP server, or report — this is purely a CLI output concern
- Use click.echo with ANSI colors (click.style) for terminal formatting

## 2. Decision Log Readability

**File:** `src/agentune/report.py`

Improve the Decision Log section of the HTML report so actions are visually distinguishable at a glance.

**Changes:**

### Action-type color coding
Each action gets a distinct color on its badge and left border:
- `continue` — green (var(--green))
- `narrow_search` / `widen_search` — blue (var(--accent))
- `revise_search` — purple (var(--purple))
- `increase_budget` — orange (var(--orange))
- `stop` — red (var(--red))
- Rejected proposals — red border + muted background (already exists, keep)

### Action-type icons (Unicode, no deps)
Prepend to the action badge:
- `continue` → ▶
- `narrow_search` → ◁ (converging)
- `widen_search` → ▷ (expanding)
- `revise_search` → ↻ (revision)
- `increase_budget` → ▲
- `stop` → ■

### Collapsible justification
Wrap justification text in a `<details>` element so the overview is scannable. The signals/importance section stays always-visible; only the free-text justification collapses.

### Highlight "aha" actions
`revise_search` decisions get a slightly larger card with a subtle purple glow border to draw the eye — this is the key demo moment.

## 3. Rejected-Proposal Visualization

**File:** `src/agentune/report.py`

Add a "Guardrails in Action" subsection within the Decision Log that surfaces rejected proposals prominently.

**Design:**

### Inline rejected-proposal cards (within Decision Log)
Rejected proposals already appear in the decision log. Enhance them:
- Strikethrough on the action name
- Red background tint on the card
- "Guardrail: {rejection_reason}" in a callout box
- "Agent then proposed: {next_accepted_action}" link showing what the agent did instead

### "Guardrails" summary box
Above the Decision Log, add a summary box:
- Count of rejected proposals (e.g., "2 proposals rejected by guardrails")
- If 0 rejected: "All proposals accepted — no guardrail interventions"
- Each rejected proposal as a one-liner: "Round 2: narrow_search blocked — cooldown violation"

This gives the presenter a quick reference point to say "the system said no here" without scrolling through the full log.

**Data source:** All data already exists in `agent_decisions` table (`accepted`, `rejection_reason` fields). The `generate_report` function already loads decisions with these fields. No backend changes needed.
