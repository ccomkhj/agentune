# Presentation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make agentune presentation-ready for PyCon with three improvements: CLI demo mode, decision log readability, and rejected-proposal visualization.

**Architecture:** All three items modify existing files — no new modules. Item 1 touches `cli.py` (add `--demo` flag to `run` command). Items 2 and 3 touch `report.py` (CSS + HTML changes to decision log section). Tests go in existing `test_report.py` and `test_cli.py`.

**Tech Stack:** Python, Click (CLI), HTML/CSS (report), pytest

---

### Task 1: Decision log action-type color coding and icons

**Files:**
- Modify: `src/agentune/report.py` — add action color/icon mappings + CSS + update `_render_html` decision card builder
- Test: `tests/test_report.py` — verify action-specific CSS classes and icons appear in output

- [ ] **Step 1: Write failing tests for action color coding**

Add to `tests/test_report.py`:

```python
class TestDecisionLogStyling:
    """Test that decision cards have action-specific color classes and icons."""

    def _make_decision(self, action="continue", accepted=True, rejection_reason=None):
        return {
            "round": 1,
            "action": action,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "justification": "Test justification",
            "reasoning": None,
            "proposed_search_space": None,
            "summary": {"best_score": 0.9, "param_importance": {}},
            "prev_search_space": [],
        }

    def test_continue_has_green_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("continue"))
        assert "action-continue" in html
        assert "\u25b6" in html  # ▶

    def test_narrow_search_has_accent_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("narrow_search"))
        assert "action-narrow_search" in html
        assert "\u25c1" in html  # ◁

    def test_revise_search_has_purple_class_and_highlight(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("revise_search"))
        assert "action-revise_search" in html
        assert "decision-highlight" in html
        assert "\u21bb" in html  # ↻

    def test_stop_has_red_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("stop"))
        assert "action-stop" in html
        assert "\u25a0" in html  # ■

    def test_rejected_keeps_rejected_class(self):
        from agentune.report import _build_decision_card
        html = _build_decision_card(self._make_decision("narrow_search", accepted=False, rejection_reason="cooldown"))
        assert "rejected" in html
        assert "cooldown" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py::TestDecisionLogStyling -v`
Expected: FAIL — `_build_decision_card` doesn't exist yet

- [ ] **Step 3: Extract `_build_decision_card` function and add action styling**

In `src/agentune/report.py`, add the action icon/color mapping near the top (after imports):

```python
# Action-type styling for decision log
ACTION_ICONS = {
    "continue": "\u25b6",       # ▶
    "narrow_search": "\u25c1",  # ◁
    "widen_search": "\u25b7",   # ▷
    "revise_search": "\u21bb",  # ↻
    "increase_budget": "\u25b2", # ▲
    "stop": "\u25a0",           # ■
}
```

Extract the inline decision-card HTML from `_render_html` (the loop at ~line 442-468) into a new function `_build_decision_card(d: dict) -> str`:

```python
def _build_decision_card(d: dict) -> str:
    """Build a single decision card with action-specific styling."""
    action = d["action"]
    icon = ACTION_ICONS.get(action, "")
    status_class = "accepted" if d["accepted"] else "rejected"
    status_label = "ACCEPTED" if d["accepted"] else "REJECTED"
    action_class = f"action-{action}"

    # revise_search gets highlight treatment
    highlight_class = " decision-highlight" if action == "revise_search" else ""

    rejection_html = ""
    if not d["accepted"] and d.get("rejection_reason"):
        rejection_html = f'<div class="rejection">Rejected: {d["rejection_reason"]}</div>'

    context_html = _build_decision_context(d)
    space_change_html = _build_space_change_html(d)

    return f"""
        <div class="decision {status_class} {action_class}{highlight_class}">
            <div class="decision-header">
                <span class="decision-round">Round {d['round']}</span>
                <span class="decision-action {action_class}">{icon} {action}</span>
                <span class="status-{status_class}">{status_label}</span>
            </div>
            {context_html}
            <details class="decision-justification">
                <summary class="label">Justification</summary>
                {d['justification']}
            </details>
            {space_change_html}
            {rejection_html}
        </div>"""
```

Update `_render_html` to call `_build_decision_card` instead of inline HTML:

Replace the decision loop (the `for d in decisions:` block that builds `decisions_html`) with:

```python
    decisions_html = ""
    for d in decisions:
        decisions_html += _build_decision_card(d)
```

Add the action-specific CSS rules inside the `<style>` block (after the existing `.decision` rules):

```css
  /* Action-type colors */
  .action-continue .decision-action, .decision.action-continue { border-left-color: var(--green); }
  .action-continue .decision-action { color: var(--green); }
  .action-narrow_search .decision-action, .action-widen_search .decision-action { color: var(--accent); }
  .decision.action-narrow_search, .decision.action-widen_search { border-left-color: var(--accent); }
  .action-revise_search .decision-action { color: var(--purple); }
  .decision.action-revise_search { border-left-color: var(--purple); }
  .action-increase_budget .decision-action { color: var(--orange); }
  .decision.action-increase_budget { border-left-color: var(--orange); }
  .action-stop .decision-action { color: var(--red); }
  .decision.action-stop { border-left-color: var(--red); }
  .decision.rejected { border-left-color: var(--red) !important; }
  /* Highlight revise_search as the "aha" moment */
  .decision-highlight { box-shadow: 0 0 12px rgba(188,140,255,0.15); border-width: 1px 1px 1px 3px; }
  /* Collapsible justification */
  .decision-justification { font-size: 0.85rem; margin-top: 0.5rem; cursor: pointer; }
  .decision-justification summary.label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report.py::TestDecisionLogStyling -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run existing report tests to check for regressions**

Run: `uv run pytest tests/test_report.py -v`
Expected: All existing tests PASS. The `TestGenerateReport` tests still find "Decision Log", "narrow_search", etc. in the output.

- [ ] **Step 6: Commit**

```bash
git add src/agentune/report.py tests/test_report.py
git commit -m "feat: add action-type color coding, icons, and collapsible justification to decision log"
```

---

### Task 2: Guardrails summary box and enhanced rejected-proposal cards

**Files:**
- Modify: `src/agentune/report.py` — add `_build_guardrails_summary` function, enhance rejected card rendering
- Test: `tests/test_report.py` — verify guardrails summary content

- [ ] **Step 1: Write failing tests for guardrails summary**

Add to `tests/test_report.py`:

```python
class TestGuardrailsSummary:
    """Test the guardrails summary box above the decision log."""

    def test_no_rejections(self):
        from agentune.report import _build_guardrails_summary
        decisions = [
            {"action": "continue", "accepted": True, "rejection_reason": None, "round": 1},
        ]
        html = _build_guardrails_summary(decisions)
        assert "All proposals accepted" in html

    def test_one_rejection(self):
        from agentune.report import _build_guardrails_summary
        decisions = [
            {"action": "narrow_search", "accepted": False, "rejection_reason": "cooldown violation", "round": 2},
            {"action": "continue", "accepted": True, "rejection_reason": None, "round": 2},
        ]
        html = _build_guardrails_summary(decisions)
        assert "1 proposal rejected" in html
        assert "narrow_search" in html
        assert "cooldown violation" in html

    def test_multiple_rejections(self):
        from agentune.report import _build_guardrails_summary
        decisions = [
            {"action": "narrow_search", "accepted": False, "rejection_reason": "cooldown", "round": 2},
            {"action": "revise_search", "accepted": False, "rejection_reason": "not eligible", "round": 3},
            {"action": "continue", "accepted": True, "rejection_reason": None, "round": 3},
        ]
        html = _build_guardrails_summary(decisions)
        assert "2 proposals rejected" in html


class TestRejectedDecisionCard:
    """Test enhanced rejected-proposal card rendering."""

    def test_rejected_card_has_strikethrough_action(self):
        from agentune.report import _build_decision_card
        d = {
            "round": 2,
            "action": "narrow_search",
            "accepted": False,
            "rejection_reason": "Cooldown violation: widen_search was applied 1 round(s) ago",
            "justification": "Tightening ranges",
            "reasoning": None,
            "proposed_search_space": None,
            "summary": {"best_score": 0.9, "param_importance": {}},
            "prev_search_space": [],
        }
        html = _build_decision_card(d)
        assert "guardrail-callout" in html
        assert "Cooldown violation" in html

    def test_rejected_card_shows_followup_action(self):
        from agentune.report import _build_decision_card
        d = {
            "round": 2,
            "action": "narrow_search",
            "accepted": False,
            "rejection_reason": "cooldown",
            "justification": "Tightening ranges",
            "reasoning": None,
            "proposed_search_space": None,
            "summary": {"best_score": 0.9, "param_importance": {}},
            "prev_search_space": [],
            "followup_action": "continue",
        }
        html = _build_decision_card(d)
        assert "Agent then proposed" in html
        assert "continue" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py::TestGuardrailsSummary tests/test_report.py::TestRejectedDecisionCard -v`
Expected: FAIL — functions don't exist or lack new behavior

- [ ] **Step 3: Implement `_build_guardrails_summary`**

Add to `src/agentune/report.py`:

```python
def _build_guardrails_summary(decisions: list[dict]) -> str:
    """Build a summary box showing rejected proposals (guardrail interventions)."""
    rejected = [d for d in decisions if not d.get("accepted", True)]

    if not rejected:
        return """
        <div class="guardrails-box guardrails-clean">
            <span class="guardrails-icon">&#x2713;</span>
            All proposals accepted &mdash; no guardrail interventions
        </div>"""

    count = len(rejected)
    label = "proposal rejected" if count == 1 else "proposals rejected"
    items = ""
    for d in rejected:
        items += (
            f'<div class="guardrails-item">'
            f'<span class="guardrails-round">Round {d["round"]}</span>: '
            f'<strong>{d["action"]}</strong> blocked &mdash; {d.get("rejection_reason", "unknown")}'
            f'</div>'
        )

    return f"""
        <div class="guardrails-box guardrails-active">
            <span class="guardrails-icon">&#x26A0;</span>
            <strong>{count} {label}</strong> by guardrails
            {items}
        </div>"""
```

- [ ] **Step 4: Enhance `_build_decision_card` for rejected proposals**

Update `_build_decision_card` in `src/agentune/report.py` — replace the `rejection_html` block:

```python
    rejection_html = ""
    if not d["accepted"] and d.get("rejection_reason"):
        rejection_html = f"""
            <div class="guardrail-callout">
                <span class="guardrail-label">&#x1f6e1; Guardrail</span>
                {d["rejection_reason"]}
            </div>"""
        followup = d.get("followup_action")
        if followup:
            rejection_html += f'<div class="guardrail-followup">Agent then proposed: <strong>{followup}</strong></div>'
```

- [ ] **Step 5: Wire guardrails summary into `_render_html` and enrich decisions with followup**

In `_render_html`, before the decision log loop, add followup enrichment and insert guardrails summary:

```python
    # Enrich rejected decisions with what the agent did next
    for i, d in enumerate(decisions):
        if not d["accepted"]:
            # Find the next accepted decision for the same round
            for j in range(i + 1, len(decisions)):
                if decisions[j]["round"] == d["round"] and decisions[j]["accepted"]:
                    d["followup_action"] = decisions[j]["action"]
                    break

    guardrails_html = _build_guardrails_summary(decisions)
```

Then in the HTML template, replace the Decision Log section:

```html
<h2>Decision Log</h2>
{guardrails_html}
{decisions_html}
```

Add guardrails CSS to the `<style>` block:

```css
  /* Guardrails summary box */
  .guardrails-box { padding: 0.75rem 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 0.85rem; }
  .guardrails-clean { background: rgba(63,185,80,0.08); border: 1px solid var(--green); color: var(--green); }
  .guardrails-active { background: rgba(248,81,73,0.08); border: 1px solid var(--red); }
  .guardrails-icon { margin-right: 0.5rem; }
  .guardrails-item { margin-top: 0.4rem; padding-left: 1.5rem; color: var(--text-muted); font-size: 0.8rem; }
  /* Guardrail callout in rejected cards */
  .guardrail-callout { font-size: 0.8rem; color: var(--red); margin-top: 0.5rem; padding: 0.5rem 0.75rem; background: rgba(248,81,73,0.08); border-radius: 4px; border-left: 3px solid var(--red); }
  .guardrail-label { font-weight: 600; margin-right: 0.4rem; }
  .guardrail-followup { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem; padding-left: 0.75rem; }
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/test_report.py -v`
Expected: All tests pass including new `TestGuardrailsSummary` and `TestRejectedDecisionCard`

- [ ] **Step 7: Commit**

```bash
git add src/agentune/report.py tests/test_report.py
git commit -m "feat: add guardrails summary box and enhanced rejected-proposal cards to report"
```

---

### Task 3: CLI `--demo` mode

**Files:**
- Modify: `src/agentune/cli.py` — add `--demo` flag to `run` command, print formatted narration block
- Test: `tests/test_cli.py` — verify demo output formatting

- [ ] **Step 1: Write failing test for demo output**

Add to `tests/test_cli.py`:

```python
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from agentune.cli import cli


class TestRunDemoMode:
    """Test --demo flag on the run command."""

    @patch("agentune.cli.RoundRunner")
    @patch("agentune.cli.load_dataset")
    @patch("agentune.cli.CampaignService")
    @patch("agentune.cli._get_db")
    def test_demo_prints_narration_block(self, mock_db, mock_svc_cls, mock_load, mock_runner_cls):
        # Setup mocks
        mock_svc = MagicMock()
        mock_svc_cls.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = {
            "id": 1, "name": "test", "state": "RUNNING",
            "metric_name": "accuracy", "objective_direction": "maximize",
            "dataset": "breast_cancer", "split_seed": 42,
            "stop_conditions": '{"max_rounds": 6}',
        }

        from agentune.runner import RunResult
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_next_round.return_value = RunResult(
            status="AWAITING_AGENT", round_number=2, stop_reason=None,
        )

        mock_load.return_value = (MagicMock(), {"metric": "accuracy", "direction": "maximize"})

        # Mock get_rounds to return summary data
        mock_svc.get_rounds.return_value = [
            {"round_number": 2, "summary": {
                "best_score": 0.95, "delta_from_prev": 0.01,
                "param_importance": {"lr": 0.4, "depth": 0.3},
                "plateau_signal": False,
            }},
        ]
        mock_svc.get_campaign_history.return_value = {
            "decisions": [
                {"round_id": 1, "action": "continue", "accepted": True,
                 "justification": "Still improving"},
            ],
        }

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "test", "--dataset", "breast_cancer", "--demo"])

        assert result.exit_code == 0
        assert "Round 2" in result.output
        assert "0.9500" in result.output
        assert "+0.0100" in result.output

    @patch("agentune.cli.RoundRunner")
    @patch("agentune.cli.load_dataset")
    @patch("agentune.cli.CampaignService")
    @patch("agentune.cli._get_db")
    def test_no_demo_prints_minimal(self, mock_db, mock_svc_cls, mock_load, mock_runner_cls):
        mock_svc = MagicMock()
        mock_svc_cls.return_value = mock_svc
        mock_svc.get_campaign_by_name.return_value = {
            "id": 1, "name": "test", "state": "RUNNING",
            "dataset": "breast_cancer", "split_seed": 42,
        }

        from agentune.runner import RunResult
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_runner.run_next_round.return_value = RunResult(
            status="AWAITING_AGENT", round_number=2, stop_reason=None,
        )
        mock_load.return_value = (MagicMock(), {"metric": "accuracy", "direction": "maximize"})

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "test", "--dataset", "breast_cancer"])

        assert result.exit_code == 0
        # Without --demo, just the minimal status line
        assert "Round 2: AWAITING_AGENT" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::TestRunDemoMode -v`
Expected: FAIL — `run` command doesn't accept `--demo` flag

- [ ] **Step 3: Add `--demo` flag and narration to `run` command**

In `src/agentune/cli.py`, modify the `run` command:

```python
@cli.command()
@click.argument("name")
@click.option("--dataset", required=True, help="Dataset name (see 'agentune init --help' for available datasets). Metric and direction are stored in the campaign config.")
@click.option("--split-seed", default=42, type=int)
@click.option("--demo", is_flag=True, help="Print formatted narration for live presentations")
def run(name: str, dataset: str, split_seed: int, demo: bool) -> None:
    """Execute the next study round for a campaign."""
    from agentune.datasets import load_dataset
    from agentune.runner import RoundRunner

    db = _get_db()
    try:
        service = CampaignService(db)
        campaign = _get_campaign_or_exit(service, name)
        split, _ = load_dataset(dataset, seed=split_seed)
        runner = RoundRunner(db, split)
        result = runner.run_next_round(campaign["id"])

        if demo:
            _print_demo_narration(service, campaign, result)
        else:
            click.echo(f"Round {result.round_number}: {result.status}")
            if result.stop_reason:
                click.echo(f"Stop reason: {result.stop_reason}")
    except Exception as error:
        _exit_for_exception(error)
    finally:
        db.close()
```

Add the `_print_demo_narration` function before the `run` command:

```python
def _print_demo_narration(service: CampaignService, campaign: dict, result) -> None:
    """Print formatted narration block for --demo mode."""
    bar = "\u2550" * 56  # ═

    # Get round summary
    rounds = service.get_rounds(campaign["id"])
    current_round = None
    for r in rounds:
        if r["round_number"] == result.round_number:
            current_round = r
            break

    summary = {}
    if current_round and current_round.get("summary"):
        summary = _load_json(current_round["summary"])

    # Get latest decision
    history = service.get_campaign_history(campaign["id"])
    latest_decision = None
    for d in reversed(history.get("decisions", [])):
        if d.get("accepted"):
            latest_decision = d
            break

    # Build score line
    best_score = summary.get("best_score")
    delta = summary.get("delta_from_prev")
    score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
    if delta is not None and delta != 0:
        sign = "+" if delta > 0 else ""
        score_str += f" ({sign}{delta:.4f})"

    # Build signals line
    signals = []
    importance = summary.get("param_importance", {})
    top_params = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if top_params:
        signals.append(", ".join(f"{n} ({v:.0%})" for n, v in top_params))
    if summary.get("plateau_signal"):
        signals.append("plateau detected")
    signals_str = "; ".join(signals) if signals else "gathering data"

    # Get max_rounds for display
    stop_cond = campaign.get("stop_conditions")
    max_rounds = None
    if stop_cond:
        sc = _load_json(stop_cond)
        max_rounds = sc.get("max_rounds")
    round_label = f"Round {result.round_number}"
    if max_rounds:
        round_label += f"/{max_rounds}"

    # Build decision line
    decision_str = ""
    if latest_decision:
        decision_str = f"  Agent decided: {latest_decision['action']}"
        justification = latest_decision.get("justification", "")
        if justification:
            # Truncate to 60 chars for readability
            if len(justification) > 60:
                justification = justification[:57] + "..."
            decision_str += f'\n  Reason: "{justification}"'

    click.echo()
    click.echo(click.style(f"  {bar}", fg="cyan"))
    click.echo(click.style(f"  {round_label} complete", fg="cyan", bold=True))
    click.echo(f"  Score: {score_str}")
    click.echo(f"  Signals: {signals_str}")
    if decision_str:
        click.echo(decision_str)
    if result.stop_reason:
        click.echo(click.style(f"  Stopped: {result.stop_reason}", fg="yellow"))
    click.echo(click.style(f"  {bar}", fg="cyan"))
    click.echo()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::TestRunDemoMode -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run all CLI tests for regressions**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/agentune/cli.py tests/test_cli.py
git commit -m "feat: add --demo flag to run command for live presentation narration"
```

---

### Task 4: Final integration verification

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/test_report.py tests/test_cli.py -v`
Expected: All tests pass

- [ ] **Step 2: Generate a report with test data to visually verify**

If a campaign exists in the local DB:
```bash
uv run agentune report <any-existing-campaign> -o /tmp/test-report.html
```

Open the HTML file and verify:
- Decision cards have colored left borders matching action type
- revise_search cards have subtle purple glow
- Justification text is inside a collapsible `<details>` element
- Guardrails summary box appears above decision log
- Rejected proposals show the guardrail callout with shield icon

- [ ] **Step 3: Commit all changes**

```bash
git add -A
git commit -m "feat: presentation improvements — demo mode, decision log styling, guardrails viz"
```
