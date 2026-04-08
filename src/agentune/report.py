"""HTML report generator — fetches all data from Postgres and renders a self-contained report."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agentune.core.campaign import CampaignService
from agentune.core.db import Database

ACTION_ICONS = {
    "continue": "\u25b6",        # ▶
    "narrow_search": "\u25c1",   # ◁
    "widen_search": "\u25b7",    # ▷
    "revise_search": "\u21bb",   # ↻
    "increase_budget": "\u25b2", # ▲
    "stop": "\u25a0",            # ■
}


def _load_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _fmt_score(score: float | None, precision: int = 4) -> str:
    if score is None:
        return "—"
    return f"{score:.{precision}f}"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds == 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


def _fmt_gen_gap(gap: float | None, best_score: float | None) -> str:
    """Format generalization gap as a relative percentage of best_score.

    The raw gap is an absolute difference (e.g. 46.16 for RMSE ~205).
    Displaying it via _fmt_pct would give nonsensical "4615.9%".
    Instead, compute gap / |best_score| * 100 for a meaningful percentage.
    """
    if gap is None:
        return "—"
    if best_score is None or best_score == 0:
        return _fmt_score(gap)
    relative = gap / abs(best_score)
    return f"{relative:.1%}"


def _build_decision_context(d: dict) -> str:
    """Build HTML showing what the agent observed before making this decision."""
    summary = d.get("summary", {})
    if not summary:
        return ""

    best_score = summary.get("best_score")
    round_best = summary.get("round_best_score")
    delta = summary.get("delta_from_prev")
    new_best = summary.get("new_best_in_round", False)
    plateau = summary.get("plateau_signal", False)
    gen_gap = summary.get("generalization_gap")
    importance = summary.get("param_importance", {})
    top_params = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # Structured reasoning from reasoning JSON (if available from Claude agent)
    reasoning = d.get("reasoning")
    if reasoning:
        return _build_structured_reasoning(reasoning)

    # Otherwise build context from summary signals
    signals = []

    score_str = _fmt_score(best_score)
    if delta is not None and delta != 0:
        sign = "+" if delta > 0 else ""
        score_str += f" ({sign}{delta:.4f})"
    signals.append(f"Best score: <strong>{score_str}</strong>")

    if new_best:
        signals.append('<span class="signal-good">New best found this round</span>')
    else:
        signals.append('<span class="signal-warn">No new best this round</span>')

    if plateau:
        signals.append('<span class="signal-warn">Plateau detected</span> (no improvement in last 30% of trials)')

    if gen_gap is not None:
        # Compute relative gap for threshold and display
        if best_score is not None and best_score != 0:
            relative_gap = gen_gap / abs(best_score)
            gap_class = "signal-warn" if relative_gap > 0.1 else "signal-good"
            gap_display = f"{relative_gap:.1%}"
        else:
            gap_class = "signal-good"
            gap_display = _fmt_score(gen_gap)
        signals.append(f'Generalization gap: <span class="{gap_class}">{gap_display}</span>')

    # Param importance
    if top_params:
        importance_items = []
        for pname, pval in top_params:
            bar_width = max(2, int(pval * 100))
            importance_items.append(
                f'<div class="importance-row">'
                f'<span class="importance-name">{pname}</span>'
                f'<div class="importance-bar-bg"><div class="importance-bar" style="width:{bar_width}%"></div></div>'
                f'<span class="importance-val">{pval:.1%}</span>'
                f'</div>'
            )
        importance_html = "\n".join(importance_items)
    else:
        importance_html = ""

    signals_html = " · ".join(signals)

    return f"""
            <div class="decision-context">
                <div class="context-signals">{signals_html}</div>
                {f'<div class="context-importance">{importance_html}</div>' if importance_html else ''}
            </div>"""


def _build_structured_reasoning(reasoning: dict) -> str:
    """Build HTML from structured reasoning JSON (from Claude agent decisions)."""
    obs = reasoning.get("observation", {})
    diag = reasoning.get("diagnosis", {})

    parts = []

    # Observation
    if obs:
        items = []
        best = obs.get("best_score")
        if best is not None:
            items.append(f"Best: <strong>{best:.4f}</strong>")
        if obs.get("new_best_in_round"):
            items.append('<span class="signal-good">New best</span>')
        if obs.get("plateau_signal"):
            items.append('<span class="signal-warn">Plateau</span>')
        top = obs.get("top_params", [])
        if top:
            items.append("Top: " + ", ".join(f"{n} ({v:.0%})" for n, v in top[:3]))
        if items:
            parts.append(f'<div class="context-signals">{" · ".join(items)}</div>')

    # Diagnosis
    reasons = diag.get("reasons", [])
    if reasons:
        reasons_html = "".join(f"<li>{r}</li>" for r in reasons)
        parts.append(f'<div class="context-diagnosis"><strong>Diagnosis:</strong><ul>{reasons_html}</ul></div>')

    if not parts:
        return ""

    return f'<div class="decision-context">{"".join(parts)}</div>'


def _build_space_change_html(d: dict) -> str:
    """Show search space changes (narrowing/widening/revision) visually."""
    proposed = d.get("proposed_search_space")
    if not proposed:
        return ""

    prev_space = d.get("prev_search_space", [])
    prev_map = {p["name"]: p for p in prev_space} if prev_space else {}
    proposed_names = {p["name"] for p in proposed}
    prev_names = {p["name"] for p in prev_space} if prev_space else set()

    rows = []
    for p in proposed:
        pname = p["name"]
        old = prev_map.get(pname)

        if pname not in prev_names:
            # Added param
            if p.get("type") == "categorical":
                new_range = f"choices: {p.get('choices', [])}"
            else:
                new_range = f"[{p.get('low', '?'):.4g}, {p.get('high', '?'):.4g}]"
            rows.append(f'<tr class="space-added"><td>+ {pname}</td><td>—</td><td>{new_range}</td></tr>')
        elif old:
            # Existing param — show range change
            if p.get("type") == "categorical":
                old_range = new_range = "categorical"
            else:
                old_range = f"[{old.get('low', '?'):.4g}, {old.get('high', '?'):.4g}]"
                new_range = f"[{p.get('low', '?'):.4g}, {p.get('high', '?'):.4g}]"
            if old_range != new_range:
                rows.append(f'<tr><td>{pname}</td><td>{old_range}</td><td>{new_range}</td></tr>')

    # Dropped params
    for pname in sorted(prev_names - proposed_names):
        old = prev_map[pname]
        if old.get("type") == "categorical":
            old_range = "categorical"
        else:
            old_range = f"[{old.get('low', '?'):.4g}, {old.get('high', '?'):.4g}]"
        rows.append(f'<tr class="space-dropped"><td>- {pname}</td><td>{old_range}</td><td>—</td></tr>')

    if not rows:
        return ""

    return f"""
            <div class="space-changes">
                <table class="space-table">
                    <thead><tr><th>Param</th><th>Before</th><th>After</th></tr></thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>"""


def _build_status_banner(campaign: dict, rounds: list[dict]) -> str:
    """Build a live status banner showing campaign progress."""
    state = campaign["state"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    completed_rounds = sum(1 for r in rounds if r.get("summary"))
    max_rounds = None
    stop_cond = campaign.get("stop_conditions")
    if stop_cond:
        sc = _load_json(stop_cond) if isinstance(stop_cond, str) else stop_cond
        max_rounds = sc.get("max_rounds")

    rounds_str = f"Round {completed_rounds}"
    if max_rounds:
        rounds_str += f" of {max_rounds}"
    rounds_str += " completed"

    if state in ("RUNNING", "AWAITING_AGENT", "CREATED"):
        banner_class = "banner-running"
        icon = "&#9654;"  # play
        label = f"Campaign in progress &mdash; {rounds_str}"
    elif state == "COMPLETED":
        banner_class = "banner-completed"
        icon = "&#10003;"  # checkmark
        reason = campaign.get("termination_reason", "")
        label = f"Campaign completed &mdash; {rounds_str}"
        if reason:
            label += f" &mdash; {reason}"
    elif state in ("FAILED",):
        banner_class = "banner-failed"
        icon = "&#10007;"  # x
        label = f"Campaign failed &mdash; {rounds_str}"
        detail = campaign.get("termination_detail", "")
        if detail:
            label += f" &mdash; {detail[:80]}"
    elif state in ("STOPPED", "PAUSED", "PAUSE_REQUESTED"):
        banner_class = "banner-stopped"
        icon = "&#9724;"  # stop
        label = f"Campaign {state.lower()} &mdash; {rounds_str}"
    else:
        banner_class = "banner-running"
        icon = "&#9654;"
        label = f"{state} &mdash; {rounds_str}"

    return f"""
    <div class="status-banner {banner_class}">
        <span class="banner-icon">{icon}</span>
        <span class="banner-label">{label}</span>
        <span class="banner-time">Last updated: {now}</span>
    </div>"""


def _build_progress_timeline(campaign: dict, rounds: list[dict], decisions: list[dict]) -> str:
    """Build a visual progress timeline with dots for each round."""
    stop_cond = campaign.get("stop_conditions")
    max_rounds = None
    if stop_cond:
        sc = _load_json(stop_cond) if isinstance(stop_cond, str) else stop_cond
        max_rounds = sc.get("max_rounds")

    # Build action map: round_number -> action taken
    action_map = {}
    for d in decisions:
        if d.get("accepted", True):
            action_map[d["round"]] = d["action"]

    completed_rounds = [r for r in rounds if r.get("summary")]
    n_completed = len(completed_rounds)
    total_dots = max(max_rounds or n_completed, n_completed)

    state = campaign["state"]
    is_running = state in ("RUNNING", "AWAITING_AGENT", "CREATED")

    dots = ""
    for i in range(1, total_dots + 1):
        if i <= n_completed:
            # Completed round
            action = action_map.get(i, "")
            action_label = f'<span class="tl-action">{action}</span>' if action else ""
            dots += f"""
            <div class="tl-step completed">
                <div class="tl-dot completed"></div>
                <div class="tl-label">R{i}</div>
                {action_label}
            </div>"""
        elif i == n_completed + 1 and is_running:
            # Current round (pulsing)
            dots += f"""
            <div class="tl-step current">
                <div class="tl-dot current"></div>
                <div class="tl-label">R{i}</div>
                <span class="tl-action">in progress</span>
            </div>"""
        else:
            # Future round
            dots += f"""
            <div class="tl-step future">
                <div class="tl-dot future"></div>
                <div class="tl-label">R{i}</div>
            </div>"""

    return f"""
    <div class="timeline-container">
        <div class="timeline-track">{dots}</div>
    </div>"""


def generate_report(db: Database, campaign_name: str) -> str:
    """Generate a self-contained HTML report for a campaign."""
    service = CampaignService(db)
    campaign = service.get_campaign_by_name(campaign_name)
    if campaign is None:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    rounds = service.get_rounds(campaign["id"])
    history = service.get_campaign_history(campaign["id"])
    decisions = history["decisions"]

    # Parse summaries
    round_data = []
    for r in rounds:
        summary = _load_json(r.get("summary")) if r.get("summary") else {}
        search_space = _load_json(r.get("search_space")) if r.get("search_space") else []
        round_data.append({
            "number": r["round_number"],
            "state": r["state"],
            "study_name": r["optuna_study_name"],
            "budget": r["budget"],
            "search_space": search_space,
            "summary": summary,
        })

    # Parse decisions — enrich with the round summary that drove each decision
    decision_data = []
    round_map = {r["id"]: r["round_number"] for r in rounds}
    summary_by_round = {rd["number"]: rd["summary"] for rd in round_data}
    space_by_round = {rd["number"]: rd["search_space"] for rd in round_data}
    for d in decisions:
        rnum = round_map.get(d["round_id"], "?")
        proposed_space = _load_json(d.get("proposed_search_space")) if d.get("proposed_search_space") else None
        reasoning = _load_json(d.get("reasoning")) if d.get("reasoning") else None
        decision_data.append({
            "round": rnum,
            "action": d["action"],
            "accepted": d["accepted"],
            "rejection_reason": d.get("rejection_reason"),
            "justification": d["justification"],
            "reasoning": reasoning,
            "proposed_search_space": proposed_space,
            "summary": summary_by_round.get(rnum, {}),
            "prev_search_space": space_by_round.get(rnum, []),
        })

    return _render_html(campaign, round_data, decision_data)


def _build_decision_card(d: dict) -> str:
    """Build a single decision card with action-specific styling."""
    action = d["action"]
    icon = ACTION_ICONS.get(action, "")
    status_class = "accepted" if d["accepted"] else "rejected"
    status_label = "ACCEPTED" if d["accepted"] else "REJECTED"
    action_class = f"action-{action}"
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


def _render_html(campaign: dict, rounds: list[dict], decisions: list[dict]) -> str:
    name = campaign["name"]
    metric = campaign["metric_name"]
    direction = campaign["objective_direction"]
    direction_symbol = "↑" if direction == "maximize" else "↓"

    # Extract score progression for chart
    scores = []
    test_scores = []
    for r in rounds:
        s = r["summary"]
        scores.append(s.get("best_score"))
        test_scores.append(s.get("test_score"))

    # Find best round
    best_score = None
    best_test = None
    best_params = {}
    for r in rounds:
        s = r["summary"]
        sc = s.get("best_score")
        if sc is not None:
            if best_score is None or (direction == "maximize" and sc > best_score) or (direction == "minimize" and sc < best_score):
                best_score = sc
                best_test = s.get("test_score")
                best_params = s.get("best_params", {})

    # Build rounds table rows
    rounds_rows = ""
    for r in rounds:
        s = r["summary"]
        importance = s.get("param_importance", {})
        top_params = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_str = ", ".join(f"{n} ({v:.0%})" for n, v in top_params) if top_params else "—"

        # Find the decision for this round
        action_str = "—"
        for d in decisions:
            if d["round"] == r["number"]:
                status = "✓" if d["accepted"] else "✗"
                action_str = f"{status} {d['action']}"
                break

        rounds_rows += f"""
        <tr>
            <td>{r['number']}</td>
            <td>{_fmt_score(s.get('best_score'))}</td>
            <td>{_fmt_score(s.get('test_score'))}</td>
            <td>{_fmt_score(s.get('delta_from_prev'))}</td>
            <td>{s.get('round_completed_trials', '—')}</td>
            <td>{_fmt_time(s.get('round_wall_time_seconds'))}</td>
            <td>{'Yes' if s.get('plateau_signal') else 'No'}</td>
            <td class="param-col">{top_str}</td>
            <td>{_fmt_gen_gap(s.get('generalization_gap'), s.get('best_score'))}</td>
            <td>{action_str}</td>
        </tr>"""

    # Build score chart (simple CSS bar chart)
    chart_html = _build_score_chart(scores, test_scores, direction)

    # Build best params table
    params_rows = ""
    for pname, pval in sorted(best_params.items()):
        if isinstance(pval, float):
            params_rows += f"<tr><td>{pname}</td><td>{pval:.6g}</td></tr>\n"
        else:
            params_rows += f"<tr><td>{pname}</td><td>{pval}</td></tr>\n"

    # Build decision log with full reasoning context
    decisions_html = ""
    for d in decisions:
        decisions_html += _build_decision_card(d)

    # Build search space evolution
    space_html = _build_search_space_evolution(rounds)

    # Build status banner and progress timeline
    banner_html = _build_status_banner(campaign, rounds)
    timeline_html = _build_progress_timeline(campaign, rounds, decisions)

    created = campaign.get("created_at", "")
    if isinstance(created, datetime):
        created = created.strftime("%Y-%m-%d %H:%M")

    total_wall = 0
    total_trials = 0
    if rounds and rounds[-1]["summary"]:
        total_wall = rounds[-1]["summary"].get("total_wall_time_seconds", 0)
        total_trials = rounds[-1]["summary"].get("total_trials", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agentune Report — {name}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922; --purple: #bc8cff;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}
  .overview {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }}
  .card-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-value {{ font-size: 1.4rem; font-weight: 600; margin-top: 0.25rem; }}
  .card-value.green {{ color: var(--green); }}
  .card-value.orange {{ color: var(--orange); }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ font-variant-numeric: tabular-nums; }}
  tr:hover {{ background: rgba(88,166,255,0.04); }}
  .param-col {{ max-width: 250px; font-size: 0.8rem; }}
  .chart-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
  .bar-chart {{ display: flex; align-items: flex-end; gap: 0.5rem; height: 180px; padding-top: 1rem; }}
  .bar-group {{ display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; justify-content: flex-end; }}
  .bar {{ width: 100%; max-width: 50px; border-radius: 4px 4px 0 0; transition: height 0.3s; position: relative; }}
  .bar.val {{ background: var(--accent); }}
  .bar.test {{ background: var(--purple); opacity: 0.7; }}
  .bars-wrapper {{ display: flex; gap: 3px; align-items: flex-end; width: 100%; justify-content: center; }}
  .bar-label {{ font-size: 0.7rem; color: var(--text-muted); margin-top: 0.4rem; }}
  .bar-value {{ font-size: 0.65rem; color: var(--text-muted); margin-bottom: 0.2rem; text-align: center; }}
  .chart-legend {{ display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.8rem; color: var(--text-muted); }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 0.3rem; vertical-align: middle; }}
  .legend-dot.val {{ background: var(--accent); }}
  .legend-dot.test {{ background: var(--purple); }}
  .decision {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }}
  .decision.rejected {{ border-left: 3px solid var(--red); }}
  .decision.accepted {{ border-left: 3px solid var(--green); }}
  .decision-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; }}
  .decision-round {{ font-size: 0.8rem; color: var(--text-muted); }}
  .decision-action {{ font-size: 1rem; font-weight: 600; color: var(--accent); }}
  .decision-context {{ background: rgba(88,166,255,0.04); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; margin: 0.5rem 0; }}
  .context-signals {{ font-size: 0.8rem; color: var(--text-muted); line-height: 1.8; }}
  .context-signals strong {{ color: var(--text); }}
  .signal-good {{ color: var(--green); }}
  .signal-warn {{ color: var(--orange); }}
  .context-importance {{ margin-top: 0.5rem; }}
  .importance-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.2rem 0; }}
  .importance-name {{ font-size: 0.75rem; font-family: monospace; width: 140px; color: var(--text-muted); }}
  .importance-bar-bg {{ flex: 1; height: 8px; background: var(--border); border-radius: 4px; max-width: 200px; }}
  .importance-bar {{ height: 100%; background: var(--accent); border-radius: 4px; }}
  .importance-val {{ font-size: 0.7rem; color: var(--text-muted); width: 40px; text-align: right; }}
  .context-diagnosis {{ font-size: 0.8rem; margin-top: 0.5rem; }}
  .context-diagnosis ul {{ margin: 0.25rem 0 0 1.25rem; }}
  .context-diagnosis li {{ color: var(--text-muted); margin: 0.15rem 0; }}
  .decision-justification {{ font-size: 0.85rem; margin-top: 0.5rem; }}
  .decision-justification .label {{ color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  /* Action-type colors */
  .action-continue .decision-action, .decision.action-continue {{ border-left-color: var(--green); }}
  .action-continue .decision-action {{ color: var(--green); }}
  .action-narrow_search .decision-action, .action-widen_search .decision-action {{ color: var(--accent); }}
  .decision.action-narrow_search, .decision.action-widen_search {{ border-left-color: var(--accent); }}
  .action-revise_search .decision-action {{ color: var(--purple); }}
  .decision.action-revise_search {{ border-left-color: var(--purple); }}
  .action-increase_budget .decision-action {{ color: var(--orange); }}
  .decision.action-increase_budget {{ border-left-color: var(--orange); }}
  .action-stop .decision-action {{ color: var(--red); }}
  .decision.action-stop {{ border-left-color: var(--red); }}
  .decision.rejected {{ border-left-color: var(--red) !important; }}
  /* Highlight revise_search as the "aha" moment */
  .decision-highlight {{ box-shadow: 0 0 12px rgba(188,140,255,0.15); border-width: 1px 1px 1px 3px; }}
  /* Collapsible justification */
  .decision-justification {{ font-size: 0.85rem; margin-top: 0.5rem; cursor: pointer; }}
  .decision-justification summary.label {{ color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .rejection {{ font-size: 0.8rem; color: var(--red); margin-top: 0.5rem; padding: 0.4rem 0.75rem; background: rgba(248,81,73,0.08); border-radius: 4px; }}
  .status-accepted {{ color: var(--green); font-size: 0.8rem; }}
  .status-rejected {{ color: var(--red); font-size: 0.8rem; }}
  .space-changes {{ margin-top: 0.5rem; }}
  .space-table {{ font-size: 0.8rem; }}
  .space-table th {{ font-size: 0.7rem; }}
  .space-added td {{ color: var(--green); }}
  .space-dropped td {{ color: var(--red); }}
  .space-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.25rem 0; }}
  .param-badge {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.75rem; font-family: monospace; }}
  .param-badge.added {{ border-color: var(--green); color: var(--green); }}
  .param-badge.dropped {{ border-color: var(--red); color: var(--red); text-decoration: line-through; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.75rem; text-align: center; }}
  /* Status banner */
  .status-banner {{ display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.25rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 0.9rem; }}
  .banner-running {{ background: rgba(88,166,255,0.1); border: 1px solid var(--accent); }}
  .banner-completed {{ background: rgba(63,185,80,0.1); border: 1px solid var(--green); }}
  .banner-failed {{ background: rgba(248,81,73,0.1); border: 1px solid var(--red); }}
  .banner-stopped {{ background: rgba(210,153,34,0.1); border: 1px solid var(--orange); }}
  .banner-icon {{ font-size: 1.1rem; }}
  .banner-running .banner-icon {{ color: var(--accent); }}
  .banner-completed .banner-icon {{ color: var(--green); }}
  .banner-failed .banner-icon {{ color: var(--red); }}
  .banner-stopped .banner-icon {{ color: var(--orange); }}
  .banner-label {{ flex: 1; }}
  .banner-time {{ font-size: 0.75rem; color: var(--text-muted); }}
  /* Progress timeline */
  .timeline-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 2rem; overflow-x: auto; }}
  .timeline-track {{ display: flex; align-items: flex-start; gap: 0; position: relative; min-width: max-content; }}
  .tl-step {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 70px; position: relative; }}
  .tl-step:not(:last-child)::after {{ content: ''; position: absolute; top: 9px; left: calc(50% + 12px); right: calc(-50% + 12px); height: 2px; background: var(--border); z-index: 0; }}
  .tl-step.completed:not(:last-child)::after {{ background: var(--accent); }}
  .tl-dot {{ width: 18px; height: 18px; border-radius: 50%; z-index: 1; border: 2px solid var(--border); background: var(--bg); }}
  .tl-dot.completed {{ background: var(--accent); border-color: var(--accent); }}
  .tl-dot.current {{ background: var(--bg); border-color: var(--accent); animation: pulse-dot 1.5s ease-in-out infinite; }}
  .tl-dot.future {{ background: var(--bg); border-color: var(--border); }}
  @keyframes pulse-dot {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(88,166,255,0.4); }} 50% {{ box-shadow: 0 0 0 6px rgba(88,166,255,0); }} }}
  .tl-label {{ font-size: 0.7rem; color: var(--text-muted); margin-top: 0.4rem; }}
  .tl-action {{ font-size: 0.6rem; color: var(--accent); margin-top: 0.15rem; max-width: 70px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .tl-step.current .tl-action {{ color: var(--orange); }}
</style>
</head>
<body>

<h1>Agentune Report</h1>
<div class="subtitle">{name} &mdash; {created}</div>

{banner_html}

<div class="overview">
  <div class="card">
    <div class="card-label">State</div>
    <div class="card-value">{campaign['state']}</div>
  </div>
  <div class="card">
    <div class="card-label">Metric</div>
    <div class="card-value">{metric} {direction_symbol}</div>
  </div>
  <div class="card">
    <div class="card-label">Best Val Score</div>
    <div class="card-value green">{_fmt_score(best_score)}</div>
  </div>
  <div class="card">
    <div class="card-label">Test Score</div>
    <div class="card-value orange">{_fmt_score(best_test)}</div>
  </div>
  <div class="card">
    <div class="card-label">Rounds</div>
    <div class="card-value">{len(rounds)}</div>
  </div>
  <div class="card">
    <div class="card-label">Total Trials</div>
    <div class="card-value">{total_trials}</div>
  </div>
  <div class="card">
    <div class="card-label">Wall Time</div>
    <div class="card-value">{_fmt_time(total_wall)}</div>
  </div>
  <div class="card">
    <div class="card-label">Termination</div>
    <div class="card-value">{campaign.get('termination_reason', '—')}</div>
  </div>
</div>

{timeline_html}

<h2>Score Progression</h2>
{chart_html}

<h2>Round Details</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Best Val</th><th>Test</th><th>Delta</th><th>Trials</th>
      <th>Time</th><th>Plateau</th><th>Top Params</th><th>Gen Gap</th><th>Decision</th>
    </tr>
  </thead>
  <tbody>
    {rounds_rows}
  </tbody>
</table>

<h2>Search Space Evolution</h2>
{space_html}

<h2>Best Hyperparameters</h2>
<table>
  <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
  <tbody>
    {params_rows}
  </tbody>
</table>

<h2>Decision Log</h2>
{decisions_html}

<footer>
  Generated by Agentune &mdash; Dataset: {campaign.get('dataset', '?')} &mdash; Backend: {campaign.get('backend', '?')}
</footer>

</body>
</html>"""


def _build_score_chart(scores: list, test_scores: list, direction: str) -> str:
    """Build a simple CSS bar chart of score progression."""
    valid_scores = [s for s in scores + test_scores if s is not None]
    if not valid_scores:
        return '<div class="chart-container">No data yet.</div>'

    min_s = min(valid_scores)
    max_s = max(valid_scores)
    score_range = max_s - min_s if max_s != min_s else 0.01

    # Scale bars so minimum is ~20% height, maximum is 100%
    def pct(s):
        if s is None:
            return 0
        return max(5, int(20 + 80 * (s - min_s) / score_range))

    bars = ""
    for i, (val, test) in enumerate(zip(scores, test_scores)):
        val_h = pct(val)
        test_h = pct(test)
        val_label = _fmt_score(val) if val is not None else ""
        test_label = _fmt_score(test) if test is not None else ""

        test_bar = ""
        if test is not None:
            test_bar = f'<div class="bar test" style="height:{test_h}%"></div>'

        bars += f"""
        <div class="bar-group">
            <div class="bar-value">{val_label}</div>
            <div class="bars-wrapper">
                <div class="bar val" style="height:{val_h}%"></div>
                {test_bar}
            </div>
            <div class="bar-label">R{i+1}</div>
        </div>"""

    return f"""
    <div class="chart-container">
        <div class="bar-chart">{bars}</div>
        <div class="chart-legend">
            <span><span class="legend-dot val"></span>Validation</span>
            <span><span class="legend-dot test"></span>Test</span>
        </div>
    </div>"""


def _build_search_space_evolution(rounds: list[dict]) -> str:
    """Show how the search space changed across rounds."""
    if not rounds:
        return "<p>No rounds.</p>"

    html = ""
    prev_params = None
    for r in rounds:
        space = r["search_space"]
        current_params = {p["name"] for p in space} if space else set()

        if prev_params is not None:
            added = current_params - prev_params
            dropped = prev_params - current_params
            kept = current_params & prev_params
        else:
            added = set()
            dropped = set()
            kept = current_params

        badges = ""
        for p in sorted(kept):
            badges += f'<span class="param-badge">{p}</span>'
        for p in sorted(added):
            badges += f'<span class="param-badge added">+ {p}</span>'
        for p in sorted(dropped):
            badges += f'<span class="param-badge dropped">{p}</span>'

        html += f'<div style="margin:0.5rem 0"><strong>Round {r["number"]}:</strong> <div class="space-row">{badges}</div></div>'
        prev_params = current_params

    return html
