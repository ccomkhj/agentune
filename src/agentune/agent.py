"""Agent reasoning module: analyzes round summaries and produces data-driven decisions.

This module implements the agent's decision logic. Given a round summary and campaign
history, it observes key signals, reasons about what they mean, and proposes the next
action with a detailed justification citing specific data.

The agent follows a strict observe → diagnose → decide → justify pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentune.core.models import (
    ActionProposal,
    RoundSummary,
)


def _fmt(val: float | None) -> str:
    """Format a float for display, handling None."""
    return f"{val:.6f}" if val is not None else "N/A"


@dataclass
class Observation:
    """Raw signals extracted from a round summary."""
    round_id: int
    round_number: int
    metric_name: str
    direction: str  # "minimize" or "maximize"
    best_score: float | None
    round_best_score: float | None
    delta_from_prev: float | None
    new_best_in_round: bool
    plateau_signal: bool
    generalization_gap: float | None
    failure_rate: float
    pruned_rate: float
    trials_added: int
    round_completed_trials: int
    top_params: list[tuple[str, float]]  # sorted by importance desc
    best_params: dict
    param_ranges_used: dict[str, tuple]
    convergence_curve: list[tuple[int, float]]


@dataclass
class Diagnosis:
    """Interpreted signals: what the observations mean."""
    is_improving: bool
    improvement_magnitude: str  # "none", "small", "significant"
    is_plateauing: bool
    is_overfitting: bool
    overfitting_severity: str  # "none", "mild", "moderate", "severe"
    has_failures: bool
    dominant_param: str | None
    dominant_param_importance: float
    search_space_exhausted: bool
    best_params_at_boundary: list[str]  # params where best value is near range edge
    reasons: list[str]  # human-readable diagnostic statements


@dataclass
class SearchSpaceChange:
    """Documents one parameter's search space change."""
    param_name: str
    param_type: str
    old_low: float | None
    old_high: float | None
    new_low: float | None
    new_high: float | None
    best_value: Any
    importance: float
    reason: str


@dataclass
class AgentDecision:
    """Complete agent decision with full reasoning chain."""
    action: str
    round_id: int
    round_number: int
    observation: Observation
    diagnosis: Diagnosis
    search_space_changes: list[SearchSpaceChange]
    justification: str
    proposed_search_space: list[dict] | None
    proposed_budget: int | None
    reference_round_ids: list[int]

    def to_reasoning_dict(self) -> dict:
        """Serialize the full reasoning chain for storage in Postgres."""
        return {
            "observation": {
                "round_id": self.observation.round_id,
                "round_number": self.observation.round_number,
                "metric_name": self.observation.metric_name,
                "direction": self.observation.direction,
                "best_score": self.observation.best_score,
                "round_best_score": self.observation.round_best_score,
                "delta_from_prev": self.observation.delta_from_prev,
                "new_best_in_round": self.observation.new_best_in_round,
                "plateau_signal": self.observation.plateau_signal,
                "generalization_gap": self.observation.generalization_gap,
                "failure_rate": self.observation.failure_rate,
                "trials_added": self.observation.trials_added,
                "round_completed_trials": self.observation.round_completed_trials,
                "top_params": self.observation.top_params,
                "best_params": self.observation.best_params,
            },
            "diagnosis": {
                "is_improving": self.diagnosis.is_improving,
                "improvement_magnitude": self.diagnosis.improvement_magnitude,
                "is_plateauing": self.diagnosis.is_plateauing,
                "is_overfitting": self.diagnosis.is_overfitting,
                "overfitting_severity": self.diagnosis.overfitting_severity,
                "has_failures": self.diagnosis.has_failures,
                "dominant_param": self.diagnosis.dominant_param,
                "dominant_param_importance": self.diagnosis.dominant_param_importance,
                "search_space_exhausted": self.diagnosis.search_space_exhausted,
                "best_params_at_boundary": self.diagnosis.best_params_at_boundary,
                "reasons": self.diagnosis.reasons,
            },
            "search_space_changes": [
                {
                    "param_name": ch.param_name,
                    "param_type": ch.param_type,
                    "old_low": ch.old_low,
                    "old_high": ch.old_high,
                    "new_low": ch.new_low,
                    "new_high": ch.new_high,
                    "best_value": ch.best_value,
                    "importance": ch.importance,
                    "reason": ch.reason,
                }
                for ch in self.search_space_changes
            ],
        }

    def to_proposal(self) -> ActionProposal:
        return ActionProposal(
            action=self.action,
            justification=self.justification,
            proposed_search_space=self.proposed_search_space,
            proposed_budget=self.proposed_budget,
            reference_round_ids=self.reference_round_ids,
            reasoning=self.to_reasoning_dict(),
        )

    def format_report(self) -> str:
        """Format a human-readable decision report."""
        lines = []
        obs = self.observation
        diag = self.diagnosis
        direction_symbol = "↓" if obs.direction == "minimize" else "↑"

        lines.append("=" * 72)
        lines.append(f"  AGENT DECISION after Round {self.round_number}")
        lines.append("=" * 72)

        # --- Observation ---
        lines.append("")
        lines.append("  OBSERVED (from round summary):")
        lines.append(f"    {obs.metric_name}: {_fmt(obs.best_score)} {direction_symbol} (cumulative best)")
        if obs.round_best_score is not None:
            lines.append(f"    Round best: {obs.round_best_score:.6f}")
        if obs.delta_from_prev is not None:
            sign = "+" if obs.delta_from_prev > 0 else ""
            lines.append(f"    Delta from previous round: {sign}{obs.delta_from_prev:.6f}")
        lines.append(f"    New best found this round: {obs.new_best_in_round}")
        lines.append(f"    Trials completed: {obs.round_completed_trials}/{obs.trials_added}")
        lines.append(f"    Plateau signal: {obs.plateau_signal}")
        if obs.generalization_gap is not None:
            lines.append(f"    Generalization gap: {obs.generalization_gap:.6f}")
        lines.append(f"    Failure rate: {obs.failure_rate:.1%}")

        lines.append("")
        lines.append("    Parameter importance (top 5):")
        for name, importance in obs.top_params[:5]:
            marker = " <<<" if name == diag.dominant_param else ""
            lines.append(f"      {name:25s} {importance:6.1%}{marker}")

        lines.append("")
        lines.append("    Best parameters:")
        for k, v in obs.best_params.items():
            if isinstance(v, float):
                lines.append(f"      {k:25s} {v:.6f}")
            else:
                lines.append(f"      {k:25s} {v}")

        # --- Diagnosis ---
        lines.append("")
        lines.append("  DIAGNOSIS:")
        for reason in diag.reasons:
            lines.append(f"    - {reason}")

        # --- Decision ---
        lines.append("")
        lines.append(f"  ACTION: {self.action}")

        # --- Search space changes ---
        if self.search_space_changes:
            lines.append("")
            lines.append("  SEARCH SPACE CHANGES:")
            lines.append(f"    {'Parameter':25s} {'Old Range':>25s}  →  {'New Range':25s} {'Reason'}")
            lines.append(f"    {'─' * 25} {'─' * 25}     {'─' * 25} {'─' * 30}")
            for ch in self.search_space_changes:
                if ch.param_type == "categorical":
                    old_range = "categorical"
                    new_range = "categorical"
                else:
                    old_range = f"[{ch.old_low:.4g}, {ch.old_high:.4g}]"
                    new_range = f"[{ch.new_low:.4g}, {ch.new_high:.4g}]"
                lines.append(f"    {ch.param_name:25s} {old_range:>25s}  →  {new_range:25s} {ch.reason}")

        if self.proposed_budget is not None:
            lines.append(f"\n  NEW BUDGET: {self.proposed_budget} trials")

        # --- Justification ---
        lines.append("")
        lines.append("  JUSTIFICATION:")
        for line in self.justification.split(". "):
            lines.append(f"    {line.strip()}.")

        lines.append("=" * 72)
        return "\n".join(lines)


class AgentReasoner:
    """Analyzes round summaries and produces data-driven decisions."""

    # Thresholds for diagnosis
    OVERFITTING_MILD = 0.05
    OVERFITTING_MODERATE = 0.15
    OVERFITTING_SEVERE = 0.30
    FAILURE_THRESHOLD = 0.1
    DOMINANT_PARAM_THRESHOLD = 0.30
    BOUNDARY_FRACTION = 0.05  # param is "at boundary" if within 5% of range edge

    def _is_at_boundary(self, best_value: float, low: float, high: float) -> bool:
        range_size = high - low if high != low else 1
        if range_size <= 0:
            return False
        return (
            abs(best_value - low) / range_size < self.BOUNDARY_FRACTION
            or abs(best_value - high) / range_size < self.BOUNDARY_FRACTION
        )

    def _narrowing_strategy(self, importance: float) -> tuple[float, str]:
        if importance >= 0.20:
            return 0.4, f"importance={importance:.1%}, focusing tightly"
        if importance >= 0.05:
            return 0.6, f"importance={importance:.1%}, moderate narrowing"
        return 0.8, f"importance={importance:.1%}, slight narrowing"

    def observe(self, summary: RoundSummary, round_number: int) -> Observation:
        """Extract raw signals from a round summary."""
        top_params = sorted(
            summary.param_importance.items(),
            key=lambda x: -x[1]
        )
        return Observation(
            round_id=summary.round_id,
            round_number=round_number,
            metric_name=summary.metric_name,
            direction=summary.objective_direction,
            best_score=summary.best_score,
            round_best_score=summary.round_best_score,
            delta_from_prev=summary.delta_from_prev,
            new_best_in_round=summary.new_best_in_round,
            plateau_signal=summary.plateau_signal,
            generalization_gap=summary.generalization_gap,
            failure_rate=summary.failure_rate,
            pruned_rate=summary.pruned_rate,
            trials_added=summary.trials_added,
            round_completed_trials=summary.round_completed_trials,
            top_params=top_params,
            best_params=summary.best_params or {},
            param_ranges_used=summary.param_ranges_used,
            convergence_curve=summary.convergence_curve,
        )

    def diagnose(self, obs: Observation) -> Diagnosis:
        """Interpret observations into a diagnosis."""
        reasons = []

        # Improvement
        is_improving = obs.new_best_in_round
        if obs.delta_from_prev is not None and obs.delta_from_prev != 0:
            abs_delta = abs(obs.delta_from_prev)
            if obs.best_score and abs_delta / abs(obs.best_score) > 0.01:
                improvement_magnitude = "significant"
                reasons.append(
                    f"Significant improvement: {obs.metric_name} changed by {obs.delta_from_prev:.6f} "
                    f"({abs_delta / abs(obs.best_score):.1%} relative)"
                )
            elif is_improving:
                improvement_magnitude = "small"
                reasons.append(f"Small improvement: delta={obs.delta_from_prev:.6f}")
            else:
                improvement_magnitude = "none"
        elif is_improving:
            improvement_magnitude = "significant"  # first round with results
            reasons.append(f"First round completed: {obs.metric_name}={_fmt(obs.best_score)}")
        else:
            improvement_magnitude = "none"
            reasons.append(f"No improvement: {obs.metric_name} still {_fmt(obs.best_score)}")

        # Plateau
        is_plateauing = obs.plateau_signal
        if is_plateauing:
            reasons.append("Plateau detected: no improvement in last 30% of round trials")

        # Overfitting
        gap = obs.generalization_gap or 0
        if gap >= self.OVERFITTING_SEVERE:
            overfitting_severity = "severe"
            is_overfitting = True
            reasons.append(f"SEVERE overfitting: generalization gap = {gap:.4f}")
        elif gap >= self.OVERFITTING_MODERATE:
            overfitting_severity = "moderate"
            is_overfitting = True
            reasons.append(f"Moderate overfitting: generalization gap = {gap:.4f}")
        elif gap >= self.OVERFITTING_MILD:
            overfitting_severity = "mild"
            is_overfitting = True
            reasons.append(f"Mild overfitting: generalization gap = {gap:.4f}")
        else:
            overfitting_severity = "none"
            is_overfitting = False

        # Failures
        has_failures = obs.failure_rate > self.FAILURE_THRESHOLD
        if has_failures:
            reasons.append(f"High failure rate: {obs.failure_rate:.1%} of trials failed")

        # Dominant param
        dominant_param = None
        dominant_importance = 0.0
        if obs.top_params:
            top_name, top_val = obs.top_params[0]
            if top_val >= self.DOMINANT_PARAM_THRESHOLD:
                dominant_param = top_name
                dominant_importance = top_val
                reasons.append(f"Dominant parameter: {top_name} ({top_val:.1%} importance)")

        # Search space exhaustion
        search_space_exhausted = is_plateauing and not is_improving
        if search_space_exhausted:
            reasons.append("Search space may be exhausted: plateau + no improvement")

        # Boundary detection
        at_boundary = []
        for name, (lo, hi) in obs.param_ranges_used.items():
            best_val = obs.best_params.get(name)
            if best_val is None or not isinstance(best_val, (int, float)):
                continue
            if self._is_at_boundary(float(best_val), lo, hi):
                at_boundary.append(name)
        if at_boundary:
            reasons.append(f"Best params near range boundary: {', '.join(at_boundary)}")

        return Diagnosis(
            is_improving=is_improving,
            improvement_magnitude=improvement_magnitude,
            is_plateauing=is_plateauing,
            is_overfitting=is_overfitting,
            overfitting_severity=overfitting_severity,
            has_failures=has_failures,
            dominant_param=dominant_param,
            dominant_param_importance=dominant_importance,
            search_space_exhausted=search_space_exhausted,
            best_params_at_boundary=at_boundary,
            reasons=reasons,
        )

    def decide(
        self,
        summary: RoundSummary,
        round_number: int,
        current_search_space: list[dict],
        prev_summaries: list[RoundSummary] | None = None,
        prev_decisions: list[dict] | None = None,
        available_params: list[dict] | None = None,
    ) -> AgentDecision:
        """Main entry point: observe, diagnose, decide, justify."""
        obs = self.observe(summary, round_number)
        diag = self.diagnose(obs)

        # Collect all reference round IDs
        ref_ids = [obs.round_id]
        if prev_summaries:
            for ps in prev_summaries[-2:]:  # last 2 previous rounds
                ref_ids.append(ps.round_id)
        ref_ids = list(set(ref_ids))

        # Check recent structural actions for cooldown
        last_structural = None
        if prev_decisions:
            for d in reversed(prev_decisions):
                if d.get("accepted") and d["action"] in ("narrow_search", "widen_search"):
                    last_structural = d["action"]
                    break

        # Decision logic
        action, changes, budget, justification, revised_space = self._choose_action(
            obs, diag, current_search_space, last_structural, prev_summaries or [],
            available_params=available_params,
        )

        proposed_space = revised_space
        if changes:
            proposed_space = self._apply_changes(current_search_space, changes)

        return AgentDecision(
            action=action,
            round_id=obs.round_id,
            round_number=round_number,
            observation=obs,
            diagnosis=diag,
            search_space_changes=changes,
            justification=justification,
            proposed_search_space=proposed_space,
            proposed_budget=budget,
            reference_round_ids=ref_ids,
        )

    def _choose_action(
        self,
        obs: Observation,
        diag: Diagnosis,
        current_space: list[dict],
        last_structural: str | None,
        prev_summaries: list[RoundSummary],
        available_params: list[dict] | None = None,
    ) -> tuple[str, list[SearchSpaceChange], int | None, str, list[dict] | None]:
        """Choose action based on diagnosis. Returns (action, changes, budget, justification, revised_space)."""
        round_ref = f"round {obs.round_number} (id={obs.round_id})"

        # High failure rate → stop (something is broken)
        if diag.has_failures:
            return "stop", [], None, (
                f"After {round_ref}: failure_rate={obs.failure_rate:.1%} exceeds threshold. "
                f"Search space or data configuration may be broken."
            ), None

        # First round → narrow based on observed param importance
        if obs.round_number == 1 and not diag.search_space_exhausted:
            changes = self._build_narrow_changes(obs, current_space)
            if changes:
                reason_parts = [
                    f"After {round_ref}: {obs.metric_name}={_fmt(obs.best_score)}",
                ]
                if diag.dominant_param:
                    reason_parts.append(f"{diag.dominant_param} leads at {diag.dominant_param_importance:.1%} importance")
                if diag.is_plateauing:
                    reason_parts.append("plateau detected — full space has diminishing returns")
                if diag.is_overfitting:
                    reason_parts.append(f"overfitting ({diag.overfitting_severity}): gap={obs.generalization_gap:.4f}")
                reason_parts.append("Narrowing search space around best parameter values to focus exploration")
                return "narrow_search", changes, None, ". ".join(reason_parts), None

        # Improving significantly → continue
        if diag.is_improving and diag.improvement_magnitude == "significant":
            delta_str = f" (delta={obs.delta_from_prev:.6f})" if obs.delta_from_prev is not None else ""
            return "continue", [], None, (
                f"After {round_ref}: {obs.metric_name} improved to {_fmt(obs.best_score)}{delta_str}. "
                f"Significant improvement — continue exploring this search space."
            ), None

        # Improving but plateauing → increase budget
        if diag.is_improving and diag.is_plateauing:
            new_budget = obs.trials_added + 20
            return "increase_budget", [], new_budget, (
                f"After {round_ref}: {obs.metric_name} improved to {_fmt(obs.best_score)} "
                f"but plateau detected. Increasing budget from {obs.trials_added} to {new_budget} "
                f"to give TPE sampler more chances in this space."
            ), None

        # Improving slightly → continue
        if diag.is_improving:
            delta_str = f" (delta={obs.delta_from_prev:.6f})" if obs.delta_from_prev is not None else ""
            return "continue", [], None, (
                f"After {round_ref}: {obs.metric_name} improved to {_fmt(obs.best_score)}{delta_str}. "
                f"Still making progress — continue with same space."
            ), None

        # Not improving + can narrow (no recent structural or cooldown passed)
        if not diag.is_improving and last_structural != "narrow_search":
            no_improve_rounds = self._count_no_improvement(prev_summaries, obs)

            # Multi-round plateau + no dominant param → revise
            if (no_improve_rounds >= 2 and diag.is_plateauing
                    and not diag.dominant_param and available_params):
                revised = self._build_revise_proposal(obs, current_space, available_params)
                if revised:
                    return "revise_search", [], None, (
                        f"After {round_ref}: {obs.metric_name}={_fmt(obs.best_score)} unchanged "
                        f"for {no_improve_rounds} rounds with plateau and no dominant parameter. "
                        f"Revising search space to explore different parameter combinations."
                    ), revised

            if no_improve_rounds >= 2:
                # 2+ rounds no improvement → stop
                return "stop", [], None, (
                    f"After {round_ref}: {obs.metric_name}={_fmt(obs.best_score)} unchanged "
                    f"for {no_improve_rounds} consecutive rounds. Diminishing returns — stopping."
                ), None
            else:
                # Try narrowing
                changes = self._build_narrow_changes(obs, current_space)
                if changes:
                    return "narrow_search", changes, None, (
                        f"After {round_ref}: no improvement ({obs.metric_name}={_fmt(obs.best_score)}). "
                        f"Narrowing search space to focus on promising parameter region."
                    ), None

        # Not improving + already narrowed → check if we should widen or stop
        if not diag.is_improving and last_structural == "narrow_search":
            no_improve_rounds = self._count_no_improvement(prev_summaries, obs)

            # If params hitting boundaries after narrow, widen to escape
            if diag.best_params_at_boundary:
                widen_changes = self._build_widen_changes(obs, current_space)
                if widen_changes:
                    boundary_names = ", ".join(diag.best_params_at_boundary)
                    return "widen_search", widen_changes, None, (
                        f"After {round_ref}: no improvement ({obs.metric_name}={_fmt(obs.best_score)}) "
                        f"after narrowing. "
                        f"Best params at boundary: {boundary_names}. "
                        f"Widening search space to escape boundary constraints."
                    ), None

            if no_improve_rounds >= 2:
                return "stop", [], None, (
                    f"After {round_ref}: {obs.metric_name}={_fmt(obs.best_score)} unchanged "
                    f"for {no_improve_rounds} rounds after narrowing. "
                    f"Search space is exhausted — stopping."
                ), None
            else:
                return "continue", [], None, (
                    f"After {round_ref}: no improvement yet ({obs.metric_name}={_fmt(obs.best_score)}) "
                    f"but only 1 round since narrowing. "
                    f"Continue to give TPE more data in the focused region."
                ), None

        # Default: continue
        return "continue", [], None, (
            f"After {round_ref}: {obs.metric_name}={_fmt(obs.best_score)}. "
            f"Continuing exploration."
        ), None

    def _build_revise_proposal(
        self,
        obs: Observation,
        current_space: list[dict],
        available_params: list[dict],
    ) -> list[dict] | None:
        """Drop low-importance params, add new ones from catalog.

        Returns full proposed search space as list[dict], or None if no valid
        revision is possible.
        """
        MIN_PARAMS = 5
        MAX_CHURN = 3

        importance_map = dict(obs.top_params)
        current_names = {spec["name"] for spec in current_space}

        # Sort current params by importance ascending (lowest first = drop candidates)
        sorted_current = sorted(
            current_space,
            key=lambda s: importance_map.get(s["name"], 0),
        )

        # Identify params eligible to drop (importance < 0.05)
        drop_candidates = [
            s for s in sorted_current
            if importance_map.get(s["name"], 0) < 0.05
        ]

        # Find params from catalog that are not in current space
        add_candidates = [
            p for p in available_params
            if p["name"] not in current_names
        ]

        if not add_candidates:
            return None  # nothing new to add

        # Determine how many to drop (up to 2, but keep at least MIN_PARAMS)
        max_droppable = max(0, len(current_space) - MIN_PARAMS)
        num_drop = min(len(drop_candidates), 2, max_droppable)

        # Determine how many to add (1-2, but total churn <= MAX_CHURN)
        remaining_churn = MAX_CHURN - num_drop
        num_add = min(len(add_candidates), 2, remaining_churn)

        if num_add < 1:
            return None  # must add at least 1

        total_churn = num_drop + num_add
        if total_churn < 1 or total_churn > MAX_CHURN:
            return None

        # Build the new space
        drop_names = {s["name"] for s in drop_candidates[:num_drop]}
        new_space = [dict(spec) for spec in current_space if spec["name"] not in drop_names]
        for p in add_candidates[:num_add]:
            new_space.append(dict(p))

        return new_space

    def _build_narrow_changes(
        self,
        obs: Observation,
        current_space: list[dict],
    ) -> list[SearchSpaceChange]:
        """Build narrowed search space changes based on best params and importance."""
        changes = []
        importance_map = dict(obs.top_params)

        for spec in current_space:
            name = spec["name"]
            ptype = spec.get("type", "float")
            importance = importance_map.get(name, 0)
            best_val = obs.best_params.get(name)

            if ptype == "categorical" or best_val is None:
                continue

            old_low = spec.get("low")
            old_high = spec.get("high")
            if old_low is None or old_high is None:
                continue

            old_range = old_high - old_low
            is_log = spec.get("log", False)

            shrink, reason = self._narrowing_strategy(importance)

            half_new_range = old_range * shrink / 2
            new_low = max(old_low, best_val - half_new_range)
            new_high = min(old_high, best_val + half_new_range)

            # Ensure valid range
            if ptype == "int":
                new_low = int(max(old_low, new_low))
                new_high = int(min(old_high, new_high))
                if new_low >= new_high:
                    new_high = new_low + 1

            if is_log and new_low <= 0:
                new_low = old_low  # preserve log-scale validity

            if new_low < new_high:
                changes.append(SearchSpaceChange(
                    param_name=name,
                    param_type=ptype,
                    old_low=old_low,
                    old_high=old_high,
                    new_low=new_low,
                    new_high=new_high,
                    best_value=best_val,
                    importance=importance,
                    reason=reason,
                ))

        return changes

    def _build_widen_changes(
        self,
        obs: Observation,
        current_space: list[dict],
    ) -> list[SearchSpaceChange]:
        """Build widened search space changes for params where best value is at boundary."""
        changes = []
        importance_map = dict(obs.top_params)

        for spec in current_space:
            name = spec["name"]
            ptype = spec.get("type", "float")
            importance = importance_map.get(name, 0)
            best_val = obs.best_params.get(name)

            if ptype == "categorical" or best_val is None:
                continue

            old_low = spec.get("low")
            old_high = spec.get("high")
            if old_low is None or old_high is None:
                continue

            if not self._is_at_boundary(float(best_val), old_low, old_high):
                continue

            is_log = spec.get("log", False)

            # Expand by 1.5x for high-importance params, 1.3x otherwise
            expand_factor = 1.5 if importance >= 0.20 else 1.3
            old_range = old_high - old_low
            new_half_range = old_range * expand_factor / 2
            midpoint = (old_low + old_high) / 2
            new_low = midpoint - new_half_range
            new_high = midpoint + new_half_range

            # Handle log params: ensure new_low > 0
            if is_log and new_low <= 0:
                new_low = old_low * 0.1

            # Handle int types
            if ptype == "int":
                import math
                new_low = math.floor(new_low)
                new_high = math.ceil(new_high)

            if new_low < new_high:
                reason = f"importance={importance:.1%}, best_value={best_val} at boundary, expanding by {expand_factor}x"
                changes.append(SearchSpaceChange(
                    param_name=name,
                    param_type=ptype,
                    old_low=old_low,
                    old_high=old_high,
                    new_low=new_low,
                    new_high=new_high,
                    best_value=best_val,
                    importance=importance,
                    reason=reason,
                ))

        return changes

    def _apply_changes(
        self,
        current_space: list[dict],
        changes: list[SearchSpaceChange],
    ) -> list[dict]:
        """Apply search space changes and return new space."""
        change_map = {ch.param_name: ch for ch in changes}
        new_space = []
        for spec in current_space:
            name = spec["name"]
            if name in change_map:
                ch = change_map[name]
                new_spec = dict(spec)
                new_spec["low"] = ch.new_low
                new_spec["high"] = ch.new_high
                new_space.append(new_spec)
            else:
                new_space.append(dict(spec))
        return new_space

    def _count_no_improvement(
        self,
        prev_summaries: list[RoundSummary],
        current_obs: Observation,
    ) -> int:
        """Count consecutive rounds with no improvement ending at current."""
        count = 0 if current_obs.new_best_in_round else 1
        for s in reversed(prev_summaries):
            if s.new_best_in_round:
                break
            count += 1
        return count
