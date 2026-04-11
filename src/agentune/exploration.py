"""Coverage-based param selection for exploration resets."""

from __future__ import annotations

from agentune.core.models import ParamSpec

# Params that every backend needs — always included in selections
CORE_PARAM_NAMES = {"learning_rate"}

# Target number of params per exploration reset
TARGET_PARAM_COUNT = 9


def select_exploration_params(backend, rounds: list[dict]) -> list[ParamSpec]:
    """Select params for an exploration reset, prioritizing untried params.

    Always includes core params (learning_rate). Fills remaining slots
    with least-used params from the backend's full catalog.
    """
    catalog = backend.available_params()

    if len(catalog) <= TARGET_PARAM_COUNT:
        return list(catalog)

    # Count how many distinct resets each param appeared in
    seen_resets: dict[str, set[int]] = {}
    for r in rounds:
        search_space = r.get("search_space", [])
        if isinstance(search_space, str):
            import json
            search_space = json.loads(search_space)
        reset_num = r.get("reset_number", 0)
        for p in search_space:
            seen_resets.setdefault(p["name"], set()).add(reset_num)

    usage_count = {p.name: len(seen_resets.get(p.name, set())) for p in catalog}

    # Separate core params from the rest
    core = [p for p in catalog if p.name in CORE_PARAM_NAMES]
    remaining = [p for p in catalog if p.name not in CORE_PARAM_NAMES]

    # Sort by usage count (ascending) — least-used first
    remaining.sort(key=lambda p: usage_count.get(p.name, 0))

    # Fill remaining slots
    slots = TARGET_PARAM_COUNT - len(core)
    selected = core + remaining[:slots]
    return selected
