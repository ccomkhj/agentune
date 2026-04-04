"""Model backend registry."""

BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str) -> type:
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[name]


from agent_hpo.backends.xgboost import XGBoostBackend
register_backend("xgboost", XGBoostBackend)
