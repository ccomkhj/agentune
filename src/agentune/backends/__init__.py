"""Model backend registry."""

BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str) -> type:
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[name]


from agentune.backends.xgboost import XGBoostBackend
from agentune.backends.lightgbm import LightGBMBackend
from agentune.backends.catboost import CatBoostBackend

register_backend("xgboost", XGBoostBackend)
register_backend("lightgbm", LightGBMBackend)
register_backend("catboost", CatBoostBackend)
