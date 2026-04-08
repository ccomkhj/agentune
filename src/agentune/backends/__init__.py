"""Model backend registry with lazy imports for optional dependencies."""

# Maps backend name -> (module_path, class_name)
_BACKEND_MODULES: dict[str, tuple[str, str]] = {
    "xgboost": ("agentune.backends.xgboost", "XGBoostBackend"),
    "lightgbm": ("agentune.backends.lightgbm", "LightGBMBackend"),
    "catboost": ("agentune.backends.catboost", "CatBoostBackend"),
}

_BACKEND_CACHE: dict[str, type] = {}


def get_backend(name: str) -> type:
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    if name not in _BACKEND_MODULES:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKEND_MODULES.keys())}")
    module_path, class_name = _BACKEND_MODULES[name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _BACKEND_CACHE[name] = cls
    return cls
