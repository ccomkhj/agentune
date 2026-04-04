"""XGBoost backend for agent-hpo."""

from __future__ import annotations

from typing import Callable

import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from agent_hpo.backends.base import suggest_from_param_spec
from agent_hpo.core.models import DatasetSplit, ParamSpec

METRICS = {
    "accuracy": (accuracy_score, False),
    "rmse": (mean_squared_error, False),
    "log_loss": (log_loss, True),
}


class XGBoostBackend:
    def create_objective(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        search_space: list[ParamSpec],
    ) -> Callable[[optuna.Trial], float]:
        metric_fn, needs_proba = METRICS[metric_name]

        def objective(trial: optuna.Trial) -> float:
            params = {spec.name: suggest_from_param_spec(trial, spec) for spec in search_space}
            params["verbosity"] = 0
            params["nthread"] = 1

            model = xgb.XGBClassifier(**params) if metric_name != "rmse" else xgb.XGBRegressor(**params)
            model.fit(dataset.X_train, dataset.y_train, verbose=False)

            if needs_proba:
                y_pred_val = model.predict_proba(dataset.X_val)
                y_pred_train = model.predict_proba(dataset.X_train)
            else:
                y_pred_val = model.predict(dataset.X_val)
                y_pred_train = model.predict(dataset.X_train)

            val_score = metric_fn(dataset.y_val, y_pred_val)
            train_score = metric_fn(dataset.y_train, y_pred_train)

            if metric_name == "rmse":
                val_score = val_score ** 0.5
                train_score = train_score ** 0.5

            trial.set_user_attr(f"train_{metric_name}", float(train_score))

            return float(val_score)

        return objective

    def default_search_space(self) -> list[ParamSpec]:
        return self._param_defs()

    def param_definitions(self) -> list[ParamSpec]:
        return self._param_defs()

    def _param_defs(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="max_depth", type="int", low=1, high=15),
            ParamSpec(name="learning_rate", type="float", low=0.001, high=1.0, log=True),
            ParamSpec(name="n_estimators", type="int", low=50, high=500),
            ParamSpec(name="min_child_weight", type="float", low=1.0, high=10.0),
            ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ParamSpec(name="colsample_bytree", type="float", low=0.5, high=1.0),
            ParamSpec(name="gamma", type="float", low=0.0, high=5.0),
            ParamSpec(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True),
        ]
