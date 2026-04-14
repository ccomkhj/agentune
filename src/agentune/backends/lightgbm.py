"""LightGBM backend for agentune."""

from __future__ import annotations

from typing import Callable

import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from agentune.backends.base import (
    DiagnosticPattern,
    ParamKnowledge,
    TuningGuide,
    suggest_from_param_spec,
)
from agentune.core.models import DatasetSplit, ParamSpec

METRICS = {
    "accuracy": (accuracy_score, False),
    "rmse": (mean_squared_error, False),
    "log_loss": (log_loss, True),
}


class LightGBMBackend:
    def create_objective(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        search_space: list[ParamSpec],
    ) -> Callable[[optuna.Trial], float]:
        metric_fn, needs_proba = METRICS[metric_name]

        def objective(trial: optuna.Trial) -> float:
            params = {spec.name: suggest_from_param_spec(trial, spec) for spec in search_space}
            params["verbosity"] = -1
            params["n_jobs"] = 1

            if metric_name == "rmse":
                model = lgb.LGBMRegressor(**params)
            else:
                model = lgb.LGBMClassifier(**params)
            model.fit(dataset.X_train, dataset.y_train)

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
        return self._default_params()

    def available_params(self) -> list[ParamSpec]:
        return self._all_params()

    def param_definitions(self) -> list[ParamSpec]:
        return self._all_params()

    def evaluate_test(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        params: dict,
    ) -> float:
        metric_fn, needs_proba = METRICS[metric_name]
        clean_params = {k: v for k, v in params.items() if k not in ("verbosity", "n_jobs")}
        clean_params["verbosity"] = -1
        clean_params["n_jobs"] = 1

        if metric_name == "rmse":
            model = lgb.LGBMRegressor(**clean_params)
        else:
            model = lgb.LGBMClassifier(**clean_params)
        model.fit(dataset.X_train, dataset.y_train)

        if needs_proba:
            y_pred = model.predict_proba(dataset.X_test)
        else:
            y_pred = model.predict(dataset.X_test)

        score = metric_fn(dataset.y_test, y_pred)
        if metric_name == "rmse":
            score = score ** 0.5
        return float(score)

    def _default_params(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="num_leaves", type="int", low=8, high=256),
            ParamSpec(name="learning_rate", type="float", low=0.001, high=0.3, log=True),
            ParamSpec(name="n_estimators", type="int", low=50, high=500),
            ParamSpec(name="min_child_samples", type="int", low=5, high=100),
            ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ParamSpec(name="subsample_freq", type="int", low=1, high=7),
            ParamSpec(name="colsample_bytree", type="float", low=0.5, high=1.0),
            ParamSpec(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="max_depth", type="int", low=3, high=15),
        ]

    def _all_params(self) -> list[ParamSpec]:
        return self._default_params() + [
            ParamSpec(name="min_split_gain", type="float", low=0.0, high=5.0),
            ParamSpec(name="max_bin", type="int", low=64, high=512),
            ParamSpec(name="min_child_weight", type="float", low=1e-3, high=10.0, log=True),
            ParamSpec(name="path_smooth", type="float", low=0.0, high=10.0),
            ParamSpec(name="scale_pos_weight", type="float", low=0.5, high=10.0),
            ParamSpec(name="extra_trees", type="categorical", choices=[True, False]),
            ParamSpec(name="feature_fraction_bynode", type="float", low=0.3, high=1.0),
        ]

    def tuning_guide(self) -> TuningGuide:
        return TuningGuide(
            backend_name="lightgbm",
            overview=(
                "LightGBM uses leaf-wise (best-first) tree growth by default, unlike XGBoost's depth-wise. "
                "This means num_leaves is the primary complexity control, not max_depth. "
                "Key tradeoff: num_leaves controls capacity directly — too many leaves overfit, too few underfit. "
                "LightGBM is faster than XGBoost on large datasets due to histogram-based splitting."
            ),
            params=[
                ParamKnowledge(
                    name="num_leaves",
                    description="Maximum number of leaves per tree. THE most important LightGBM param.",
                    role="tree_structure",
                    effect_when_high="More complex trees, captures fine-grained patterns, overfits on small data",
                    effect_when_low="Simpler trees, underfits if data is complex",
                    interactions=["max_depth (use max_depth to prevent deep unbalanced trees; RULE: num_leaves < 2^max_depth)", "min_child_samples (both limit tree growth)"],
                ),
                ParamKnowledge(
                    name="learning_rate",
                    description="Shrinkage rate per tree.",
                    role="learning",
                    effect_when_high="Faster convergence, fewer trees needed, risk of overshooting",
                    effect_when_low="Better generalization, needs more n_estimators",
                    interactions=["n_estimators (inverse — lower lr needs more trees)"],
                ),
                ParamKnowledge(
                    name="n_estimators",
                    description="Number of boosting rounds.",
                    role="learning",
                    effect_when_high="More capacity, may overfit with high learning_rate",
                    effect_when_low="Underfitting if learning_rate is low",
                    interactions=["learning_rate (inverse relationship)"],
                ),
                ParamKnowledge(
                    name="min_child_samples",
                    description="Minimum samples in a leaf. LightGBM's main regularization knob.",
                    role="regularization",
                    effect_when_high="Prevents overfitting on small groups, smoother predictions",
                    effect_when_low="Allows very specific splits, good for large datasets",
                    interactions=["num_leaves (both control tree complexity)"],
                ),
                ParamKnowledge(
                    name="subsample",
                    description="Fraction of rows used per tree (bagging_fraction in LightGBM).",
                    role="sampling",
                    effect_when_high="Uses all data, deterministic",
                    effect_when_low="More randomness, reduces overfitting",
                    interactions=["subsample_freq (must be >0 for subsampling to activate)", "colsample_bytree (both add noise)"],
                ),
                ParamKnowledge(
                    name="colsample_bytree",
                    description="Fraction of features used per tree (feature_fraction).",
                    role="sampling",
                    effect_when_high="Uses all features",
                    effect_when_low="Feature bagging, helps with many noisy features",
                    interactions=["subsample (both reduce overfitting)"],
                ),
                ParamKnowledge(
                    name="reg_alpha",
                    description="L1 regularization on leaf weights.",
                    role="regularization",
                    effect_when_high="Sparser model, some leaf weights driven to zero",
                    effect_when_low="No L1 penalty",
                    interactions=["reg_lambda (L2 complement)"],
                ),
                ParamKnowledge(
                    name="reg_lambda",
                    description="L2 regularization on leaf weights.",
                    role="regularization",
                    effect_when_high="Smoother predictions, prevents extreme leaf weights",
                    effect_when_low="No L2 penalty",
                    interactions=["reg_alpha (L1 complement)"],
                ),
                ParamKnowledge(
                    name="max_depth",
                    description="Max tree depth. Set -1 for no limit (leaf-wise growth controls complexity via num_leaves).",
                    role="tree_structure",
                    effect_when_high="Allows deep, complex trees",
                    effect_when_low="Limits depth, prevents very unbalanced trees. Use with high num_leaves.",
                    interactions=["num_leaves (RULE: set max_depth so num_leaves < 2^max_depth to prevent overfitting)"],
                ),
                ParamKnowledge(
                    name="min_split_gain",
                    description="Minimum gain to make a split (min_gain_to_split).",
                    role="regularization",
                    effect_when_high="Fewer splits, simpler trees (like XGBoost's gamma)",
                    effect_when_low="More splits allowed",
                    interactions=["num_leaves (both control tree complexity)"],
                ),
            ],
            diagnostics=[
                DiagnosticPattern(
                    signal="Large generalization gap with high num_leaves importance",
                    diagnosis="Overfitting — too many leaves for the dataset size",
                    recommended_action="narrow_search: reduce num_leaves range, increase min_child_samples",
                    params_to_adjust=["num_leaves", "min_child_samples"],
                ),
                DiagnosticPattern(
                    signal="Plateau with no dominant param",
                    diagnosis="Current param set may be insufficient",
                    recommended_action="revise_search: add min_split_gain or path_smooth for regularization",
                    params_to_adjust=["min_split_gain", "path_smooth"],
                ),
                DiagnosticPattern(
                    signal="subsample has high importance but subsample_freq not in search space",
                    diagnosis="Subsampling not actually active — LightGBM needs subsample_freq > 0",
                    recommended_action="revise_search: add subsample_freq to search space",
                    params_to_adjust=["subsample_freq"],
                ),
                DiagnosticPattern(
                    signal="Small generalization gap but low score",
                    diagnosis="Underfitting — model too simple",
                    recommended_action="widen_search: increase num_leaves and n_estimators ranges",
                    params_to_adjust=["num_leaves", "n_estimators"],
                ),
                DiagnosticPattern(
                    signal="Imbalanced dataset with poor minority class",
                    diagnosis="Class imbalance not handled",
                    recommended_action="revise_search: add scale_pos_weight",
                    params_to_adjust=["scale_pos_weight"],
                ),
            ],
            tuning_order=[
                "num_leaves",
                "min_child_samples",
                "max_depth",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "learning_rate",
            ],
        )
