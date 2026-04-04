"""CatBoost backend for agentune."""

from __future__ import annotations

from typing import Callable

import catboost as cb
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


class CatBoostBackend:
    def create_objective(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        search_space: list[ParamSpec],
    ) -> Callable[[optuna.Trial], float]:
        metric_fn, needs_proba = METRICS[metric_name]

        def objective(trial: optuna.Trial) -> float:
            params = {spec.name: suggest_from_param_spec(trial, spec) for spec in search_space}
            params["verbose"] = 0
            params["thread_count"] = 1

            if metric_name == "rmse":
                model = cb.CatBoostRegressor(**params)
            else:
                model = cb.CatBoostClassifier(**params)
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
        clean_params = {k: v for k, v in params.items() if k not in ("verbose", "thread_count")}
        clean_params["verbose"] = 0
        clean_params["thread_count"] = 1

        if metric_name == "rmse":
            model = cb.CatBoostRegressor(**clean_params)
        else:
            model = cb.CatBoostClassifier(**clean_params)
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
            ParamSpec(name="depth", type="int", low=4, high=10),
            ParamSpec(name="learning_rate", type="float", low=0.01, high=0.3, log=True),
            ParamSpec(name="iterations", type="int", low=50, high=500),
            ParamSpec(name="l2_leaf_reg", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="random_strength", type="float", low=1e-8, high=10.0, log=True),
            ParamSpec(name="bagging_temperature", type="float", low=0.0, high=10.0),
            ParamSpec(name="border_count", type="int", low=32, high=255),
        ]

    def _all_params(self) -> list[ParamSpec]:
        return self._default_params() + [
            ParamSpec(name="min_data_in_leaf", type="int", low=1, high=100),
            ParamSpec(name="rsm", type="float", low=0.3, high=1.0),
            ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
            ParamSpec(name="max_leaves", type="int", low=8, high=64),
            ParamSpec(name="grow_policy", type="categorical", choices=["SymmetricTree", "Depthwise", "Lossguide"]),
            ParamSpec(name="scale_pos_weight", type="float", low=0.5, high=10.0),
            ParamSpec(name="auto_class_weights", type="categorical", choices=["Balanced", "SqrtBalanced"]),
        ]

    def tuning_guide(self) -> TuningGuide:
        return TuningGuide(
            backend_name="catboost",
            overview=(
                "CatBoost uses ordered boosting and symmetric trees by default, giving strong out-of-box performance. "
                "It handles categorical features natively (not used here since we pre-encode). "
                "Key difference from XGBoost/LightGBM: random_strength and bagging_temperature control stochasticity "
                "instead of subsample/colsample. Symmetric trees mean depth directly controls complexity."
            ),
            params=[
                ParamKnowledge(
                    name="depth",
                    description="Tree depth. CatBoost uses symmetric (balanced) trees, so depth directly determines 2^depth leaves.",
                    role="tree_structure",
                    effect_when_high="2^depth leaves — exponential complexity. depth=10 means 1024 leaves.",
                    effect_when_low="Simple trees, fast training, may underfit",
                    interactions=["l2_leaf_reg (regularization counters depth complexity)", "learning_rate (deeper trees need lower lr)"],
                ),
                ParamKnowledge(
                    name="learning_rate",
                    description="Shrinkage per tree.",
                    role="learning",
                    effect_when_high="Faster training, fewer iterations needed",
                    effect_when_low="Better generalization, needs more iterations",
                    interactions=["iterations (inverse — lower lr needs more)"],
                ),
                ParamKnowledge(
                    name="iterations",
                    description="Number of boosting rounds.",
                    role="learning",
                    effect_when_high="More capacity, slower training",
                    effect_when_low="May underfit with low learning_rate",
                    interactions=["learning_rate (inverse relationship)"],
                ),
                ParamKnowledge(
                    name="l2_leaf_reg",
                    description="L2 regularization on leaf values. CatBoost's primary regularizer.",
                    role="regularization",
                    effect_when_high="Smoother predictions, prevents extreme leaf values, combats overfitting",
                    effect_when_low="Less regularization, sharper predictions",
                    interactions=["depth (deeper trees need more regularization)"],
                ),
                ParamKnowledge(
                    name="random_strength",
                    description="Multiplier for random noise added to split scores. Noise variance decreases during training — early splits get more randomness, later splits are more deterministic.",
                    role="regularization",
                    effect_when_high="More exploration in split selection, prevents overfitting to specific splits",
                    effect_when_low="Greedy split selection, may overfit",
                    interactions=["bagging_temperature (both add stochasticity, but via different mechanisms)"],
                ),
                ParamKnowledge(
                    name="bagging_temperature",
                    description="Controls intensity of Bayesian bootstrap (only with bootstrap_type=Bayesian, the default). 0=equal weights (no effect), 1=standard exponential weights, >1=aggressive reweighting.",
                    role="sampling",
                    effect_when_high="More aggressive sampling, stronger regularization effect",
                    effect_when_low="Less randomness in data selection",
                    interactions=["random_strength (both add noise to reduce overfitting)"],
                ),
                ParamKnowledge(
                    name="border_count",
                    description="Number of histogram bins for numerical features.",
                    role="tree_structure",
                    effect_when_high="More precise splits but slower and can overfit",
                    effect_when_low="Faster, more generalized splits",
                    interactions=["depth (more bins + deeper trees = more overfitting risk)"],
                ),
                ParamKnowledge(
                    name="rsm",
                    description="Random subspace method — fraction of features randomly selected at each split selection (not per-tree, per-split).",
                    role="sampling",
                    effect_when_high="Uses all features for each split",
                    effect_when_low="Feature bagging, helps with noisy high-dimensional data",
                    interactions=["subsample (both reduce overfitting via randomness)"],
                ),
                ParamKnowledge(
                    name="min_data_in_leaf",
                    description="Minimum samples in a leaf. Only works with Depthwise and Lossguide grow_policy, NOT with default SymmetricTree.",
                    role="regularization",
                    effect_when_high="Prevents overfitting on small groups",
                    effect_when_low="Allows fine-grained splits, may overfit on small datasets",
                    interactions=["depth (both control effective tree complexity)", "grow_policy (must be Depthwise or Lossguide for this param to have effect)"],
                ),
            ],
            diagnostics=[
                DiagnosticPattern(
                    signal="Large generalization gap with high depth importance",
                    diagnosis="Overfitting — symmetric trees with high depth create too many leaves",
                    recommended_action="narrow_search: reduce depth range, increase l2_leaf_reg",
                    params_to_adjust=["depth", "l2_leaf_reg"],
                ),
                DiagnosticPattern(
                    signal="Plateau with strong random_strength/bagging_temperature importance",
                    diagnosis="Stochastic params dominate — model is sensitive to noise, not signal",
                    recommended_action="narrow_search: fix stochastic params at moderate values, focus on structural params",
                    params_to_adjust=["random_strength", "bagging_temperature"],
                ),
                DiagnosticPattern(
                    signal="Plateau with no dominant param",
                    diagnosis="Default symmetric tree structure may not suit the data",
                    recommended_action="revise_search: add grow_policy to try Depthwise or Lossguide trees",
                    params_to_adjust=["grow_policy", "max_leaves"],
                ),
                DiagnosticPattern(
                    signal="Small generalization gap but low score",
                    diagnosis="Underfitting — model too regularized or too simple",
                    recommended_action="widen_search: increase depth and iterations, reduce l2_leaf_reg range",
                    params_to_adjust=["depth", "iterations", "l2_leaf_reg"],
                ),
                DiagnosticPattern(
                    signal="Imbalanced dataset",
                    diagnosis="Class weights not tuned",
                    recommended_action="revise_search: add scale_pos_weight or auto_class_weights",
                    params_to_adjust=["scale_pos_weight"],
                ),
            ],
            tuning_order=[
                "learning_rate",
                "depth",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
                "border_count",
                "rsm",
                "min_data_in_leaf",
            ],
        )
