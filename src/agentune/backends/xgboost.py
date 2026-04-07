"""XGBoost backend for agentune."""

from __future__ import annotations

from typing import Callable

import optuna
import xgboost as xgb
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
        return self._default_params()

    def available_params(self) -> list[ParamSpec]:
        return self._all_params()

    def param_definitions(self) -> list[ParamSpec]:
        return self._all_params()

    def _default_params(self) -> list[ParamSpec]:
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

    def evaluate_test(
        self,
        dataset: DatasetSplit,
        metric_name: str,
        params: dict,
    ) -> float:
        """Train on train set, evaluate on held-out test set."""
        metric_fn, needs_proba = METRICS[metric_name]
        clean_params = {k: v for k, v in params.items() if k not in ("verbosity", "nthread")}
        clean_params["verbosity"] = 0
        clean_params["nthread"] = 1

        model = xgb.XGBClassifier(**clean_params) if metric_name != "rmse" else xgb.XGBRegressor(**clean_params)
        model.fit(dataset.X_train, dataset.y_train, verbose=False)

        if needs_proba:
            y_pred = model.predict_proba(dataset.X_test)
        else:
            y_pred = model.predict(dataset.X_test)

        score = metric_fn(dataset.y_test, y_pred)
        if metric_name == "rmse":
            score = score ** 0.5
        return float(score)

    def _all_params(self) -> list[ParamSpec]:
        return self._default_params() + [
            ParamSpec(name="max_leaves", type="int", low=0, high=256),
            ParamSpec(name="max_bin", type="int", low=32, high=1024),
            ParamSpec(name="colsample_bylevel", type="float", low=0.3, high=1.0),
            ParamSpec(name="colsample_bynode", type="float", low=0.3, high=1.0),
            ParamSpec(name="scale_pos_weight", type="float", low=0.1, high=100.0, log=True),
            ParamSpec(name="grow_policy", type="categorical", choices=["depthwise", "lossguide"]),
            ParamSpec(name="max_delta_step", type="float", low=0.0, high=10.0),
            ParamSpec(name="num_parallel_tree", type="int", low=1, high=5),
            ParamSpec(name="max_cat_to_onehot", type="int", low=1, high=64),
            ParamSpec(name="max_cat_threshold", type="int", low=1, high=256),
        ]

    def tuning_guide(self) -> TuningGuide:
        return TuningGuide(
            backend_name="xgboost",
            overview=(
                "XGBoost builds trees sequentially, each correcting the previous. "
                "Key tradeoff: model capacity (depth, estimators) vs regularization (gamma, lambda, alpha). "
                "Fix learning_rate at ~0.1, tune max_depth + min_child_weight together, then gamma, "
                "then sampling (subsample, colsample_bytree), then reg_alpha/reg_lambda, then lower learning_rate last."
            ),
            params=[
                ParamKnowledge(
                    name="learning_rate",
                    description="Shrinkage applied to each tree's contribution. Lower = more trees needed but better generalization.",
                    role="learning",
                    effect_when_high="Faster convergence but risk overshooting; fewer trees needed",
                    effect_when_low="Better generalization but needs more n_estimators; slower training",
                    interactions=["n_estimators (inverse relationship — lower lr needs more trees)"],
                ),
                ParamKnowledge(
                    name="max_depth",
                    description="Maximum tree depth. Controls model complexity directly.",
                    role="tree_structure",
                    effect_when_high="More complex trees, captures interactions but overfits on small data",
                    effect_when_low="Simpler trees, underfits if data has complex interactions",
                    interactions=["min_child_weight (both limit tree growth)", "gamma (pruning complements depth limit)"],
                ),
                ParamKnowledge(
                    name="n_estimators",
                    description="Number of boosting rounds (trees).",
                    role="learning",
                    effect_when_high="More capacity, risk of overfitting if lr is too high",
                    effect_when_low="Underfitting, especially with low learning_rate",
                    interactions=["learning_rate (inverse — lower lr needs more estimators)"],
                ),
                ParamKnowledge(
                    name="min_child_weight",
                    description="Minimum sum of instance weight (hessian) in a child node. For classification, hessian = p*(1-p), not sample count.",
                    role="regularization",
                    effect_when_high="More conservative splits, prevents overfitting on noisy features",
                    effect_when_low="Allows very specific splits, good for imbalanced data",
                    interactions=["max_depth (both control tree complexity)"],
                ),
                ParamKnowledge(
                    name="gamma",
                    description="Minimum loss reduction required to make a split. Pre-pruning: prevents splits during tree construction if gain is below gamma.",
                    role="regularization",
                    effect_when_high="Aggressive pruning, simpler trees",
                    effect_when_low="More splits allowed, more complex trees",
                    interactions=["max_depth (gamma prunes what depth allows)", "reg_alpha/reg_lambda (all regularize)"],
                ),
                ParamKnowledge(
                    name="subsample",
                    description="Fraction of training rows used per tree. Stochastic gradient boosting.",
                    role="sampling",
                    effect_when_high="Uses all data, deterministic but may overfit",
                    effect_when_low="More randomness, reduces overfitting, but noisy gradients",
                    interactions=["colsample_bytree (both add stochasticity)"],
                ),
                ParamKnowledge(
                    name="colsample_bytree",
                    description="Fraction of features used per tree.",
                    role="sampling",
                    effect_when_high="Uses all features, may overfit on noisy features",
                    effect_when_low="Feature bagging, reduces overfitting on high-dimensional data",
                    interactions=["subsample (both reduce overfitting via randomness)", "colsample_bylevel, colsample_bynode (multiplicative: features_per_split = total * bytree * bylevel * bynode)"],
                ),
                ParamKnowledge(
                    name="reg_alpha",
                    description="L1 regularization on leaf weights. Encourages sparsity.",
                    role="regularization",
                    effect_when_high="Sparser model, drives some leaf weights to zero",
                    effect_when_low="No L1 penalty, all features contribute",
                    interactions=["reg_lambda (L2 complement)", "gamma (all three regularize)"],
                ),
                ParamKnowledge(
                    name="reg_lambda",
                    description="L2 regularization on leaf weights. Smooths predictions.",
                    role="regularization",
                    effect_when_high="Smoother, more conservative predictions",
                    effect_when_low="Less smoothing, sharper predictions",
                    interactions=["reg_alpha (L1 complement)"],
                ),
                ParamKnowledge(
                    name="scale_pos_weight",
                    description="Balance between positive and negative class weights.",
                    role="sampling",
                    effect_when_high="Upweights positive class, helps with imbalanced data",
                    effect_when_low="Equal weighting (default=1); values <1 downweight positive class",
                    interactions=[],
                ),
                ParamKnowledge(
                    name="max_leaves",
                    description="Maximum number of leaf nodes (used with grow_policy=lossguide).",
                    role="tree_structure",
                    effect_when_high="More complex trees with many leaves",
                    effect_when_low="Simpler trees, acts like depth limit",
                    interactions=["grow_policy (only effective with lossguide)", "max_depth (alternative complexity control)"],
                ),
                ParamKnowledge(
                    name="max_bin",
                    description="Maximum number of discrete bins for histogram-based splitting.",
                    role="tree_structure",
                    effect_when_high="Finer split granularity, better accuracy on continuous features, slower",
                    effect_when_low="Coarser splits, faster training, acts as regularization",
                    interactions=["tree_method (only matters for hist/approx)"],
                ),
                ParamKnowledge(
                    name="colsample_bylevel",
                    description="Fraction of features sampled per tree depth level. Multiplicative with colsample_bytree.",
                    role="sampling",
                    effect_when_high="All features available at each level",
                    effect_when_low="Feature bagging per level, reduces correlation between splits",
                    interactions=["colsample_bytree, colsample_bynode (multiplicative: bytree * bylevel * bynode)"],
                ),
                ParamKnowledge(
                    name="colsample_bynode",
                    description="Fraction of features sampled per split. Finest-grained feature subsampling.",
                    role="sampling",
                    effect_when_high="All features available per split",
                    effect_when_low="Random feature selection per split, like random forest behavior",
                    interactions=["colsample_bytree, colsample_bylevel (multiplicative)"],
                ),
                ParamKnowledge(
                    name="max_delta_step",
                    description="Maximum delta step for each leaf output. Helps with extremely imbalanced classification.",
                    role="regularization",
                    effect_when_high="Clips leaf outputs more aggressively, stabilizes updates",
                    effect_when_low="No clipping (default=0); leaf outputs unconstrained",
                    interactions=["scale_pos_weight (both help with imbalanced data)"],
                ),
                ParamKnowledge(
                    name="num_parallel_tree",
                    description="Number of parallel trees per boosting round. Values >1 create a boosted random forest.",
                    role="tree_structure",
                    effect_when_high="Ensemble of trees per round, stronger regularization, slower",
                    effect_when_low="Standard single-tree boosting (default=1)",
                    interactions=["n_estimators (total trees = n_estimators * num_parallel_tree)"],
                ),
                ParamKnowledge(
                    name="max_cat_to_onehot",
                    description="Threshold for one-hot vs partition-based splits on categorical features.",
                    role="tree_structure",
                    effect_when_high="More categories use one-hot (simpler but combinatorially limited)",
                    effect_when_low="More categories use partition-based splits (can find complex groupings)",
                    interactions=["max_cat_threshold (controls partition complexity for high-cardinality categoricals)"],
                ),
                ParamKnowledge(
                    name="max_cat_threshold",
                    description="Maximum categories considered per partition-based split. Limits complexity for high-cardinality features.",
                    role="tree_structure",
                    effect_when_high="Considers more category groupings, better accuracy, slower",
                    effect_when_low="Fewer groupings considered, faster, may miss optimal splits",
                    interactions=["max_cat_to_onehot (determines which features use partition-based splits)"],
                ),
            ],
            diagnostics=[
                DiagnosticPattern(
                    signal="Large generalization gap (train >> val) and high max_depth importance",
                    diagnosis="Overfitting due to tree complexity",
                    recommended_action="narrow_search: reduce max_depth range, increase gamma and min_child_weight",
                    params_to_adjust=["max_depth", "gamma", "min_child_weight"],
                ),
                DiagnosticPattern(
                    signal="Large generalization gap with high subsample/colsample importance",
                    diagnosis="Overfitting due to memorization — not enough stochasticity",
                    recommended_action="narrow_search: lower subsample and colsample_bytree ranges",
                    params_to_adjust=["subsample", "colsample_bytree"],
                ),
                DiagnosticPattern(
                    signal="Plateau with no dominant param (all <15% importance)",
                    diagnosis="Current param set may be wrong. Model is insensitive to all tuned params.",
                    recommended_action="revise_search: swap low-importance params for regularization or structural params",
                    params_to_adjust=[],
                ),
                DiagnosticPattern(
                    signal="Best learning_rate near lower bound AND n_estimators near upper bound",
                    diagnosis="Learning rate too low for the budget — not enough trees to converge",
                    recommended_action="widen_search: increase n_estimators upper bound, or increase_budget",
                    params_to_adjust=["n_estimators", "learning_rate"],
                ),
                DiagnosticPattern(
                    signal="Score improving but plateau in late trials of each round",
                    diagnosis="TPE needs more trials to exploit promising regions",
                    recommended_action="increase_budget: give TPE more trials per round",
                    params_to_adjust=[],
                ),
                DiagnosticPattern(
                    signal="Small generalization gap but low absolute score",
                    diagnosis="Underfitting — model too simple for the data",
                    recommended_action="widen_search: increase max_depth, n_estimators ranges; reduce regularization",
                    params_to_adjust=["max_depth", "n_estimators", "gamma", "reg_alpha", "reg_lambda"],
                ),
                DiagnosticPattern(
                    signal="Imbalanced dataset with poor minority class performance",
                    diagnosis="Class imbalance not addressed",
                    recommended_action="revise_search: add scale_pos_weight to the search space",
                    params_to_adjust=["scale_pos_weight"],
                ),
            ],
            tuning_order=[
                "max_depth",
                "min_child_weight",
                "gamma",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "learning_rate",
            ],
        )
