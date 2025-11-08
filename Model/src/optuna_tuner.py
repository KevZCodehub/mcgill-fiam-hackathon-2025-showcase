"""
Optuna tuning module for the stock return model (regression + rank-aware evaluation).

Features:
---------
1. Multi-fidelity:
   - Trials use fewer trees (multi_fidelity_n_estimators_trial).
   - Best params optionally refit with full n_estimators (final_n_estimators).

2. Adaptive skip:
   - Skip tuning if recent improvement below threshold.

3. Trial logging:
   - CSV log with objective, IC stats, Sharpe, monthly IC & LS distributions (JSON-encoded).

4. Global warm-up & refinement:
   - Global tuning across sampled windows.
   - Relative search narrowing around previous best.

5. Cross-sectional preprocessing:
   - (Optional) Month-constant feature removal.
   - (Optional) Monthly cross-sectional z-scoring.
   - (Optional) Target clipping + smoothing.

6. Rank-oriented early stopping:
   - Custom monthly Spearman IC feval controlling early stopping (still regression objective).

7. Objectives:
   - spearman_ic_mean
   - sharpe_long_short
   - hybrid_ic_sharpe

Pluggable extensions:
---------------------
- Add regime-specific objectives.
- Add stability-penalized objective.
- Add sample weighting (liquidity / market-cap) if needed.

Author: Adapted per user requirements (2025).
"""

from __future__ import annotations

import os
import json
import time
import math
import random
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from scipy.stats import spearmanr
from lightgbm import log_evaluation, early_stopping

logger = logging.getLogger(__name__)

# Import preprocessing helpers (ensure feature_preprocess.py exists)
from feature_preprocess import (
    remove_month_constant,
    cross_sectional_z,
    clip_and_smooth_return
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# JSON Sanitization
# ------------------------------------------------------------------
def _json_sanitize(obj):
    """
    Recursively convert objects (NumPy / pandas types) into JSON-serializable
    Python native types.
    """
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str)) or obj is None:
        return obj
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (_pd.Series, _pd.Index)):
        return [_json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, _pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, _np.ndarray):
        return [_json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, list):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    try:
        return str(obj)
    except Exception:
        return "UNSERIALIZABLE"


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass
class TuningConfig:
    # Objective control
    objective_name: str = "spearman_ic_mean"   # spearman_ic_mean | sharpe_long_short | hybrid_ic_sharpe
    hybrid_weight_ic: float = 0.6
    direction: str = "maximize"

    # Trial budgeting
    n_trials_per_window: int = 10
    timeout_per_window: Optional[int] = None
    n_startup_trials: int = 5
    enable_pruning: bool = True

    # Multi-fidelity
    multi_fidelity_n_estimators_trial: int = 400
    final_n_estimators: int = 1400
    refit_final_by_default: bool = False

    # Early stopping rounds (passed to LightGBM)
    early_stopping_rounds: int = 120

    # Portfolio / evaluation settings
    quantile_bins: int = 10
    min_cross_section: int = 15
    min_months_for_eval: int = 6
    annualize_sharpe: bool = True

    # Preprocessing toggles
    remove_month_constants: bool = True
    monthly_zscore: bool = True
    target_clip: bool = True
    z_clip: float = 5.0
    smooth_scale: float = 0.05

    # Rank-aware feval
    use_monthly_spearman_feval: bool = True
    min_months_for_feval: int = 4

    # Adaptive skip settings
    adaptive_skip: bool = True
    adaptive_window: int = 3
    adaptive_delta_threshold: float = 0.002
    min_windows_before_skip: int = 2

    # Relative search narrowing
    relative_search_narrowing: bool = True
    relative_range_factors: Dict[str, float] = field(default_factory=lambda: {
        "num_leaves": 0.5,
        "learning_rate": 0.5,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.3,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 0.5,
        "max_depth": 0.5
    })

    # Base static params (merged after tuning)
    base_static_params: Dict[str, Any] = field(default_factory=lambda: {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbosity": -1,
        "metric": "l2",
        "n_jobs": 14,
        "random_state": 42
    })

    # Random
    random_state: int = 42

    # Logging
    trial_log_path: str = "optuna_trial_log.csv"

    # GPU
    use_gpu: bool = False

    # Fail-safe
    allow_nan_objective: bool = False


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_monthly_ic(df: pd.DataFrame,
                       pred_col: str,
                       ret_col: str,
                       min_cross_section: int) -> List[float]:
    ics = []
    for date, g in df.groupby("char_eom", observed=True):
        if len(g) < min_cross_section:
            continue
        if g[pred_col].nunique() <= 1 or g[ret_col].nunique() <= 1:
            continue
        ic, _ = spearmanr(g[pred_col], g[ret_col])
        if not np.isnan(ic):
            ics.append(ic)
    return ics


def compute_long_short_returns(df: pd.DataFrame,
                               pred_col: str,
                               ret_col: str,
                               quantile_bins: int,
                               min_cross_section: int) -> List[float]:
    ls = []
    for date, g in df.groupby("char_eom", observed=True):
        if len(g) < quantile_bins * 5:  # ensure quantile stability
            continue
        g = g.copy()
        try:
            g["bin"] = pd.qcut(g[pred_col], q=quantile_bins, labels=False, duplicates="drop")
        except Exception:
            continue
        if g["bin"].nunique() < quantile_bins:
            continue
        top_ret = g[g["bin"] == g["bin"].max()][ret_col].mean()
        bot_ret = g[g["bin"] == g["bin"].min()][ret_col].mean()
        if np.isfinite(top_ret) and np.isfinite(bot_ret):
            ls.append(top_ret - bot_ret)
    return ls


def compute_long_short_sharpe_from_series(series: List[float],
                                          annualize: bool) -> Optional[float]:
    if len(series) == 0:
        return None
    arr = np.array(series, dtype=float)
    if arr.std(ddof=1) == 0:
        return None
    sharpe_m = arr.mean() / arr.std(ddof=1)
    return sharpe_m * math.sqrt(12) if annualize else sharpe_m


def make_val_spearman_feval(month_index_array: np.ndarray,
                            min_obs: int,
                            min_cs_per_month: int = 30):
    """
    Create LightGBM custom feval returning average monthly Spearman IC.
    Only months with at least min_cs_per_month observations & variance are used.
    """
    month_index_array = np.asarray(month_index_array)

    def feval(preds, dataset):
        y = dataset.get_label()
        ic_list = []
        for m in np.unique(month_index_array):
            mask = month_index_array == m
            if mask.sum() < min_cs_per_month:
                continue
            if np.unique(preds[mask]).size <= 2:
                continue
            ic, _ = spearmanr(preds[mask], y[mask])
            if np.isfinite(ic):
                ic_list.append(ic)
        if len(ic_list) < min_obs:
            score = 0.0
        else:
            score = float(np.mean(ic_list))
        return "monthly_spearman", score, True

    return feval


# ------------------------------------------------------------------
# Search Spaces
# ------------------------------------------------------------------
def default_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "l2",
        "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.03, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 256, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.85),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 6),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.05, 5.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 0.1, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-5, 1e-2, log=True),
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": 14
    }


def narrowed_search_space(trial: optuna.Trial,
                          prev_best: Dict[str, Any],
                          factors: Dict[str, float]) -> Dict[str, Any]:
    space = {}

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def int_range(key, lo, hi, log=False):
        base = prev_best[key]
        span = max(1, int(base * factors.get(key, 0.3)))
        low = clamp(base - span, lo, hi)
        high = clamp(base + span, lo, hi)
        if low == high:
            high = min(hi, low + 1)
        return trial.suggest_int(key, low, high, log=log)

    def float_range(key, lo, hi, log=False):
        base = prev_best[key]
        span = base * factors.get(key, 0.3)
        low = clamp(base - span, lo, hi)
        high = clamp(base + span, lo, hi)
        if low == high:
            high = min(hi, low + (0.0005 if not log else low * 0.05))
        return trial.suggest_float(key, low, high, log=log)

    space["num_leaves"] = int_range("num_leaves", 16, 512, log=True)
    space["learning_rate"] = float_range("learning_rate", 0.003, 0.15, log=True)
    space["feature_fraction"] = float_range("feature_fraction", 0.3, 0.95)
    space["bagging_fraction"] = float_range("bagging_fraction", 0.3, 1.0)
    space["bagging_freq"] = int_range("bagging_freq", 1, 12)
    space["lambda_l1"] = float_range("lambda_l1", 1e-4, 100.0, log=True)
    space["lambda_l2"] = float_range("lambda_l2", 1e-4, 150.0, log=True)
    space["min_data_in_leaf"] = int_range("min_data_in_leaf", 10, 600, log=True)
    space["max_depth"] = int_range("max_depth", 4, 24)
    return space


# ------------------------------------------------------------------
# Main Tuner
# ------------------------------------------------------------------
class OptunaWindowTuner:
    def __init__(self,
                 tuning_config: TuningConfig,
                 base_params: Optional[Dict[str, Any]] = None):
        self.cfg = tuning_config
        self.previous_best_params: Optional[Dict[str, Any]] = base_params.copy() if base_params else None
        self.history: List[Dict[str, Any]] = []
        self.global_studies: List[optuna.Study] = []
        self.window_objective_trace: List[float] = []
        self.last_objective_value: Optional[float] = None

        random.seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)

        # Trial log setup
        if self.cfg.trial_log_path and not os.path.exists(self.cfg.trial_log_path):
            with open(self.cfg.trial_log_path, "w", encoding="utf-8") as f:
                header = [
                    "study_type", "window_id", "trial", "objective_value",
                    "ic_mean", "ic_std", "sharpe", "months_used",
                    "duration_sec", "params_json",
                    "monthly_ic_json", "monthly_ls_json"
                ]
                f.write(",".join(header) + "\n")

    # -----------------------------
    # Adaptive window skipping
    # -----------------------------
    def should_tune_window(self) -> bool:
        if not self.cfg.adaptive_skip:
            return True
        n = len(self.window_objective_trace)
        if n < self.cfg.min_windows_before_skip:
            return True
        if n < self.cfg.adaptive_window + 1:
            return True
        recent = self.window_objective_trace[-(self.cfg.adaptive_window + 1):]
        deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        max_improve = max(deltas)
        if max_improve < self.cfg.adaptive_delta_threshold:
            logger.info(f"[AdaptiveSkip] Skipping tuning (max recent Δ={max_improve:.4f} < "
                        f"{self.cfg.adaptive_delta_threshold})")
            return False
        return True

    # -----------------------------
    # Global warm-up
    # -----------------------------
    def global_warmup(self,
                      sampled_windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
                      features: List[str],
                      sectors: List[str],
                      target: str,
                      n_trials: int = 40,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        logger.info(f"[Warmup] Global warm-up on {len(sampled_windows)} windows, trials={n_trials}")
        params = self.tune_global(
            sampled_windows=sampled_windows,
            features=features,
            sectors=sectors,
            target=target,
            n_trials=n_trials,
            timeout=timeout
        )
        logger.info("[Warmup] Completed.")
        return params

    # -----------------------------
    # Per-window tuning
    # -----------------------------
    def tune_window(self,
                    window_id: int,
                    train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    features: List[str],
                    sectors: List[str],
                    target: str,
                    refit_full: Optional[bool] = None) -> Dict[str, Any]:
        logger.info(f"[WindowTuning] Window {window_id} objective={self.cfg.objective_name}")

        study = optuna.create_study(
            direction=self.cfg.direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.cfg.random_state,
                n_startup_trials=self.cfg.n_startup_trials
            ),
            pruner=optuna.pruners.MedianPruner() if self.cfg.enable_pruning else optuna.pruners.NopPruner()
        )

        def objective(trial: optuna.Trial):
            sampled_params = self._sample_params(trial)
            score, diag = self._train_and_eval(
                params=sampled_params,
                train_df=train_df.copy(),
                val_df=val_df.copy(),
                features=features,
                sectors=sectors,
                target=target,
                n_estimators=self.cfg.multi_fidelity_n_estimators_trial
            )
            if score is None or (not self.cfg.allow_nan_objective and not np.isfinite(score)):
                raise optuna.TrialPruned()
            self._log_trial(
                study_type="window",
                window_id=window_id,
                trial=trial.number,
                objective_value=score,
                diagnostics=diag,
                params=sampled_params
            )
            return score

        study.optimize(
            objective,
            n_trials=self.cfg.n_trials_per_window,
            timeout=self.cfg.timeout_per_window,
            gc_after_trial=True,
            show_progress_bar=False
        )

        best_params = study.best_params
        best_params_full = self._assemble_final_params(best_params, full=True)

        if refit_full is None:
            refit_full = self.cfg.refit_final_by_default

        if refit_full:
            logger.info("[WindowTuning] Re-fitting best params with full n_estimators.")
            self._refit_full_model(train_df, val_df, features, sectors, target, best_params_full)

        self.previous_best_params = best_params_full
        self.history.append({
            "scope": "window",
            "window_id": window_id,
            "best_value": study.best_value,
            "best_params": best_params_full,
            "objective": self.cfg.objective_name,
            "n_trials": len(study.trials)
        })
        self.window_objective_trace.append(study.best_value)
        self.last_objective_value = study.best_value

        logger.info(f"[WindowTuning] Window {window_id} best value={study.best_value:.6f}")
        return best_params_full

    # -----------------------------
    # Global tuning
    # -----------------------------
    def tune_global(self,
                    sampled_windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
                    features: List[str],
                    sectors: List[str],
                    target: str,
                    n_trials: int = 100,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        logger.info(f"[Global] Tuning across {len(sampled_windows)} sampled windows.")

        study = optuna.create_study(
            direction=self.cfg.direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.cfg.random_state,
                n_startup_trials=max(self.cfg.n_startup_trials, 8)
            ),
            pruner=optuna.pruners.MedianPruner() if self.cfg.enable_pruning else optuna.pruners.NopPruner()
        )

        def objective(trial: optuna.Trial):
            sampled_params = self._sample_params(trial, global_mode=True)
            win_scores = []
            agg_ic = []
            agg_ls = []
            start_time = time.time()

            for (tr_df, va_df) in sampled_windows:
                score, diag = self._train_and_eval(
                    params=sampled_params,
                    train_df=tr_df.copy(),
                    val_df=va_df.copy(),
                    features=features,
                    sectors=sectors,
                    target=target,
                    n_estimators=self.cfg.multi_fidelity_n_estimators_trial
                )
                if score is not None and np.isfinite(score):
                    win_scores.append(score)
                if diag:
                    agg_ic.extend(diag.get("monthly_ic", []))
                    agg_ls.extend(diag.get("monthly_ls", []))

            if len(win_scores) == 0:
                raise optuna.TrialPruned()

            mean_score = float(np.mean(win_scores))
            diag_global = {
                "ic_mean": np.mean(agg_ic) if agg_ic else np.nan,
                "ic_std": np.std(agg_ic, ddof=1) if len(agg_ic) > 1 else np.nan,
                "sharpe": compute_long_short_sharpe_from_series(agg_ls, self.cfg.annualize_sharpe) if agg_ls else np.nan,
                "months_used": len(agg_ic)
            }
            self._log_trial(
                study_type="global",
                window_id=-1,
                trial=trial.number,
                objective_value=mean_score,
                diagnostics=diag_global,
                params=sampled_params,
                duration_sec=time.time() - start_time
            )
            return mean_score

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
            gc_after_trial=True
        )

        best_params = study.best_params
        best_params_full = self._assemble_final_params(best_params, full=True)
        self.previous_best_params = best_params_full
        self.global_studies.append(study)
        self.history.append({
            "scope": "global",
            "best_value": study.best_value,
            "best_params": best_params_full,
            "objective": self.cfg.objective_name,
            "n_trials": len(study.trials)
        })
        logger.info(f"[Global] Best global value={study.best_value:.6f}")
        return best_params_full

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _sample_params(self, trial: optuna.Trial, global_mode: bool = False) -> Dict[str, Any]:
        if self.previous_best_params and self.cfg.relative_search_narrowing and not global_mode:
            try:
                narrowed = narrowed_search_space(trial, self.previous_best_params, self.cfg.relative_range_factors)
                return narrowed
            except KeyError:
                return default_search_space(trial)
        return default_search_space(trial)

    def _assemble_final_params(self, tuned: Dict[str, Any], full: bool) -> Dict[str, Any]:
        params = tuned.copy()
        # merge static (do not overwrite tuned)
        for k, v in self.cfg.base_static_params.items():
            params.setdefault(k, v)
        params["random_state"] = self.cfg.random_state
        params["n_estimators"] = self.cfg.final_n_estimators if full else self.cfg.multi_fidelity_n_estimators_trial
        if self.cfg.use_gpu:
            params["device_type"] = "gpu"
            params.setdefault("max_bin", 255)
        return params

    def _train_and_eval(self,
                        params: Dict[str, Any],
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        features: List[str],
                        sectors: List[str],
                        target: str,
                        n_estimators: int) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        start = time.time()

        # Ensure categorical
        for s in sectors:
            if s in train_df.columns:
                train_df[s] = train_df[s].astype("category")
                val_df[s] = val_df[s].astype("category")

        # Month-constant removal (train only)
        removed_constants = []
        final_features = features.copy()
        if self.cfg.remove_month_constants:
            final_features, removed_constants = remove_month_constant(train_df, final_features)

        # Separate numeric & categorical
        categorical = [c for c in sectors if c in train_df.columns]
        numeric = [f for f in final_features if f not in categorical and f in train_df.columns]

        # Cross-sectional z-score
        if self.cfg.monthly_zscore and numeric:
            train_df = cross_sectional_z(train_df, numeric)
            val_df = cross_sectional_z(val_df, numeric)

        # Target clipping/smoothing
        if self.cfg.target_clip and target in train_df.columns:
            train_df = clip_and_smooth_return(train_df,
                                              target=target,
                                              z_clip=self.cfg.z_clip,
                                              smooth_scale=self.cfg.smooth_scale)

        # If no features left
        feature_cols = numeric + categorical
        if not feature_cols:
            return None, None

        # Build LightGBM datasets
        train_month_index = train_df['char_eom'].factorize()[0]
        val_month_index = val_df['char_eom'].factorize()[0]

        train_ds = lgb.Dataset(train_df[feature_cols],
                               label=train_df[target],
                               free_raw_data=False,
                               categorical_feature=categorical)
        val_ds = lgb.Dataset(val_df[feature_cols],
                             label=val_df[target],
                             free_raw_data=False,
                             categorical_feature=categorical)

        # Prepare parameters
        trial_params = params.copy()
        trial_params.update({
            "n_estimators": n_estimators
        })
        # Remove n_estimators for native lgb.train usage
        num_boost_round = trial_params.pop("n_estimators", n_estimators)

        feval = None
        if self.cfg.use_monthly_spearman_feval and self.cfg.objective_name.startswith("spearman"):
            feval = make_val_spearman_feval(
                month_index_array=val_month_index,
                min_obs=self.cfg.min_months_for_feval,
                min_cs_per_month=self.cfg.min_cross_section
            )

        # Train
        try:
            model = lgb.train(
                trial_params,
                train_ds,
                num_boost_round=num_boost_round,
                valid_sets=[val_ds],
                feval=feval,
                callbacks=[
                    log_evaluation(period=0), early_stopping(self.cfg.early_stopping_rounds)
                ]
            )
        except Exception as e:
            logger.debug(f"[TrainFail] {e}")
            return None, None

        # Predict
        preds = model.predict(
            val_df[feature_cols],
            num_iteration=getattr(model, "best_iteration", None)
        )

        val_eval = val_df.copy()
        val_eval["prediction"] = preds
        logger.info(
            f"[Diag] val months={val_eval['char_eom'].nunique()} "
            f"rows={len(val_eval)}"
            f"pred_unique_monthly="
            f"{val_eval.groupby('char_eom')['prediction'].nunique().describe().to_dict()}"
        )
        # Compute objective & diagnostics
        objective_value, diagnostics = self._compute_objective(
            val_eval, pred_col="prediction", target_col=target
        )
        diagnostics["duration_sec"] = time.time() - start
        diagnostics["removed_constants"] = removed_constants
        diagnostics["best_iteration"] = getattr(model, "best_iteration", None)
        return objective_value, diagnostics

    def _compute_objective(self,
                           val_df: pd.DataFrame,
                           pred_col: str,
                           target_col: str) -> Tuple[Optional[float], Dict[str, Any]]:
        name = self.cfg.objective_name

        monthly_ic = compute_monthly_ic(
            val_df,
            pred_col=pred_col,
            ret_col=target_col,
            min_cross_section=self.cfg.min_cross_section
        )

        ls_returns = compute_long_short_returns(
            val_df,
            pred_col=pred_col,
            ret_col=target_col,
            quantile_bins=self.cfg.quantile_bins,
            min_cross_section=self.cfg.min_cross_section
        )

        ic_mean = np.mean(monthly_ic) if monthly_ic else np.nan
        ic_std = np.std(monthly_ic, ddof=1) if len(monthly_ic) > 1 else np.nan
        sharpe_val = compute_long_short_sharpe_from_series(ls_returns, self.cfg.annualize_sharpe)

        diagnostics = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "sharpe": sharpe_val,
            "monthly_ic": monthly_ic,
            "monthly_ls": ls_returns,
            "months_used": len(monthly_ic)
        }

        # Objective routing
        if name == "spearman_ic_mean":
            if len(monthly_ic) < self.cfg.min_months_for_eval:
                return None, diagnostics
            return ic_mean, diagnostics

        if name == "sharpe_long_short":
            if sharpe_val is None or len(ls_returns) < self.cfg.min_months_for_eval:
                return None, diagnostics
            return sharpe_val, diagnostics

        if name == "hybrid_ic_sharpe":
            if len(monthly_ic) < self.cfg.min_months_for_eval or sharpe_val is None:
                return None, diagnostics
            hybrid = self.cfg.hybrid_weight_ic * ic_mean + (1 - self.cfg.hybrid_weight_ic) * sharpe_val
            diagnostics["hybrid"] = hybrid
            return hybrid, diagnostics

        raise ValueError(f"Unknown objective '{name}'")

    def _log_trial(self,
                   study_type: str,
                   window_id: int,
                   trial: int,
                   objective_value: float,
                   diagnostics: Dict[str, Any],
                   params: Dict[str, Any],
                   duration_sec: Optional[float] = None):
        if not self.cfg.trial_log_path:
            return

        safe_params = _json_sanitize(params)
        monthly_ic_safe = _json_sanitize(diagnostics.get("monthly_ic", []))
        monthly_ls_safe = _json_sanitize(diagnostics.get("monthly_ls", []))

        def _to_float(x):
            try:
                if x is None or (isinstance(x, float) and (x != x)):
                    return ""
                return f"{float(x):.8f}"
            except Exception:
                return ""

        row = {
            "study_type": study_type,
            "window_id": window_id,
            "trial": trial,
            "objective_value": _to_float(objective_value),
            "ic_mean": _to_float(diagnostics.get("ic_mean")),
            "ic_std": _to_float(diagnostics.get("ic_std")),
            "sharpe": _to_float(diagnostics.get("sharpe")),
            "months_used": diagnostics.get("months_used", 0),
            "duration_sec": _to_float(duration_sec if duration_sec is not None else diagnostics.get("duration_sec")),
            "params_json": json.dumps(safe_params),
            "monthly_ic_json": json.dumps(monthly_ic_safe),
            "monthly_ls_json": json.dumps(monthly_ls_safe)
        }

        with open(self.cfg.trial_log_path, "a", encoding="utf-8") as f:
            f.write(",".join(str(v) for v in row.values()) + "\n")

        # Console summary (brief)
        logger.info(f"[TrialLog] {study_type} win={window_id} trial={trial} "
                    f"obj={row['objective_value']} ic={row['ic_mean']} sharpe={row['sharpe']}")

    def _refit_full_model(self,
                          train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          features: List[str],
                          sectors: List[str],
                          target: str,
                          best_params_full: Dict[str, Any]) -> lgb.Booster:
        """
        Final refit on train+val combined using the same preprocessing logic.
        """
        combined = pd.concat([train_df, val_df], ignore_index=True)

        for s in sectors:
            if s in combined.columns:
                combined[s] = combined[s].astype("category")

        # Month-constant removal (recomputed on full train)
        final_features = features.copy()
        if self.cfg.remove_month_constants:
            final_features, removed = remove_month_constant(combined, final_features)
            logger.info(f"[Refit] Removed month-constant (refit): {removed}")

        categorical = [c for c in sectors if c in combined.columns]
        numeric = [f for f in final_features if f not in categorical and f in combined.columns]

        if self.cfg.monthly_zscore and numeric:
            combined = cross_sectional_z(combined, numeric)

        if self.cfg.target_clip and target in combined.columns:
            combined = clip_and_smooth_return(combined,
                                              target=target,
                                              z_clip=self.cfg.z_clip,
                                              smooth_scale=self.cfg.smooth_scale)

        feature_cols = numeric + categorical
        if not feature_cols:
            logger.warning("[Refit] No features left after preprocessing.")
            return None

        params = best_params_full.copy()
        num_boost_round = params.pop("n_estimators", self.cfg.final_n_estimators)

        # Use native train for consistency
        train_ds = lgb.Dataset(combined[feature_cols],
                               label=combined[target],
                               free_raw_data=False,
                               categorical_feature=categorical)
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[train_ds],
            callbacks=[log_evaluation(period=0)]
        )
        return model

    # -----------------------------
    # Reporting / Persistence
    # -----------------------------
    def best_history_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        rows = []
        for rec in self.history:
            flat = {k: v for k, v in rec.items() if k not in ("best_params",)}
            for p, val in rec["best_params"].items():
                flat[p] = val
            rows.append(flat)
        return pd.DataFrame(rows)

    def describe_last_best(self) -> Optional[Dict[str, Any]]:
        return self.previous_best_params

    def save_history_csv(self, path: str):
        df = self.best_history_dataframe()
        if not df.empty:
            df.to_csv(path, index=False)
            logger.info(f"[History] Saved history to {path}")

    def suggest_sampled_windows(self,
                                all_windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
                                k: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if k >= len(all_windows):
            return all_windows
        return random.sample(all_windows, k)


# ------------------------------------------------------------------
# Standalone Guard
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("optuna_tuner.py is intended to be imported into the backtest pipeline.")