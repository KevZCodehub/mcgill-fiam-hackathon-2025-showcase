import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import spearmanr
import warnings
import lightgbm as lgb
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def _cross_sectional_z_inplace(df, features, date_col="char_eom"):
    g = df.groupby(date_col, observed=True)
    for f in features:
        mu = g[f].transform('mean')
        sigma = g[f].transform('std').replace(0, np.nan)
        df[f] = (df[f] - mu) / sigma
    return df

def compute_RSI(series, period=14):
    """Compute RSI indicator"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def downside_volatility(returns, threshold=0):
    """Calculate downside volatility"""
    neg_returns = returns[returns < threshold]
    if len(neg_returns) < 2:
        return np.nan
    return np.sqrt((neg_returns ** 2).mean())

def process_month(args):
    """Improved month processing with better normalization"""
    char_eom, monthly_raw, stock_vars, new_set = args
    group = monthly_raw.copy()
    warning_log = []

    print(f"Processing month-end: {char_eom}")

    for var in stock_vars:
        unique_values = group[var].dropna().nunique()
        if unique_values <= 1:
            warning_log.append((char_eom, var, "identical or all missing pre-imputation"))

        # Use more robust imputation
        var_median = group[var].median(skipna=True)
        if pd.isna(var_median):
            var_median = new_set[var].median()  # Use median instead of mean
            if pd.isna(var_median):
                var_median = 0
            warning_log.append((char_eom, var, "used global median imputation"))

        group[var] = group[var].fillna(var_median)

    return group, warning_log




@dataclass
class ModelConfig:
    lgbm_params: Dict = None
    use_native_lgb: bool = True  # use lgb.train with custom feval
    monthly_zscore: bool = True
    remove_month_constants: bool = True
    target_clip: bool = True
    z_clip: float = 5.0
    smooth_scale: float = 0.05
    min_months_for_ic: int = 4
    log_top_k_features: int = 10  # number of top features to log
    store_importance_types: List[str] = None

    def __post_init__(self):
        if self.lgbm_params is None:
            self.lgbm_params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.01,
                "num_leaves": 192,
                "min_data_in_leaf": 40,
                "feature_fraction": 0.75,
                "bagging_fraction": 0.8,
                "bagging_freq": 3,
                "lambda_l2": 1.0,
                "lambda_l1": 0.0,
                "min_split_gain": 5e-4,
                "n_estimators": 2000,  # cap for native interface
                "verbosity": -1,
                "boosting_type": "gbdt",
                "random_state": 42,
                "n_jobs": 14
            }
        if self.store_importance_types is None:
            self.store_importance_types = ["gain", "split"]


class StockReturnPredictor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_cols = None
        self.categorical_features: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    @staticmethod
    def _make_monthly_spearman_feval(month_index_array: np.ndarray, min_obs: int):
        month_index_array = np.asarray(month_index_array)

        def feval(preds, dataset):
            y = dataset.get_label()
            ic_list = []
            months = month_index_array
            for m in np.unique(months):
                mask = months == m
                if mask.sum() < 30:  # ensure cross-section size
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

    def prepare_data_window(self,
                            data: pd.DataFrame,
                            train_start: pd.Timestamp,
                            train_end: pd.Timestamp,
                            val_end: pd.Timestamp,
                            test_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = data[(data['date'] >= train_start) & (data['date'] <= train_end)].copy()
        validate = data[(data['date'] > train_end) & (data['date'] <= val_end)].copy()
        test = data[(data['date'] > val_end) & (data['date'] <= test_end)].copy()
        for df in (train, validate, test):
            if 'sector' in df.columns:
                df['sector'] = df['sector'].astype('category')
            if 'regime_label' in df.columns:
                df['regime_label'] = df['regime_label'].astype('category')
            if 'gvkey' in df.columns:
                df['gvkey'] = df['gvkey'].astype('str')
        logger.info(f"Window split - Train: {len(train)}, Val: {len(validate)}, Test: {len(test)}")
        return train, validate, test

    def train(self,
              train: pd.DataFrame,
              validate: pd.DataFrame,
              features: List[str],
              categorical: List[str],
              target: str = "stock_ret"):

        from src.feature_preprocess import remove_month_constant, cross_sectional_z, clip_and_smooth_return

        if self.config.remove_month_constants:
            features, removed = remove_month_constant(train, features)
            self.removed_constants = removed
            if removed:
                logger.info(f"Removed month-constant: {removed}")
        else:
            self.removed_constants = []

        categorical = [c for c in categorical if c in train.columns]
        for c in categorical:
            train[c] = train[c].astype('category')
            validate[c] = validate[c].astype('category')

        numeric = [f for f in features if f not in categorical and f in train.columns]

        if self.config.monthly_zscore and numeric:
            train = _cross_sectional_z_inplace(train, numeric)
            validate = _cross_sectional_z_inplace(validate, numeric)

        if self.config.target_clip and target in train.columns:
            train = clip_and_smooth_return(train,
                                           target=target,
                                           z_clip=self.config.z_clip,
                                           smooth_scale=self.config.smooth_scale)

        self.feature_cols = numeric + categorical
        self.categorical_features = categorical

        if not self.feature_cols:
            logger.warning("No usable features after preprocessing.")
            self.model = None
            return None

        if self.config.use_native_lgb:
            train_month_index = train['char_eom'].factorize()[0]
            val_month_index = validate['char_eom'].factorize()[0]
            self.month_index_val = val_month_index

            train_ds = lgb.Dataset(train[self.feature_cols],
                                   label=train[target],
                                   free_raw_data=False,
                                   categorical_feature=categorical)
            val_ds = lgb.Dataset(validate[self.feature_cols],
                                 label=validate[target],
                                 free_raw_data=False,
                                 categorical_feature=categorical)

            feval = self._make_monthly_spearman_feval(val_month_index,
                                                      min_obs=self.config.min_months_for_ic)

            params = self.config.lgbm_params.copy()
            num_boost_round = params.pop("n_estimators", 2000)

            self.model = lgb.train(
                params,
                train_ds,
                num_boost_round=num_boost_round,
                valid_sets=[val_ds],
                feval=feval,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200),
                    lgb.log_evaluation(period=100)
                ]
            )
        else:
            model = lgb.LGBMRegressor(**self.config.lgbm_params)
            model.fit(
                train[self.feature_cols], train[target],
                eval_set=[(validate[self.feature_cols], validate[target])],
                categorical_feature=categorical,
                eval_metric="l2",
            )
            self.model = model

        # Capture feature importances
        self._compute_and_store_feature_importance()
        self._log_top_features()
        return self.model


    def predict(self, test: pd.DataFrame) :
        if self.model is None:
            return pd.Series(0.0, index=test.index)
        test = test.copy()
        for c in self.categorical_features:
            if c in test.columns:
                test[c] = test[c].astype('category')

        if self.config.monthly_zscore:
            numeric = [f for f in self.feature_cols if f not in self.categorical_features]
            test = _cross_sectional_z_inplace(test, numeric)

        preds = self.model.predict(
            test[self.feature_cols],
            num_iteration=getattr(self.model, 'best_iteration', None)
        )
        test['lightgbm'] = preds
        return test
    # ---------- Importance Utilities ----------



    def _compute_and_store_feature_importance(self):
        """
        Populates self.feature_importance with columns:
           feature, importance_gain, importance_split
        (depending on configured importance types)
        """
        if self.model is None or not self.feature_cols:
            self.feature_importance = None
            return

        rows = {"feature": self.feature_cols}
        if isinstance(self.model, lgb.Booster):
            # Native booster
            for imp_type in self.config.store_importance_types:
                key = f"importance_{imp_type}"
                rows[key] = self.model.feature_importance(importance_type=imp_type)
        else:
            # Sklearn API
            # Only split importance available directly (feature_importances_)
            if "split" in self.config.store_importance_types:
                rows["importance_split"] = self.model.feature_importances_
            if "gain" in self.config.store_importance_types:
                try:
                    booster = self.model.booster_
                    rows["importance_gain"] = booster.feature_importance(importance_type="gain")
                except Exception:
                    pass

        df = pd.DataFrame(rows)
        # Provide a unified 'importance' column (prefer gain if exists)
        if "importance_gain" in df.columns:
            df["importance"] = df["importance_gain"]
        elif "importance_split" in df.columns:
            df["importance"] = df["importance_split"]
        else:
            df["importance"] = 0.0
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        self.feature_importance = df

    def _log_top_features(self):
        if self.feature_importance is None:
            logger.info("No feature importance available.")
            return
        k = self.config.log_top_k_features
        logger.info("\nTop %d features:", k)
        for _, row in self.feature_importance.head(k).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.1f}")

    def get_feature_importance(self,
                               top_k: Optional[int] = None,
                               importance_type: str = "gain") -> pd.DataFrame:
        """
        Returns a copy of feature importance table.
        importance_type: "gain", "split", or "importance" (auto).
        """
        if self.feature_importance is None:
            return pd.DataFrame()
        df = self.feature_importance.copy()
        if importance_type in ("gain", "split"):
            col = f"importance_{importance_type}"
            if col in df.columns:
                order_col = col
            else:
                order_col = "importance"
        else:
            order_col = "importance"
        df = df.sort_values(order_col, ascending=False)
        if top_k:
            df = df.head(top_k)
        return df.reset_index(drop=True)
