import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from joblib import Parallel, delayed
import gc
from scipy.stats import spearmanr
import logging

CURRENT_DIR = Path(__file__).resolve().parent
MODEL_DIR = CURRENT_DIR.parent
SRC_DIR = MODEL_DIR / "src"
DATA_DIR = MODEL_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Local module imports
from src.data_loader import load_data
from src.regime_features import load_and_prepare_macro, merge_macro_with_equity
from models import (
    ModelConfig, StockReturnPredictor, process_month,
    compute_RSI, downside_volatility
)
from src.optuna_tuner import TuningConfig, OptunaWindowTuner
# New preprocessing helpers (for compatibility with updated pipeline)
try:
    from feature_preprocess import cross_sectional_z, remove_month_constant, clip_and_smooth_return
except ImportError:
    cross_sectional_z = None  # Will guard later

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quant_backtest_fixed.log')
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing steps"""

    @staticmethod
    def clean_data(df: pd.DataFrame, stock_vars: list, ret_var: str) -> pd.DataFrame:
        """Clean and filter data (light filters to preserve cross-sectional breadth)"""
        # Remove penny stocks
        df = df[df['prc'] > 1.0].copy()

        # Remove extreme return outliers
        df = df[df[ret_var].between(-0.9, 3.0)].copy()

        # Ensure minimum history (≥60 obs)
        stock_counts = df.groupby('id').size()
        valid_stocks = stock_counts[stock_counts >= 30].index
        df = df[df['id'].isin(valid_stocks)].copy()

        return df


def compute_oos_r2_cross_sectional(df,
                                   ret_col='stock_ret',
                                   pred_col='lightgbm',
                                   date_col='char_eom',
                                   min_per_month=30):
    """
    Computes uncentered out-of-sample R^2 measures.
    Returns dict with overall & monthly stats plus a per-month DataFrame.
    """
    # Keep only rows with valid numbers
    sub = df[[date_col, ret_col, pred_col]].dropna()
    if sub.empty:
        return {}

    # Overall raw (no scaling)
    y = sub[ret_col].values
    s = sub[pred_col].values
    denom_all = np.sum(y**2)
    if denom_all <= 0:
        overall_raw_r2 = np.nan
        overall_scaled_r2 = np.nan
    else:
        overall_raw_r2 = 1.0 - np.sum((y - s)**2) / denom_all
        # Optimal scalar b
        ss = np.sum(s**2)
        if ss > 0:
            b_all = np.sum(y * s) / ss
            overall_scaled_r2 = 1.0 - np.sum((y - b_all * s)**2) / denom_all
        else:
            b_all = 0.0
            overall_scaled_r2 = np.nan

    monthly_rows = []
    for dt, g in sub.groupby(date_col):
        if len(g) < min_per_month:
            continue
        y_m = g[ret_col].values
        s_m = g[pred_col].values
        denom_m = np.sum(y_m**2)
        if denom_m <= 0:
            continue
        # Raw
        r2_raw_m = 1.0 - np.sum((y_m - s_m)**2) / denom_m
        # Scaled
        ss_m = np.sum(s_m**2)
        if ss_m > 0:
            b_m = np.sum(y_m * s_m) / ss_m
            r2_scaled_m = 1.0 - np.sum((y_m - b_m * s_m)**2) / denom_m
        else:
            b_m = 0.0
            r2_scaled_m = np.nan
        monthly_rows.append({
            date_col: dt,
            'n': len(g),
            'b_opt': b_m,
            'r2_raw': r2_raw_m,
            'r2_scaled': r2_scaled_m
        })

    monthly_df = pd.DataFrame(monthly_rows).sort_values(date_col)

    results = {
        'overall_raw_r2': overall_raw_r2,
        'overall_scaled_r2': overall_scaled_r2,
        'mean_monthly_raw_r2': monthly_df['r2_raw'].mean() if not monthly_df.empty else np.nan,
        'mean_monthly_scaled_r2': monthly_df['r2_scaled'].mean() if not monthly_df.empty else np.nan,
        'monthly_detail': monthly_df
    }
    return results

def compute_portfolio_metrics_detailed(pred_out: pd.DataFrame, ret_var: str = 'stock_ret'):
    """
    (Unchanged) Compute portfolio performance across several quantile constructions.
    Returns a dict of metrics keyed by portfolio type.
    """
    results = {}
    portfolio_types = {
        'quintile': 5,
        'decile': 10,
        'tercile': 3,
        'median': 2
    }

    for pf_type, n_bins in portfolio_types.items():
        monthly_returns = []
        for date, group in pred_out.groupby('char_eom'):
            if len(group) < n_bins * 10:
                continue
            group = group.copy()
            group['bin'] = pd.qcut(group['lightgbm'], q=n_bins, labels=False, duplicates='drop')
            bin_returns = group.groupby('bin')[ret_var].mean()
            if len(bin_returns) == n_bins:
                long_ret = bin_returns.iloc[-1]
                short_ret = bin_returns.iloc[0]
                ls_ret = long_ret - short_ret
                monthly_returns.append({
                    'date': date,
                    'long': long_ret,
                    'short': short_ret,
                    'long_short': ls_ret,
                    'spread': bin_returns.iloc[-1] - bin_returns.iloc[0]
                })
        if monthly_returns:
            returns_df = pd.DataFrame(monthly_returns)
            results[pf_type] = {
                'mean_return': returns_df['long_short'].mean(),
                'long_return': returns_df['long'].mean() * 12,
                'short_return': returns_df['short'].mean() * 12,
                'annualized_return': returns_df['long_short'].mean() * 12,
                'volatility': returns_df['long_short'].std(),
                'annualized_vol': returns_df['long_short'].std() * np.sqrt(12),
                'sharpe': (returns_df['long_short'].mean() /
                           (returns_df['long_short'].std() + 1e-10)) * np.sqrt(12),
                'win_rate': (returns_df['long_short'] > 0).mean(),
                'avg_spread': returns_df['spread'].mean(),
                'information_ratio': (returns_df['long_short'].mean() /
                                      (returns_df['long_short'].std() + 1e-10))
            }
            cumulative = (1 + returns_df['long_short']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            results[pf_type]['max_drawdown'] = drawdown.min()
            results[pf_type]['calmar'] = (results[pf_type]['annualized_return'] /
                                          abs(results[pf_type]['max_drawdown'] + 1e-10))
    return results


def main():
    print(f"\n{'=' * 60}")
    print(f"FIXED RANKER BACKTEST - {datetime.datetime.now()}")
    print(f"{'=' * 60}\n")

    pd.set_option("mode.chained_assignment", None)

    # Load factor metadata
    stock_vars = list(load_data(filename="factors.csv")["variable"].values)
    ret_var = "stock_ret"

    needed_cols = stock_vars + ["date", "ret_eom", "year", "month", "id",
                                "gvkey", "char_eom", "excntry", ret_var, "prc", "sector"]

    dtype_dict = {col: np.float32 for col in stock_vars + [ret_var, "prc"]}
    dtype_dict.update({"year": np.int32, "month": np.int32, "id": "category",
                       "gvkey": "str", "sector": "category"})

    print("Loading CSV dataset...")
    raw = load_data(
        filename="data_w_sectors_w_ticker.csv",
        usecols=needed_cols,
        dtype=dtype_dict,
        parse_dates=["ret_eom", "char_eom", "date"],
        low_memory=False
    )

    # Initial cleaning
    preprocessor = DataPreprocessor()
    new_set = raw[raw[ret_var].notna()].copy()
    new_set = preprocessor.clean_data(new_set, stock_vars, ret_var)

    print(f"Data shape after cleaning: {new_set.shape}")
    print(f"Date range: {new_set['date'].min()} to {new_set['date'].max()}")
    print(f"Unique stocks: {new_set['id'].nunique()}")

    # ========== Monthly Processing (existing process_month still used) ==========
    # Supplemental Data Cleaning
    def precompute_monthly_features(df: pd.DataFrame, stock_vars: list):
        monthly = df.groupby("char_eom", observed=True)
        tasks = [(char_eom, monthly_raw, stock_vars, df) for char_eom, monthly_raw in monthly]
        results = Parallel(n_jobs=16)(
            delayed(process_month)(task) for task in tasks
        )
        data_list = [result[0] for result in results]
        data_local = pd.concat(data_list, ignore_index=True)
        return data_local

    data = precompute_monthly_features(new_set, stock_vars)

    # ========== Stock-level Engineering ==========
    # Features are engineered on price, at t=0
    def compute_engineered_features(
            df: pd.DataFrame,
            sector_col: str = "sector",
            n_jobs: int = 8,
            add_leave_one_out: bool = False
    ) -> pd.DataFrame:

        df = df.copy()
        if 'prc' not in df.columns:
            return df

        df = df.sort_values(['id', 'char_eom']).reset_index(drop=True)

        def process_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            group['ret_monthly'] = group['prc'].pct_change()

            for window in [3, 6, 12]:
                group[f'MA_{window}m'] = group['prc'].rolling(window, min_periods=1).mean()
                group[f'price_to_MA_{window}m'] = group['prc'] / (group[f'MA_{window}m'] + 1e-12) - 1

            for months in [1, 3, 6, 12]:
                group[f'ret_{months}m'] = group['prc'].pct_change(periods=months)

            for window in [6, 12]:
                group[f'vol_{window}m'] = group['ret_monthly'].rolling(
                    window, min_periods=max(2, window // 3)
                ).std()
                group[f'vol_{window}m'] = -group[f'vol_{window}m']

            group['momentum_12m'] = group['prc'].shift(0) / group['prc'].shift(12) - 1
            group['momentum_acceleration'] = group['momentum_12m'].diff()

            group['RSI_12m'] = compute_RSI(group['prc'], 12)
            group['RSI_12m'] = (group['RSI_12m'] - 50) / 50
            group['RSI_6m'] = compute_RSI(group['prc'], 6)
            group['RSI_6m'] = (group['RSI_6m'] - 50) / 50

            roll_mean = group["prc"].rolling(20, min_periods=5).mean()
            roll_std = group["prc"].rolling(20, min_periods=5).std()
            group["bollinger_dist"] = (group["prc"] - roll_mean) / (2 * roll_std.replace(0, np.nan))

            group['downside_vol_12m'] = (
                group['ret_monthly']
                .rolling(12, min_periods=4)
                .apply(lambda x: downside_volatility(x) if len(x) > 0 else np.nan, raw=False)
            )
            group['downside_vol_12m'] = -group['downside_vol_12m']
            return group

        groups = [g for _, g in df.groupby('id', observed=True, sort=False)]
        processed = Parallel(n_jobs=n_jobs)(
            delayed(process_group)(g) for g in groups
        )
        df = pd.concat(processed, ignore_index=True)

        if sector_col in df.columns:
            if not pd.api.types.is_categorical_dtype(df[sector_col]):
                df[sector_col] = df[sector_col].astype('category')

            sector_base_feats = [
                'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
                'momentum_12m', 'momentum_acceleration',
                'vol_6m', 'vol_12m',
                'downside_vol_12m',
                'RSI_6m', 'RSI_12m',
                'bollinger_dist'
            ]
            sector_base_feats = [c for c in sector_base_feats if c in df.columns]

            df['sector_size'] = df.groupby(['char_eom', sector_col])['id'].transform('count')

            for feat in sector_base_feats:
                df[f'sector_mean_{feat}'] = df.groupby(['char_eom', sector_col])[feat].transform('mean')
                df[f'sector_std_{feat}'] = df.groupby(['char_eom', sector_col])[feat].transform('std')
                df[f'rel_{feat}'] = df[feat] - df[f'sector_mean_{feat}']
                df[f'z_sector_{feat}'] = (df[feat] - df[f'sector_mean_{feat}']) / (
                    df[f'sector_std_{feat}'].replace(0, np.nan))
                df[f'prank_sector_{feat}'] = (
                    df.groupby(['date', sector_col])[feat]
                    .transform(lambda x: x.rank(pct=True, method='average'))
                )

            if add_leave_one_out:
                eps = 1e-12
                for feat in sector_base_feats:
                    sum_name = f'_sector_sum_{feat}'
                    df[sum_name] = df.groupby(['char_eom', sector_col])[feat].transform('sum')
                    df[f'sector_mean_excl_{feat}'] = np.where(
                        df['sector_size'] > 1,
                        (df[sum_name] - df[feat]) / (df['sector_size'] - 1),
                        np.nan
                    )
                    df.drop(columns=[sum_name], inplace=True)

            rolling_specs = [
                ('sector_mean_ret_1m', 3),
                ('sector_mean_ret_1m', 9),
                ('sector_mean_momentum_12m', 3),
            ]
            needed_cols = ['char_eom', sector_col] + list({col for col, _ in rolling_specs})
            sector_panel = df[needed_cols].drop_duplicates().sort_values(['char_eom', sector_col])
            for col, window in rolling_specs:
                if col in sector_panel.columns:
                    sector_panel[f'{col}_rollmean_{window}'] = (
                        sector_panel.groupby(sector_col)[col]
                        .transform(lambda x: x.rolling(window, min_periods=max(2, window // 2)).mean())
                    )
            df = df.merge(
                sector_panel[[sector_col, 'char_eom'] + [c for c in sector_panel.columns if 'rollmean' in c]],
                on=[sector_col, 'char_eom'],
                how='left'
            )

            df['sector_herfindahl_eqw'] = 1.0 / df['sector_size'].replace(0, np.nan)
            #df['sector_is_small'] = (df['sector_size'] < 5).astype(int)

        return df

    orig_cols = set(data.columns)
    data = compute_engineered_features(data)
    new_cols = [c for c in data.columns if c not in orig_cols]
    print(f"Engineered features added: {len(new_cols)} columns")

    # Date alignment to ensure one snapshot per (id, ret_eom)
    '''
    data["ret_eom_ym"] = data["ret_eom"].dt.to_period("M")
    data["date_ym"] = data["date"].dt.to_period("M")
    data["date_diff"] = (data["ret_eom"] - data["date"]).dt.days
    data = data[data["date_diff"] >= 0]
    data = data.loc[data.groupby(["id", "ret_eom_ym"], observed=True)["date_diff"].idxmin()]
    data = data.drop(columns=["ret_eom_ym", "date_ym", "date_diff"])
    '''
    # Macro / regimes
    macro = load_and_prepare_macro(
        path=str(DATA_DIR / "macro_features.csv"),
        date_col="char_eom",
        lag_months=0,
        min_periods_z=24,
        coverage_min_frac=0.5
    )
    print("Macro rows:", len(macro))
    data = merge_macro_with_equity(data, macro)

    if "regime_label" in data.columns:
        data["regime_label"] = data["regime_label"].astype("category")

    print(f"Shape after processing: {data.shape}")
    data = data.sort_values(["id", "char_eom"])

    # ===================== Model & Tuner Config =====================
    config = ModelConfig()  # Uses native LightGBM w/ monthly Spearman feval by default
    predictor = StockReturnPredictor(config)

    tuning_enabled = True
    tuning_cfg = TuningConfig(
        objective_name="sharpe_long_short",
        n_trials_per_window=10,
        multi_fidelity_n_estimators_trial=800,
        final_n_estimators=2000,
        use_monthly_spearman_feval=True,
        remove_month_constants=True,
        monthly_zscore=True,
        target_clip=True,
        use_gpu=False,
    )

    tuner = OptunaWindowTuner(tuning_cfg, base_params=config.lgbm_params)

    sampled_windows = []
    sampled_windows_cap = 5

    # ===================== Window Schedule =====================
    starting = pd.to_datetime("20050101", format="%Y%m%d")
    max_counter = 0
    while (starting + pd.DateOffset(years=13 + max_counter)) <= pd.to_datetime("20260101", format="%Y%m%d"):
        max_counter += 1

    print(f"\nRunning {max_counter} backtesting windows...")
    categorical_features = ['regime_label']
    #categorical_features = None
    #if 'regime_label' in data.columns:
        #categorical_features.append('regime_label')

    # ========= Helper: safe final training (retain backward compatibility) =========
    def _train_final_model(train_df, val_df, features, target):
        """
        Backward-compatible wrapper.
        Older code expects predictor.train_model(...)
        New code uses predictor.train(...)
        """
        if hasattr(predictor, 'train_model'):
            return predictor.train_model(
                train_df, val_df,
                features=features,
                sectors=categorical_features,
                target=target
            )
        else:
            return predictor.train(
                train=train_df,
                validate=val_df,
                features=features,
                categorical=categorical_features,
                target=target
            )

    # ========= Window Processor =========
    def process_window(counter: int):
        try:
            cutoff = [
                starting,
                starting + pd.DateOffset(years=10 + counter),
                starting + pd.DateOffset(years=12 + counter),
                starting + pd.DateOffset(years=13 + counter)
            ]
            train_start, train_end = cutoff[0], cutoff[1]
            val_end = cutoff[2]
            test_end = cutoff[3]

            print(f"\n{'=' * 60}")
            print(f"Window {counter + 1}/{max_counter}")
            print(f"Train: {train_start.date()} to {train_end.date()}")
            print(f"Val:   {train_end.date()} to {val_end.date()}")
            print(f"Test:  {val_end.date()} to {test_end.date()}")
            print(f"{'=' * 60}")

            train, validate, test = predictor.prepare_data_window(
                data, train_start, train_end, val_end, test_end
            )

            if len(train) < 1000 or len(validate) < 500 or len(test) < 100:
                print(f"Warning: Insufficient samples in window {counter + 1}")
                return pd.DataFrame(), pd.DataFrame()

            base_exclude = set(['id', 'ret_eom', "char_eom", "excntry", 'date', 'year', 'month',
                                'gvkey', ret_var, 'sector', 'ggroup'])
            if 'regime_label' in train.columns:
                base_exclude.add('regime_label')  # passed separately as categorical
            all_features = [c for c in train.columns if c not in base_exclude]

            print(f"Training with {len(all_features)} total features")

            # Optionally store sampled windows for later global refinement
            if tuning_enabled and len(sampled_windows) < sampled_windows_cap:
                sampled_windows.append({
                    "train": train.copy(),
                    "val": validate.copy(),
                    "features": all_features.copy()
                })

            # ===================== Tuning =====================
            if tuning_enabled and tuner.should_tune_window():
                best_params = tuner.tune_window(
                    window_id=counter,
                    train_df=train.copy(),
                    val_df=validate.copy(),
                    features=all_features,
                    sectors=categorical_features,
                    target=ret_var,
                    refit_full=False
                )
                config.lgbm_params.update(best_params)
            elif tuning_enabled and tuner.previous_best_params:
                # Reuse previously tuned params
                config.lgbm_params.update(tuner.previous_best_params)

            # ===================== Final Training =====================
            model = _train_final_model(train, validate, all_features, ret_var)
            if model is None:
                return pd.DataFrame(), pd.DataFrame()

            # ===================== Prediction =====================
            # Ensure test monthly z-score matches training regime if needed
            raw_market_equity = test["market_equity"].values
            raw_beta_60m = test["beta_60m"].values


            raw_preds = predictor.predict(test)
            test["market_equity"] = raw_market_equity
            test["beta_60m"] = raw_beta_60m
            # New predictor.predict returns Series; old path returned full DataFrame — unify:
            if isinstance(raw_preds, pd.Series):
                test_preds = raw_preds
            else:  # Backward safety
                test_preds = raw_preds['lightgbm'] if 'lightgbm' in raw_preds.columns else pd.Series(0.0, index=test.index)
            test['gvkey'] = test['gvkey'].astype(str)   #force gvkey string
            reg_pred = test[["date", "year", "month", "ret_eom", "excntry", "char_eom", "id", "gvkey", "sector", "market_equity", "beta_60m", "regime_label", "prc", ret_var]].copy()

            reg_pred["lightgbm"] = test_preds.values

            if len(reg_pred) > 0:
                ic, _ = spearmanr(reg_pred['lightgbm'], reg_pred[ret_var])
                print(f"Window {counter + 1} Spearman IC: {ic:.4f}")

            importance = predictor.feature_importance.copy() if getattr(predictor, 'feature_importance', None) is not None else pd.DataFrame()

            del train, validate, test
            gc.collect()
            return reg_pred, importance

        except Exception as e:
            logger.error(f"Error in window {counter + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), pd.DataFrame()

    # ===================== Execute Windows (sequential with tuning) =====================
    if tuning_enabled:
        results = []
        for counter in range(max_counter):
            results.append(process_window(counter))
    else:
        # Parallel path (disabled if tuning because tuner not thread-safe)
        results = Parallel(n_jobs=8)(
            delayed(process_window)(counter)
            for counter in range(max_counter)
        )

    pred_out_list = [r[0] for r in results if not r[0].empty]
    importance_list = [r[1] for r in results if not r[1].empty]

    if not pred_out_list:
        print("ERROR: No predictions generated!")
        return


    pred_out = pd.concat(pred_out_list, ignore_index=True)
    '''
    # ===================== Optional Global Refinement =====================
    if tuning_enabled and sampled_windows:
        sampled_feature_sets = [set(w["features"]) for w in sampled_windows]
        common_features = sorted(set.intersection(*sampled_feature_sets)) if sampled_feature_sets else []
        if common_features:
            global_tune_windows = [
                (w["train"][common_features + ['char_eom', 'sector', ret_var] +
                            (['regime_label'] if 'regime_label' in w["train"].columns else [])],
                 w["val"][common_features + ['char_eom', 'sector', ret_var] +
                          (['regime_label'] if 'regime_label' in w["val"].columns else [])])
                for w in sampled_windows
            ]
            print("\nStarting global refinement tuning on sampled windows...")
            global_best = tuner.tune_global(
                sampled_windows=global_tune_windows,
                features=common_features,
                sectors=categorical_features,
                target=ret_var,
                n_trials=1  # light refinement; increase for deeper search
            )
            print("Global tuning complete. Updating final params.")
            config.lgbm_params.update(global_best)
        tuner.save_history_csv("optuna_tuning_history.csv")
    elif tuning_enabled:
        tuner.save_history_csv("optuna_tuning_history.csv")
    '''

    # Save predictions
    out_path = DATA_DIR / "results.csv"
    print(f"\nSaving predictions to: {out_path}")
    pred_out.to_csv(out_path, index=False)

    # Save feature importance
    if importance_list:
        importance_df = pd.concat(importance_list, ignore_index=True)
        importance_agg = importance_df.groupby("feature")["importance"].agg(['mean', 'std']).reset_index()
        importance_agg = importance_agg.sort_values("mean", ascending=False)

        importance_path = DATA_DIR / "feature_importance.csv"
        importance_agg.to_csv(importance_path, index=False)

        print("\nTop 20 features:")
        print(importance_agg.head(20))

    # Comprehensive performance evaluation
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL PERFORMANCE EVALUATION")
    print("=" * 60)

    yreal = pred_out[ret_var].values
    ypred = pred_out["lightgbm"].values

    # 1. RANKING METRICS
    print("\n1. RANKING PERFORMANCE METRICS:")

    # Overall Spearman
    spearman_overall, p_value = spearmanr(ypred, yreal)
    print(f"   Overall Spearman ρ: {spearman_overall:.4f} (p-value: {p_value:.4e})")

    # Monthly Information Coefficient (IC)
    monthly_ic = pred_out.groupby('char_eom').apply(
        lambda g: spearmanr(g['lightgbm'], g[ret_var])[0] if len(g) >= 30 else np.nan
    ).dropna()

    print(f"\n   Information Coefficient (Monthly IC):")
    print(f"     Mean IC:        {monthly_ic.mean():.4f}")
    print(f"     Median IC:      {monthly_ic.median():.4f}")
    print(f"     Std IC:         {monthly_ic.std():.4f}")
    print(f"     IC Ratio (IR):  {monthly_ic.mean() / (monthly_ic.std() + 1e-10):.4f}")
    print(f"     % Positive IC:  {(monthly_ic > 0).mean() * 100:.1f}%")
    print(f"     % IC > 0.03:    {(monthly_ic > 0.03).mean() * 100:.1f}%")

    # IC by year
    pred_out['year'] = pd.to_datetime(pred_out['char_eom']).dt.year
    yearly_ic = pred_out.groupby('year').apply(
        lambda g: spearmanr(g['lightgbm'], g[ret_var])[0] if len(g) >= 100 else np.nan
    ).dropna()

    if len(yearly_ic) > 0:
        print(f"\n   Yearly IC Performance:")
        for year, ic in yearly_ic.items():
            print(f"     {year}: {ic:.4f}")

    # 2. PORTFOLIO PERFORMANCE
    print("\n2. PORTFOLIO PERFORMANCE ANALYSIS:")

    portfolio_results = compute_portfolio_metrics_detailed(pred_out, ret_var)

    for pf_type, metrics in portfolio_results.items():
        print(f"\n   {pf_type.upper()} Portfolio:")
        print(f"     Annualized Return:    {metrics['annualized_return'] * 100:.2f}%")
        print(f"     Annualized Vol:       {metrics['annualized_vol'] * 100:.2f}%")
        print(f"     Annualized Long:       {metrics['long_return'] * 100:.2f}%")
        print(f"     Annualized Short:       {metrics['short_return'] * 100:.2f}%")
        print(f"     Sharpe Ratio:         {metrics['sharpe']:.3f}")
        print(f"     Information Ratio:    {metrics['information_ratio']:.3f}")
        print(f"     Win Rate:             {metrics['win_rate'] * 100:.1f}%")
        print(f"     Max Drawdown:         {metrics['max_drawdown'] * 100:.2f}%")
        print(f"     Calmar Ratio:         {metrics['calmar']:.3f}")

    # 3. NDCG and Ranking Quality
    print("\n3. RANKING QUALITY METRICS (NDCG):")

    from sklearn.metrics import ndcg_score

    def compute_ndcg_at_k(group, k_list=[10, 20, 50]):
        """Compute NDCG at different K values"""
        if len(group) < max(k_list):
            return {f'ndcg@{k}': np.nan for k in k_list}

        # Create relevance scores based on actual returns
        relevance = pd.qcut(group[ret_var], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
        if relevance.isna().all():
            return {f'ndcg@{k}': np.nan for k in k_list}

        y_true = relevance.fillna(0).values.reshape(1, -1)
        y_score = group['lightgbm'].values.reshape(1, -1)

        results = {}
        for k in k_list:
            try:
                results[f'ndcg@{k}'] = ndcg_score(y_true, y_score, k=k)
            except:
                results[f'ndcg@{k}'] = np.nan

        return results

    ndcg_results = pred_out.groupby('char_eom').apply(compute_ndcg_at_k)

    for k in [10, 20, 50]:
        ndcg_values = [r[f'ndcg@{k}'] for r in ndcg_results if not np.isnan(r[f'ndcg@{k}'])]
        if ndcg_values:
            print(f"   NDCG@{k}: Mean={np.mean(ndcg_values):.4f}, "
                  f"Median={np.median(ndcg_values):.4f}, "
                  f"Std={np.std(ndcg_values):.4f}")


    # 5. Stability Analysis
    print("\n4. MODEL STABILITY ANALYSIS:")

    # Rolling IC stability
    pred_out_sorted = pred_out.sort_values('char_eom')
    dates = pred_out_sorted['char_eom'].unique()

    if len(dates) >= 12:
        rolling_ic = []
        for i in range(len(dates) - 11):
            window_data = pred_out_sorted[pred_out_sorted['char_eom'].isin(dates[i:i + 12])]
            if len(window_data) >= 100:
                ic, _ = spearmanr(window_data['lightgbm'], window_data[ret_var])
                rolling_ic.append(ic)

        if rolling_ic:
            print(f"   12-Month Rolling IC:")
            print(f"     Mean:     {np.mean(rolling_ic):.4f}")
            print(f"     Std:      {np.std(rolling_ic):.4f}")
            print(f"     Min:      {np.min(rolling_ic):.4f}")
            print(f"     Max:      {np.max(rolling_ic):.4f}")

    oos_r2_stats = compute_oos_r2_cross_sectional(pred_out,
                                                  ret_col='stock_ret',
                                                  pred_col='lightgbm',
                                                  date_col='ret_eom')

    print("\n6. OUT-OF-SAMPLE R^2 (Uncentered, Through-Origin):")
    if oos_r2_stats:
        print(f"   Overall Raw R^2:       {oos_r2_stats['overall_raw_r2']:.4f}")
        print(f"   Overall Scaled R^2:    {oos_r2_stats['overall_scaled_r2']:.4f}")
        print(f"   Mean Monthly Raw R^2:  {oos_r2_stats['mean_monthly_raw_r2']:.4f}")
        print(f"   Mean Monthly Scaled R^2:{oos_r2_stats['mean_monthly_scaled_r2']:.4f}")
        # Optionally save monthly detail
        oos_r2_stats['monthly_detail'].to_csv("oos_r2_monthly.csv", index=False)
    else:
        print("   (No data)")

    print("\n" + "=" * 60)
    print(f"BACKTEST COMPLETED AT {datetime.datetime.now()}")
    print("=" * 60)



if __name__ == "__main__":
    main()