import numpy as np
import pandas as pd
from typing import List, Tuple

def remove_month_constant(train_df: pd.DataFrame,
                          features: List[str],
                          date_col: str = "char_eom",
                          frac_threshold: float = 0.95) -> Tuple[List[str], List[str]]:
    """
    Remove features that have <=1 unique value in at least `frac_threshold`
    fraction of months (i.e. month-constant for most months).
    """
    bad = []
    g = train_df.groupby(date_col, observed=True)
    for f in features:
        if f not in train_df.columns:
            continue
        nunq = g[f].nunique()
        if len(nunq) == 0:
            continue
        if (nunq <= 1).mean() >= frac_threshold:
            bad.append(f)
    kept = [f for f in features if f not in bad]
    return kept, bad

def cross_sectional_z(df: pd.DataFrame,
                      features: List[str],
                      date_col: str = "char_eom") -> pd.DataFrame:
    """
    Cross-sectional z-score by month (mean 0, std 1 inside each month).
    Leaves NaNs if std==0; caller may handle/impute or leave (LightGBM can split on NaN).
    """
    df = df.copy()
    g = df.groupby(date_col, observed=True)
    for f in features:
        if f not in df.columns:
            continue
        mu = g[f].transform('mean')
        sigma = g[f].transform('std').replace(0, np.nan)
        df[f] = (df[f] - mu) / sigma
    return df

def clip_and_smooth_return(df: pd.DataFrame,
                           target: str = "stock_ret",
                           date_col: str = "char_eom",
                           z_clip: float = 5.0,
                           smooth_scale: float = 0.05) -> pd.DataFrame:
    """
    1. Cross-sectional z of target per month.
    2. Clip to [-z_clip, z_clip].
    3. Smooth with tanh scaling for regression stability.
    Stores original in {target}_orig.
    """
    df = df.copy()
    g = df.groupby(date_col, observed=True)
    mu = g[target].transform('mean')
    sigma = g[target].transform('std').replace(0, np.nan)
    z = (df[target] - mu) / sigma
    z = z.clip(-z_clip, z_clip)
    df[target + "_orig"] = df[target]
    df[target] = np.tanh(z * (1.0 / z_clip) / smooth_scale)
    return df