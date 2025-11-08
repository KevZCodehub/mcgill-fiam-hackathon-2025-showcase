import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# ------------------------------------------------------------
# Expanding z-score that:
# - Uses only past & current non-null values
# - Requires a minimum number (min_periods_non_null) of non-null observations
# - Does NOT "borrow" future distribution info
# ------------------------------------------------------------
def expanding_zscore(series: pd.Series, min_periods_non_null: int = 24) -> pd.Series:
    """
    Compute expanding z-score for a series using only past data.
    Z_t is defined only if the series has at least `min_periods_non_null`
    non-null observations up to and including time t.

    Parameters
    ----------
    series : pd.Series
        Time-ordered series.
    min_periods_non_null : int
        Minimum number of non-null observations needed before producing a z-score.

    Returns
    -------
    pd.Series
        Expanding z-scores with NaN where insufficient history or original value is NaN.
    """
    s = series.astype(float)
    notnull_mask = s.notna()

    # Cumulative counts of non-null values
    cum_count = notnull_mask.cumsum()

    # Replace NaNs with 0 for cumulative sums; they will be filtered using mask later
    filled = s.fillna(0.0)
    cum_sum = filled.cumsum()
    cum_sum_sq = (filled ** 2).cumsum()

    # Where we have at least min_periods_non_null non-null obs
    enough = cum_count >= min_periods_non_null

    # Initialize result with NaNs
    z = pd.Series(np.nan, index=s.index, dtype=float)

    # Indices where we can compute stats
    valid_idx = enough & notnull_mask
    if valid_idx.any():
        count = cum_count[valid_idx].astype(float)
        sum_ = cum_sum[valid_idx]
        sum_sq = cum_sum_sq[valid_idx]

        # Sample variance
        var = (sum_sq - (sum_ ** 2) / count) / (count - 1)
        # Numerical safety
        var = var.where(var > 0.0, np.nan)
        std = np.sqrt(var)

        # Current values
        current_vals = s[valid_idx]
        mean = sum_ / count
        z.loc[valid_idx] = (current_vals - mean) / std.replace(0, np.nan)

    return z


def _first_valid_index_mask(series: pd.Series) -> pd.Series:
    """
    Returns a 0/1 mask (Series) where 1 indicates the series is at or after
    its first non-null observation (structural availability).
    """
    s = series
    first_valid = s.first_valid_index()
    if first_valid is None:
        return pd.Series(0, index=s.index, dtype=int)
    mask = (s.index >= first_valid).astype(int)
    return pd.Series(mask, index=s.index, dtype=int)


def _select_available_name(macro: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    """
    Given a list of candidate column names, return the first one that exists in macro.
    """
    for name in preferred:
        if name in macro.columns:
            return name
    return None


def load_and_prepare_macro(path: str,
                           date_col: str = "char_eom",
                           lag_months: int = 1,
                           min_periods_z: int = 24,
                           coverage_min_frac: float = 0.5,
                           risk_off_hi: float = 0.5,
                           risk_on_lo: float = -0.3) -> pd.DataFrame:
    """
    Load monthly macro data and build regime features.

    Key Features:
    -------------
    - Expanding z-scores per feature (no future leakage; requires min_periods_z non-null points).
    - Composite scores:
        growth_score        = mean of available growth components' z-scores
        risk_aversion_score = mean of available risk components' z-scores
      Only defined if valid fraction >= coverage_min_frac.
    - Regime classification:
        risk_off: risk_aversion_score > risk_off_hi AND growth_score < 0
        risk_on : risk_aversion_score < risk_on_lo AND growth_score > 0
        else    : neutral
      If either composite is NaN => 'unknown'.
    - Availability / missingness flags:
        <COL>_struct_available: 1 if at or after first real observation
        <COL>_orig_missing    : 1 if original raw value is NaN that month
    - Lagging macro features by lag_months (simulate release delay).

    Parameters
    ----------
    path : str
        CSV path of macro data.
    date_col : str
        Column containing month-end date aligning with equity char_eom.
    lag_months : int
        Shift forward (positive) to simulate publication lag. Set 0 if no lag.
    min_periods_z : int
        Minimum non-null months needed before computing a z-score for that feature.
    coverage_min_frac : float
        Minimum fraction of components required to accept a composite score.
    risk_off_hi : float
        Threshold for risk_aversion_score to define risk_off (upper bound).
    risk_on_lo : float
        Threshold (lower bound) for risk_aversion_score to help define risk_on.

    Returns
    -------
    pd.DataFrame
        Columns include:
        - date_col
        - growth_score, risk_aversion_score
        - regime_label (categorical)
        - <feature>_Z for each raw feature
        - coverage / availability / missingness flags
    """
    macro = pd.read_csv(path, parse_dates=[date_col])
    macro = macro.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

    # Normalize column naming / handle synonyms
    # Accept either LEI_YOY or LEI_YOY_SURPRISE
    lei_col = _select_available_name(macro, ["LEI_YOY_SURPRISE", "LEI_YOY"])
    if lei_col and lei_col != "LEI_YOY":
        # Standardize to LEI_YOY internally
        macro.rename(columns={lei_col: "LEI_YOY"}, inplace=True)

    # Create unemployment trend
    if "UNEMPLOYMENT" in macro.columns:
        macro["UNEMPLOYMENT_TREND"] = macro["UNEMPLOYMENT"] - macro["UNEMPLOYMENT"].shift(3)

    # Treasury slope (10y - 2y) if available
    if {"US10Y", "US2Y"}.issubset(macro.columns):
        macro["SLOPE_10Y_2Y"] = macro["US10Y"] - macro["US2Y"]

    # Candidate raw features
    raw_candidates = [
        "LEI_YOY",
        "YIELD_10Y_FED_DIFF",
        "VIX", "VVIX",
        "PUT_CALL",
        "RECESS_PROB",
        "UNEMPLOYMENT",
        "UNEMPLOYMENT_TREND",
        "M_AND_A_AMOUNT",
        "SENTIMENT_BEARISH",
        "CONSUMER_SENTIMENT",
        "SLOPE_10Y_2Y"
    ]
    # Filter to those present
    raw_features = [c for c in raw_candidates if c in macro.columns]

    # Structural availability flags
    for col in raw_features:
        macro[f"{col}_struct_available"] = _first_valid_index_mask(macro[col])

    # Original missing flags
    for col in raw_features:
        macro[f"{col}_orig_missing"] = macro[col].isna().astype(int)

    # Expanding z-scores per feature (only when the value is non-null and enough history exists)
    for col in raw_features:
        macro[f"{col}_Z"] = expanding_zscore(macro[col], min_periods_non_null=min_periods_z)

    # Define component groups (must reference normalized names)
    growth_components = [
        "LEI_YOY_Z",
        "YIELD_10Y_FED_DIFF_Z",
        "UNEMPLOYMENT_TREND_Z",
        "M_AND_A_AMOUNT_Z",
        "CONSUMER_SENTIMENT_Z",
        "SLOPE_10Y_2Y_Z"
    ]
    risk_components = [
        "VIX_Z",
        "VVIX_Z",
        "PUT_CALL_Z",
        "RECESS_PROB_Z",
        "UNEMPLOYMENT_TREND_Z",
        "SENTIMENT_BEARISH_Z"
    ]

    def _composite(df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
        existing = [c for c in cols if c in df.columns]
        if not existing:
            df[f"{prefix}_score"] = np.nan
            df[f"{prefix}_coverage_frac"] = 0.0
            return df
        # Count of non-null components each month
        nn = df[existing].notna().sum(axis=1)
        coverage_frac = nn / len(cols)
        df[f"{prefix}_coverage_frac"] = coverage_frac
        # Mean of available components
        df[f"{prefix}_score"] = df[existing].mean(axis=1)
        # Invalidate if coverage too low
        df.loc[df[f"{prefix}_coverage_frac"] < coverage_min_frac, f"{prefix}_score"] = np.nan
        return df

    macro = _composite(macro, growth_components, "growth")
    macro = _composite(macro, risk_components, "risk_aversion")

    # Regime label
    def _classify(row) -> str:
        g = row["growth_score"]
        r = row["risk_aversion_score"]
        if pd.isna(g) or pd.isna(r):
            return "unknown"
        if (r > risk_off_hi) and (g < 0):
            return "risk_off"
        if (r < risk_on_lo) and (g > 0):
            return "risk_on"
        return "neutral"

    macro["regime_label"] = macro.apply(_classify, axis=1).astype("category")

    # Optional leak-control lag
    if lag_months > 0:
        shift_cols = (
            [c for c in macro.columns if c.endswith("_Z")] +
            ["growth_score", "risk_aversion_score", "regime_label"]
        )
        macro[shift_cols] = macro[shift_cols].shift(lag_months)

    # Drop rows where BOTH composites are NaN after lag (optional; you can keep them)
    # Keeping 'unknown' regime if one composite is missing might still be informative.
    # Here, we keep them; filtering is optional.
    # macro = macro.dropna(subset=["growth_score", "risk_aversion_score"], how="all")

    # Final column ordering
    z_cols = [c for c in macro.columns if c.endswith("_Z")]
    availability_cols = [c for c in macro.columns if c.endswith("_struct_available")]
    missing_cols = [c for c in macro.columns if c.endswith("_orig_missing")]
    coverage_cols = ["growth_coverage_frac", "risk_aversion_coverage_frac"]

    ordered = (
        [date_col, "growth_score", "risk_aversion_score", "regime_label"] #+
       # coverage_cols +
        #z_cols +
       # availability_cols +
      #  missing_cols
    )
    ordered = [c for c in ordered if c in macro.columns]

    return macro[ordered].copy()


def merge_macro_with_equity(equity_df: pd.DataFrame,
                            macro_df: pd.DataFrame,
                            date_col_equity: str = "char_eom",
                            date_col_macro: str = "char_eom") -> pd.DataFrame:
    """
    Left-join macro / regime features into panel by month-end date.
    """
    merged = equity_df.merge(
        macro_df,
        left_on=date_col_equity,
        right_on=date_col_macro,
        how="left"
    )
    return merged