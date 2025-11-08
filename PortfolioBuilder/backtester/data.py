import os
import pandas as pd
import numpy as np
from backtester.config import Paths


def load_results(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.RESULTS_CSV)
    # Parse dates
    for c in ["char_eom", "ret_eom"]:
        df[c] = pd.to_datetime(df[c])
    # Coerce ids
    if "gvkey" in df.columns:
        df["gvkey"] = df["gvkey"].astype(str)
    df["id"] = df["id"].astype(str)
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str)
    if "excntry" in df.columns:
        df["excntry"] = df["excntry"].astype(str)
    df = df.sort_values(["char_eom", "id"]).reset_index(drop=True)
    return df


def load_market(paths: Paths) -> pd.DataFrame:
    """
    Market CSV must have: year, month, ret, rf.
    We'll key on year-month to merge to the realized-month (ret_eom).
    """
    mkt = pd.read_csv(paths.MARKET_CSV)
    if "year" not in mkt.columns or "month" not in mkt.columns:
        raise ValueError("market_data.csv must have 'year' and 'month' columns.")
    mkt["ret"] = mkt["ret"].astype(float)
    mkt["rf"] = mkt["rf"].astype(float)
    mkt["ym"] = mkt["year"].astype(int).astype(str) + "-" + mkt["month"].astype(int).astype(str).str.zfill(2)
    mkt["mkt_excess"] = mkt["ret"] - mkt["rf"]
    return mkt[["ym", "ret", "rf", "mkt_excess"]].copy()


def load_msci_world(paths: Paths) -> pd.DataFrame:
    """
    MSCI World CSV must have: year, month, ret.
    We'll key on year-month to merge to the realized-month (ret_eom).
    """
    msci = pd.read_csv(paths.MSCI_WORLD_CSV)
    if "year" not in msci.columns or "month" not in msci.columns or "ret" not in msci.columns:
        raise ValueError(
            f"MSCI World CSV file must have 'year', 'month', and 'ret' columns. "
            f"File: {paths.MSCI_WORLD_CSV}"
        )
    msci["ret"] = msci["ret"].astype(float)
    msci["ym"] = msci["year"].astype(int).astype(str) + "-" + msci["month"].astype(int).astype(str).str.zfill(2)
    return msci[["ym", "ret"]].copy()


def load_regimes(paths: Paths) -> pd.DataFrame:
    reg = pd.read_csv(paths.REGIME_CSV)
    if "char_eom" not in reg.columns or "state" not in reg.columns:
        raise ValueError("regime_labels.csv must have columns 'char_eom' and 'state'.")
    reg["char_eom"] = pd.to_datetime(reg["char_eom"])
    reg["state"] = reg["state"].astype(str)
    return reg[["char_eom", "state"]].copy()


def attach_market_by_realized_month(results: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Attach market data by the realized month (ret_eom).
    """
    df = results.copy()
    df["ym_ret"] = df["ret_eom"].dt.strftime("%Y-%m")
    df = df.merge(market, left_on="ym_ret", right_on="ym", how="left", validate="m:1")
    if df["ret"].isna().any() or df["rf"].isna().any():
        missing = df[df["ret"].isna() | df["rf"].isna()]["ym_ret"].unique()
        raise ValueError(f"Missing market data for realized months: {missing}")
    df = df.drop(columns=["ym"])
    return df


def attach_regime_by_decision_date(results: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """
    Attach regime state by decision date (char_eom).
    """
    df = results.merge(regimes, on="char_eom", how="left", validate="m:1")
    if df["state"].isna().any():
        missing = df[df["state"].isna()]["char_eom"].dt.strftime("%Y-%m").unique()
        raise ValueError(f"Missing regime state for decision months: {missing}")
    return df


def attach_msci_world_by_realized_month(results: pd.DataFrame, msci_world: pd.DataFrame) -> pd.DataFrame:
    """
    Attach MSCI World data by the realized month (ret_eom).
    """
    df = results.copy()
    df["ym_ret"] = df["ret_eom"].dt.strftime("%Y-%m")

    # Rename the 'ret' column in msci_world before merging to avoid conflicts
    msci_world_renamed = msci_world.rename(columns={"ret": "msci_world_ret"})

    df = df.merge(msci_world_renamed, left_on="ym_ret", right_on="ym", how="left", validate="m:1")

    if df["msci_world_ret"].isna().any():
        missing = df[df["msci_world_ret"].isna()]["ym_ret"].unique()
        raise ValueError(f"Missing MSCI World data for realized months: {missing}")

    df = df.drop(columns=["ym"])
    return df


def get_month_panels(df: pd.DataFrame):
    """
    Iterate cross-sectional panels by decision month (char_eom).
    Realized returns and market series in each panel correspond to the following month (ret_eom).
    """
    for month, g in df.groupby("char_eom"):
        yield month, g.reset_index(drop=True)