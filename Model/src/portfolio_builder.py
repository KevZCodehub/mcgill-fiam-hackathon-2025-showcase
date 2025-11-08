import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import statsmodels.api as sm
import csv
from typing import Optional, List, Literal, Dict, Any


# ==================================================
# Configuration
# ==================================================

@dataclass
class PortfolioConfig:
    # Core
    score_col: str = "lightgbm"
    return_col: str = "stock_ret"
    date_col: str = "ret_eom"
    id_col: str = "id"
    gvkey_col: str = "gvkey"

    # Selection mode
    selection_mode: Literal["fixed", "decile"] = "decile"  # "fixed" (top N) or "decile" (top/bottom %)
    max_positions: int = 200  # used only if selection_mode == "fixed"
    decile_pct: float = 0.10  # fraction for top/bottom if selection_mode == "decile"

    weighting_scheme: str = "equal"  # "equal" | "score" | "rank"
    weight_cap: Optional[float] = None

    annualization_factor: int = 12
    min_obs_per_month: int = 50

    # Market factors (simple CAPM)
    market_csv: Optional[Path] = None
    market_date_merge_cols: Optional[List[str]] = None  # e.g. ["year","month"]
    rf_col: str = "rf"
    raw_mkt_col: str = "ret"  # raw total market return (if supplied)
    mkt_col: str = "mktrf"  # market excess (mkt - rf) will be derived if missing

    # Output
    out_dir: Path = Path("portfolio_outputs")
    save_holdings: bool = True
    save_perf: bool = True
    save_counts: bool = True
    save_top_holdings: bool = True
    verbose: bool = True


# ==================================================
# Helpers
# ==================================================

def compute_spread_oos_r2_from_holdings(holdings: pd.DataFrame,
                                        cfg: PortfolioConfig) -> dict:
    """
    Computes uncentered OOS R^2 for the long-short spread time series.
    Assumes holdings already have: date_col, portfolio, weight (EW if you used equal),
    score_col, return_col.
    """
    if holdings.empty:
        return {}
    date_col = cfg.date_col
    ret_col = cfg.return_col
    score_col = cfg.score_col

    rows = []
    for dt, g in holdings.groupby(date_col, observed=True):
        # We prefer using the union of all stocks in the original *test* month,
        # but here we only have selected holdings. For a purer measure,
        # pass in full test predictions instead of holdings.
        # This version uses holdings only (biased upward if tails have higher signal).
        if g.empty:
            continue
        # Fit b_t through origin on combined long+short subset
        y = g[ret_col].values
        s = g[score_col].values
        ss = np.sum(s**2)
        if ss <= 0:
            continue
        b_t = np.sum(y * s) / ss
        g['pred_ret'] = b_t * g[score_col]

        long_mean_pred = g.loc[g['portfolio'] == 'LONG', 'pred_ret'].mean()
        short_mean_pred = g.loc[g['portfolio'] == 'SHORT', 'pred_ret'].mean()
        long_mean_real = g.loc[g['portfolio'] == 'LONG', ret_col].mean()
        short_mean_real = g.loc[g['portfolio'] == 'SHORT', ret_col].mean()

        rows.append({
            date_col: dt,
            'b_t': b_t,
            'pred_long': long_mean_pred,
            'pred_short': short_mean_pred,
            'real_long': long_mean_real,
            'real_short': short_mean_real,
            'pred_spread': long_mean_pred - short_mean_pred,
            'real_spread': long_mean_real - short_mean_real
        })

    df_ts = pd.DataFrame(rows).sort_values(date_col)
    if df_ts.empty:
        return {}

    y = df_ts['real_spread'].values
    yhat = df_ts['pred_spread'].values
    denom = np.sum(y**2)
    if denom <= 0:
        oos_r2 = np.nan
    else:
        oos_r2 = 1.0 - np.sum((y - yhat)**2) / denom

    return {
        'spread_oos_r2': oos_r2,
        'n_months': len(df_ts),
        'detail': df_ts
    }

def _normalize_gvkey(df: pd.DataFrame, gvkey_col: str):
    if gvkey_col in df.columns:
        df[gvkey_col] = (
            df[gvkey_col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .where(lambda s: ~s.isin(["nan", "None"]), None)
        )


def compute_weights(sub: pd.DataFrame, cfg: PortfolioConfig, side: str) -> pd.Series:
    s = sub[cfg.score_col].astype(float)
    n = len(sub)
    if n == 0:
        return pd.Series([], dtype=float)

    if cfg.weighting_scheme == "equal":
        w = pd.Series(1.0 / n, index=sub.index)

    elif cfg.weighting_scheme == "score":
        if side == "long":
            base = s - s.min()
        else:
            base = s.max() - s
        if (base <= 0).all():
            base = pd.Series(1.0, index=sub.index)
        w = base / base.sum()

    elif cfg.weighting_scheme == "rank":
        if side == "long":
            ranks = s.rank(method="first")
        else:
            ranks = s.rank(method="first", ascending=True)
        w = ranks / ranks.sum()
    else:
        raise ValueError(f"Unknown weighting_scheme={cfg.weighting_scheme}")

    # Optional capping
    if cfg.weight_cap is not None:
        cap = cfg.weight_cap
        if cap <= 0:
            raise ValueError("weight_cap must be positive.")
        loop_ct = 0
        while (w > cap).any():
            excess = (w - cap).clip(lower=0)
            pool = excess.sum()
            w = w.clip(upper=cap)
            rem = w[w < cap]
            if rem.empty:
                w[:] = 1.0 / len(w)
                break
            scale = pool / rem.sum()
            w.loc[rem.index] += w.loc[rem.index] * scale
            loop_ct += 1
            if loop_ct > 10:
                break
        w /= w.sum()

    return w


# ==================================================
# Portfolio Construction
# ==================================================

def build_monthly_holdings(pred: pd.DataFrame, cfg: PortfolioConfig):
    """
    Returns:
      holdings: long & short rows
      counts: per-month counts used (k_long, k_short)
    Selection reconciliation:
      - main.py 'Annualized Long' = average monthly top decile return * 12 (equal-weight within decile)
      - This implementation reproduces that if selection_mode='decile' and weighting_scheme='equal'
    """
    holdings_records = []
    count_records = []

    for dt, g in pred.groupby(cfg.date_col, observed=True):
        N = len(g)
        if N < cfg.min_obs_per_month:
            continue

        # Determine counts
        if cfg.selection_mode == "decile":
            k = max(1, int(N * cfg.decile_pct))
        else:  # fixed
            k = min(cfg.max_positions, N // 2)  # ensure space for both sides
        k_long = k
        k_short = k

        g_sorted = g.sort_values(cfg.score_col)

        short_subset = g_sorted.head(k_short).copy()
        long_subset = g_sorted.tail(k_long).copy()

        w_long = compute_weights(long_subset, cfg, "long")
        w_short = compute_weights(short_subset, cfg, "short")

        long_subset["weight"] = w_long
        long_subset["portfolio"] = "LONG"
        long_subset["side"] = "long"

        short_subset["weight"] = w_short
        short_subset["portfolio"] = "SHORT"
        short_subset["side"] = "short"

        combined = pd.concat([long_subset, short_subset], axis=0)
        combined["contrib_ret"] = combined["weight"] * combined[cfg.return_col]
        holdings_records.append(combined)

        count_records.append({
            cfg.date_col: dt,
            "universe_size": N,
            "k_long": k_long,
            "k_short": k_short
        })

    if not holdings_records:
        holdings = pd.DataFrame(columns=[
            cfg.date_col, cfg.id_col, cfg.gvkey_col, "portfolio", "side",
            "weight", cfg.return_col, cfg.score_col, "contrib_ret"
        ])
        counts = pd.DataFrame(columns=[cfg.date_col, "universe_size", "k_long", "k_short"])
    else:
        holdings = pd.concat(holdings_records, ignore_index=True)
        counts = pd.DataFrame(count_records)

    _normalize_gvkey(holdings, cfg.gvkey_col)
    return holdings, counts


def compute_performance(holdings: pd.DataFrame,
                        cfg: PortfolioConfig,
                        market_factors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()

    port_rets = (
        holdings
        .groupby([cfg.date_col, "portfolio"], observed=True)
        .apply(lambda df: df["contrib_ret"].sum())
        .rename("ret")
        .reset_index()
    )
    pivot = port_rets.pivot(index=cfg.date_col, columns="portfolio", values="ret").sort_index()
    for col in ["LONG", "SHORT"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["LONG_SHORT"] = pivot["LONG"] - pivot["SHORT"]

    if market_factors is not None and not market_factors.empty:
        pf = pivot.reset_index()
        if cfg.market_date_merge_cols:
            if "year" not in pf:  # derive
                pf["year"] = pd.to_datetime(pf[cfg.date_col]).dt.year
            if "month" not in pf:
                pf["month"] = pd.to_datetime(pf[cfg.date_col]).dt.month
            pf = pf.merge(market_factors, how="left", on=cfg.market_date_merge_cols)
        else:
            if cfg.date_col not in market_factors.columns:
                raise ValueError(f"Market factors need {cfg.date_col} or supply merge keys.")
            pf = pf.merge(market_factors, how="left", on=cfg.date_col)

        if cfg.rf_col in pf.columns and cfg.raw_mkt_col in pf.columns and cfg.raw_mkt_col:
            if cfg.mkt_col not in pf.columns:
                pf[cfg.mkt_col] = pf[cfg.raw_mkt_col] - pf[cfg.rf_col]
            for c in ["LONG", "SHORT", "LONG_SHORT"]:
                pf[f"{c}_EXCESS"] = pf[c] - pf[cfg.rf_col]

        pivot = pf.set_index(cfg.date_col)

    return pivot


# ==================================================
# Metrics
# ==================================================

def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def compute_turnover(holdings: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    rows = []
    for port in ["LONG", "SHORT"]:
        sub = holdings[holdings["portfolio"] == port]
        panel = sub.pivot_table(index=cfg.date_col,
                                columns=cfg.id_col,
                                values="weight",
                                aggfunc="sum").fillna(0.0).sort_index()
        prev = None
        for dt, row in panel.iterrows():
            if prev is None:
                prev = row
                continue
            to = 0.5 * np.abs(row - prev).sum()
            rows.append({"portfolio": port, cfg.date_col: dt, "turnover": to})
            prev = row
    return pd.DataFrame(rows)


def summarize_performance(perf: pd.DataFrame, cfg: PortfolioConfig, label: str) -> dict:
    if perf.empty or label not in perf.columns:
        return {}
    r = perf[label].dropna()
    if r.empty:
        return {}
    mean_m = r.mean()
    vol_m = r.std(ddof=1)
    ann_r = mean_m * cfg.annualization_factor
    ann_v = vol_m * np.sqrt(cfg.annualization_factor)
    sharpe = ann_r / (ann_v + 1e-12)
    mdd = max_drawdown(r)
    max_one_month_loss = r.min()
    calmar = ann_r / abs(mdd + 1e-12) if pd.notnull(mdd) else np.nan
    win = (r > 0).mean()
    return dict(
        mean_monthly=mean_m,
        ann_return=ann_r,
        ann_vol=ann_v,
        sharpe=sharpe,
        max_drawdown=mdd,
        max_monthly_loss=max_one_month_loss,
        calmar=calmar,
        win_rate=win,
        n_months=len(r)
    )


def capm_alpha(perf: pd.DataFrame, cfg: PortfolioConfig, label: str = "LONG") -> dict:
    if perf.empty:
        return {}
    ret_col = f"{label}_EXCESS" if f"{label}_EXCESS" in perf.columns else label
    if cfg.mkt_col not in perf.columns:
        return {}
    df = perf[[ret_col, cfg.mkt_col]].dropna().copy()
    if df.empty:
        return {}
    X = sm.add_constant(df[cfg.mkt_col])
    y = df[ret_col]
    model = sm.OLS(y, X).fit()
    alpha = model.params.get("const", np.nan)
    beta = model.params.get(cfg.mkt_col, np.nan)
    alpha_t = model.tvalues.get("const", np.nan)
    resid_std = model.resid.std(ddof=1)
    ir_alpha = alpha / (resid_std + 1e-12) * np.sqrt(cfg.annualization_factor)
    return {
        "alpha_month": alpha,
        "alpha_ann": alpha * (cfg.annualization_factor),
        "alpha_t": alpha_t,
        "beta": beta,
        "information_ratio_alpha": ir_alpha,
        "n_obs": int(df.shape[0])
    }


# ==================================================
# Orchestrator
# ==================================================

def run_portfolio_builder(pred_csv: str,
                          cfg: PortfolioConfig,
                          market_csv: Optional[str] = None) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_csv, parse_dates=["date", cfg.date_col])
    required_cols = [cfg.score_col, cfg.return_col, cfg.id_col, cfg.gvkey_col, cfg.date_col]
    for c in required_cols:
        if c not in pred.columns:
            raise ValueError(f"Missing required column '{c}' in predictions.")
    _normalize_gvkey(pred, cfg.gvkey_col)

    holdings, counts = build_monthly_holdings(pred, cfg)
    if holdings.empty:
        print("No holdings generated.")
        return {}

    market_factors = None
    if market_csv:
        market_factors = pd.read_csv(market_csv)
        if cfg.market_date_merge_cols and not all(k in market_factors.columns for k in cfg.market_date_merge_cols):
            raise ValueError(f"Market file must contain merge keys {cfg.market_date_merge_cols}")
        # Derive market excess if raw return present
        if cfg.raw_mkt_col and cfg.rf_col in market_factors.columns and cfg.raw_mkt_col in market_factors.columns:
            if cfg.mkt_col not in market_factors.columns:
                market_factors[cfg.mkt_col] = market_factors[cfg.raw_mkt_col] - market_factors[cfg.rf_col]

    perf = compute_performance(holdings, cfg, market_factors=market_factors)

    turnover_df = compute_turnover(holdings, cfg)

    # Summaries
    summary_long = summarize_performance(perf, cfg, "LONG")
    summary_short = summarize_performance(perf, cfg, "SHORT")
    summary_ls = summarize_performance(perf, cfg, "LONG_SHORT")

    capm_long = capm_alpha(perf, cfg, "LONG")
    capm_short = capm_alpha(perf, cfg, "SHORT")
    capm_ls = capm_alpha(perf, cfg, "LONG_SHORT")




    if cfg.verbose:
        print("\n=== Portfolio Performance Summary ===")

        def _print_block(name, data):
            print(f"\n[{name}]")
            if not data:
                print("  (no data)")
                return
            for k, v in data.items():
                if isinstance(v, (int, float, np.floating)):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")

        _print_block("LONG (returns)", summary_long)
        _print_block("SHORT (returns)", summary_short)
        _print_block("LONG_SHORT (returns)", summary_ls)
        _print_block("CAPM LONG", capm_long)
        _print_block("CAPM SHORT", capm_short)
        _print_block("CAPM LONG_SHORT", capm_ls)

        if not turnover_df.empty:
            tl = turnover_df[turnover_df["portfolio"] == "LONG"]["turnover"].mean()
            ts = turnover_df[turnover_df["portfolio"] == "SHORT"]["turnover"].mean()
            print(f"\nAverage Turnover  LONG: {tl:.4f}  SHORT: {ts:.4f}")

        if not counts.empty:
            avg_k_long = counts["k_long"].mean()
            avg_k_short = counts["k_short"].mean()
            print(f"\nAverage k_long: {avg_k_long:.1f}  Average k_short: {avg_k_short:.1f}")
            print("Sample of monthly counts:")
            print(counts.tail(5))

    # Save holdings
    if cfg.save_holdings:
        holdings_out = holdings[[cfg.date_col, cfg.id_col, cfg.gvkey_col,
                                 "portfolio", "side", "weight",
                                 cfg.return_col, cfg.score_col, "contrib_ret"]].copy()
        holdings_path = cfg.out_dir / "portfolio_holdings.csv"
        holdings_out.to_csv(holdings_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        if cfg.verbose:
            print(f"\nSaved holdings -> {holdings_path}")

    # Save counts
    if cfg.save_counts and not counts.empty:
        counts_path = cfg.out_dir / "portfolio_counts.csv"
        counts.to_csv(counts_path, index=False)
        if cfg.verbose:
            print(f"Saved counts -> {counts_path}")

    # Save top-holdings (latest month long decile)
    if cfg.save_top_holdings and not holdings.empty:
        last_date = holdings[cfg.date_col].max()
        latest_long = holdings[(holdings["portfolio"] == "LONG") &
                               (holdings[cfg.date_col] == last_date)].copy()
        top_path = cfg.out_dir / "top_holdings_latest.csv"
        latest_long.sort_values("weight", ascending=False).to_csv(top_path, index=False)
        if cfg.verbose:
            print(f"Saved top holdings (latest month) -> {top_path}")

    # Save performance
    if cfg.save_perf:
        perf_out = perf.reset_index()
        perf_path = cfg.out_dir / "portfolio_performance.csv"
        perf_out.to_csv(perf_path, index=False)
        if cfg.verbose:
            print(f"Saved performance -> {perf_path}")
    spread_r2_stats = compute_spread_oos_r2_from_holdings(holdings, cfg)
    if cfg.verbose and spread_r2_stats:
        print(f"\nSpread OOS R^2 (uncentered, holdings-based): {spread_r2_stats['spread_oos_r2']:.4f}")
        spread_r2_stats['detail'].to_csv(cfg.out_dir / "spread_oos_r2_detail.csv", index=False)
    return {
        "holdings": holdings,
        "counts": counts,
        "performance": perf,
        "turnover": turnover_df,
        "summary": {
            "LONG": {"returns": summary_long, "capm": capm_long},
            "SHORT": {"returns": summary_short, "capm": capm_short},
            "LONG_SHORT": {"returns": summary_ls, "capm": capm_ls}
        }
    }


# ==================================================
# Script Entry
# ==================================================

if __name__ == "__main__":
    # Example usage; adjust paths
    pred_csv_path = "/home/victorgaron/PycharmProjects/Finalv2/data/results.csv"
    market_csv_path = "/home/victorgaron/PycharmProjects/Finalv2/data/market.csv"  # expected: year, month, rf, ret (optional raw market return)

    cfg = PortfolioConfig(
        selection_mode="fixed",  # ensures comparability with main.py decile logic
        max_positions=100,
        weighting_scheme="score",  # use equal for direct comparability to average bin returns
        market_csv=Path(market_csv_path),
        market_date_merge_cols=["year", "month"],
        raw_mkt_col="ret",  # set empty string "" if you don't have raw market and only have excess
        out_dir=Path("portfolio_outputs"),
        verbose=True
    )

    run_portfolio_builder(pred_csv_path, cfg, market_csv=market_csv_path)