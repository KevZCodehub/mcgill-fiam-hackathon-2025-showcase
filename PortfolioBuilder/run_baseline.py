import os
from typing import Dict, Tuple, Set, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.config import Paths, Constraints, StrategyParams, INITIAL_AUM_USD, BPS_TO_DECIMAL
from backtester.data import (
    load_results,
    load_market,
    load_msci_world,
    load_regimes,
    attach_market_by_realized_month,
    attach_regime_by_decision_date,
    attach_msci_world_by_realized_month,
    get_month_panels,
)
from backtester.strategy_baseline import BaselineSectorTiltStrategy
from backtester.optimizer import select_long_short_ids, allocate_weights
from backtester.turnover import weight_based_turnover
from backtester.report import (
    sharpe_excess,
    cagr_from_total,
    ir_vs_market,
    capm_alpha,
    save_plots,
    write_summary,
    max_drawdown,
    max_monthly_loss,
    annualized_std,
    tracking_error,
    returns_by_year,
    write_yearly_returns,
    save_sector_allocation_plot,
    rolling_3year_returns,
    save_rolling_3year_plot,
    calculate_aum_evolution,
    save_aum_plot,
    write_aum_report,
    save_cap_composition_plot,
    market_capture_ratio,
    three_year_beat_benchmark,
)


def classify_cap_size(market_cap: float, threshold: float = 15.0) -> str:
    """
    Classify stock as Large Cap or Small Cap based on market cap.

    Args:
        market_cap: Market capitalization in billions USD
        threshold: Threshold in billions USD (default 15.0)

    Returns:
        'Large' if market_cap >= threshold
        'Small' if market_cap < threshold
        'Unknown' if market_cap is NaN or missing
    """
    if pd.isna(market_cap):
        return 'Unknown'
    return 'Large' if market_cap >= threshold else 'Small'


def classify_cap_size_vectorized(market_caps: pd.Series, threshold: float = 15.0) -> pd.Series:
    """
    Vectorized version of classify_cap_size for better performance.

    Args:
        market_caps: Series of market capitalizations in billions USD
        threshold: Threshold in billions USD (default 15.0)

    Returns:
        Series of classifications ('Large', 'Small', or 'Unknown')
    """
    result = pd.Series('Unknown', index=market_caps.index, dtype=str)
    valid_mask = market_caps.notna()
    result[valid_mask & (market_caps >= threshold)] = 'Large'
    result[valid_mask & (market_caps < threshold)] = 'Small'
    return result


def get_market_cap_column(panel: pd.DataFrame, ids_index: pd.Index) -> pd.Series:
    """
    Extract market cap data from panel, checking column names in priority order.

    Priority: market_equity (primary) -> mktcap (legacy) -> market_cap (legacy)

    Args:
        panel: DataFrame containing stock data
        ids_index: Index of stock IDs to extract

    Returns:
        Series of market caps in billions USD. Returns 0.0 for stocks when market cap
        data is not available (these will be classified as 'Small' cap, which is
        conservative as it applies higher trading costs).
    """
    idx_panel = panel.set_index("id")
    if "market_equity" in panel.columns:
        return idx_panel.reindex(ids_index)["market_equity"]
    elif "mktcap" in panel.columns:
        return idx_panel.reindex(ids_index)["mktcap"]
    elif "market_cap" in panel.columns:
        return idx_panel.reindex(ids_index)["market_cap"]
    else:
        # If market cap not available, return zeros (will be classified as Small cap)
        return pd.Series(0.0, index=ids_index)


def calculate_cap_dependent_costs(w: pd.Series, prev_w: pd.Series, market_caps: pd.Series, cons: Constraints) -> Tuple[
    float, pd.Series]:
    """
    Calculate trading costs based on market cap size.

    Args:
        w: Current weights
        prev_w: Previous weights
        market_caps: Market caps in billions USD for each stock
        cons: Constraints object with cost parameters

    Returns:
        Tuple of (total_cost, cost_by_stock)
    """
    prev_w_aligned = prev_w.reindex(w.index).fillna(0.0)
    trade_abs = (w - prev_w_aligned).abs()

    # Classify each stock by cap size using vectorized operation
    cap_classes = classify_cap_size_vectorized(
        market_caps.reindex(w.index),
        cons.LARGE_CAP_THRESHOLD
    )

    # Calculate cost rates by stock using vectorized map
    cost_rates = cap_classes.map({
        'Large': cons.LARGE_CAP_COST_BPS / BPS_TO_DECIMAL,
        'Small': cons.SMALL_CAP_COST_BPS / BPS_TO_DECIMAL,
        'Unknown': cons.SMALL_CAP_COST_BPS / BPS_TO_DECIMAL  # Conservative: use higher cost
    })

    # Calculate costs per stock
    cost_by_stock = trade_abs * cost_rates
    total_cost = float(cost_by_stock.sum())

    return total_cost, cost_by_stock


def run():
    paths = Paths()
    cons = Constraints()
    params = StrategyParams()

    results = load_results(paths)
    market = load_market(paths)
    msci_world = load_msci_world(paths)
    regimes = load_regimes(paths)

    # Attach market by realized month (ret_eom) and regimes by decision date (char_eom)
    df = attach_market_by_realized_month(results, market)
    df = attach_regime_by_decision_date(df, regimes)
    df = attach_msci_world_by_realized_month(df, msci_world)

    # Containers
    reteom_index = []
    port_excess_rets = []
    port_gross_excess_rets = []  # Track gross returns (before trading costs)
    rf_series = []
    mkt_excess_series = []
    msci_world_series = []
    weights_by_reteom: Dict[pd.Timestamp, pd.Series] = {}
    holdings_rows: List[pd.DataFrame] = []
    sector_allocation_rows: List[Dict] = []
    cap_composition_rows: List[Dict] = []  # Track cap composition over time

    strategy = BaselineSectorTiltStrategy(cons, params)

    prev_w = pd.Series(dtype=float)
    prev_long_ids: Set[str] = set()
    prev_short_ids: Set[str] = set()

    history_df = pd.DataFrame(columns=["state", "sector", "stock_ret", "ret_eom"])

    for char_month, panel in get_month_panels(df):
        # char_month is the decision date (t=0) when portfolio weights are determined
        # ret_month is the realization date (t=1) when returns are realized
        # This ensures proper point-in-time handling: decisions use only past data

        # Strict past-only history for tilts (only use returns realized before char_month)
        past = history_df[history_df["ret_eom"] < char_month]
        if not past.empty:
            strategy.update_history(past)

        regime = panel["state"].mode().iloc[0]
        ret_month = pd.to_datetime(panel["ret_eom"].iloc[0])

        # Prepare scores using information available at char_month (t=0)
        scores_df = strategy.prepare_scores(panel, regime)

        # Decide gross targets (regime vs default)
        if params.ENABLE_REGIME_GROSS_TARGETS:
            desired_long, desired_short = params.REGIME_GROSS_TARGETS.get(regime, params.DEFAULT_GROSS_TARGETS)
        else:
            desired_long, desired_short = params.DEFAULT_GROSS_TARGETS

        # Decide sector caps for this regime (optional)
        if params.ENABLE_REGIME_SECTOR_GROSS_CAPS:
            sector_caps = params.REGIME_SECTOR_GROSS_CAPS.get(regime, {})
        else:
            sector_caps = {}

        long_ids, short_ids = select_long_short_ids(
            scores_df, prev_long_ids, prev_short_ids, params, cons, params.TARGET_HOLDINGS
        )

        w = allocate_weights(
            df=scores_df,
            long_ids=long_ids,
            short_ids=short_ids,
            prev_w=prev_w,
            gross_targets=(desired_long, desired_short),
            constraints=cons,
            smoothing_to_prev=params.SMOOTHING_TO_PREV,
            min_stock_weight_abs=params.MIN_STOCK_WEIGHT_ABS,
            sector_gross_caps=sector_caps,
        )

        # Calculate portfolio return using weights decided at char_month (t=0)
        # and returns realized from char_month to ret_month (t=1)
        realized = panel.set_index("id")["stock_ret"].reindex(w.index).fillna(0.0)
        gross_port_excess = float((w * realized).sum())

        # Get market caps if available, otherwise use conservative (small cap) costs
        market_caps = get_market_cap_column(panel, w.index)

        # Calculate cap-dependent trading costs
        trade_cost, cost_by_stock = calculate_cap_dependent_costs(w, prev_w, market_caps, cons)
        net_port_excess = gross_port_excess - trade_cost

        # Calculate portfolio-level trade absolute for tracking
        prev_w_aligned = prev_w.reindex(w.index).fillna(0.0)
        trade_abs = (w - prev_w_aligned).abs()

        # Track cap composition
        if "market_equity" in panel.columns or "mktcap" in panel.columns or "market_cap" in panel.columns:
            cap_classes = classify_cap_size_vectorized(market_caps, cons.LARGE_CAP_THRESHOLD)
            large_cap_weight = float(w[cap_classes == 'Large'].abs().sum())
            small_cap_weight = float(w[cap_classes == 'Small'].abs().sum())
            unknown_cap_weight = float(w[cap_classes == 'Unknown'].abs().sum())

            cap_composition_rows.append({
                'date': ret_month,
                'large_cap_gross': large_cap_weight,
                'small_cap_gross': small_cap_weight,
                'unknown_cap_gross': unknown_cap_weight
            })

        long_gross = float(w[w > 0].sum())
        short_gross = float(-w[w < 0].sum())
        total_gross = long_gross + short_gross
        net_exposure = float(w.sum())

        # Record portfolio returns indexed by ret_month (realization date, t=1)
        # This ensures all performance metrics are calculated at the correct time
        reteom_index.append(ret_month)
        port_excess_rets.append(net_port_excess)
        port_gross_excess_rets.append(gross_port_excess)  # Track gross returns
        rf_series.append(float(panel["rf"].iloc[0]))
        mkt_excess_series.append(float(panel["mkt_excess"].iloc[0]))
        msci_world_series.append(float(panel["msci_world_ret"].iloc[0]))

        weights_by_reteom[ret_month] = w.copy()

        # Track sector allocations
        if (w.abs() > 0).any():
            idx_panel = panel.set_index("id")
            sectors_for_weights = idx_panel.reindex(w.index)["sector"].astype(str)

            # Calculate gross exposure by sector
            sector_gross = pd.DataFrame({
                'weight': w,
                'sector': sectors_for_weights
            }).groupby('sector')['weight'].apply(lambda x: x.abs().sum())

            for sector, gross_exp in sector_gross.items():
                sector_allocation_rows.append({
                    'date': ret_month,
                    'sector': sector,
                    'gross_exposure': gross_exp
                })

        # Holdings rows
        if (w.abs() > 0).any():
            idx_panel = panel.set_index("id")
            idx_scores = scores_df.set_index("id")

            active = w[w.abs() > 0].to_frame(name="weight")
            active["position"] = np.where(active["weight"] > 0, "long", "short")
            active["abs_weight"] = active["weight"].abs()

            active["realized_return_excess"] = realized.reindex(active.index)
            active["pnl_contribution_excess"] = active["weight"] * active["realized_return_excess"]

            active["weight_prev"] = prev_w_aligned.reindex(active.index)
            active["trade_abs"] = (active["weight"] - active["weight_prev"]).abs()
            active["trade_cost"] = cost_by_stock.reindex(active.index).fillna(0.0)

            # Add market cap and cap classification using helper function
            active_market_caps = get_market_cap_column(panel, active.index)
            active["market_cap"] = active_market_caps
            active["cap_class"] = classify_cap_size_vectorized(
                active["market_cap"], cons.LARGE_CAP_THRESHOLD
            )

            active["sector"] = idx_panel.reindex(active.index)["sector"].astype(str)
            active["excntry"] = idx_panel.reindex(active.index)["excntry"].astype(
                str) if "excntry" in idx_panel.columns else "Unknown"
            active["lightgbm"] = idx_panel.reindex(active.index)["lightgbm"]
            active["beta_60m"] = idx_panel.reindex(active.index)[
                "beta_60m"] if "beta_60m" in idx_panel.columns else np.nan
            active["adj_score"] = idx_scores.reindex(active.index)["adj_score"]
            active["decile"] = idx_scores.reindex(active.index)["decile"]

            # Include sector cap used (if enabled)
            if params.ENABLE_REGIME_SECTOR_GROSS_CAPS and sector_caps:
                active["sector_cap"] = active["sector"].map(sector_caps).astype(float)
            else:
                active["sector_cap"] = np.nan

            # Include country cap
            active["country_cap"] = cons.COUNTRY_GROSS_CAP

            active["char_eom"] = char_month
            active["ret_eom"] = ret_month
            active["state"] = regime
            active["rf"] = float(panel["rf"].iloc[0])
            active["mkt_excess"] = float(panel["mkt_excess"].iloc[0])
            active["long_gross"] = long_gross
            active["short_gross"] = short_gross
            active["total_gross"] = total_gross
            active["net_exposure"] = net_exposure
            active["portfolio_trade_abs_total"] = trade_abs.sum()
            active["portfolio_trade_cost_total"] = trade_cost

            active = active.reset_index().rename(columns={"index": "id"})
            holdings_rows.append(active)

        prev_w = w.copy()
        prev_long_ids = set([i for i, val in w.items() if val > 0])
        prev_short_ids = set([i for i, val in w.items() if val < 0])

        history_df = pd.concat(
            [history_df, panel[["state", "sector", "stock_ret", "ret_eom"]].copy()],
            axis=0, ignore_index=True
        )

    # Build realized-time series indexed by ret_eom (realization date)
    # All returns are indexed by ret_eom, which is when returns are realized (t=1)
    # Investment decisions were made at char_eom (t=0)
    index = pd.to_datetime(pd.Index(reteom_index))
    port_excess = pd.Series(port_excess_rets, index=index, name="port_excess")
    port_gross_excess = pd.Series(port_gross_excess_rets, index=index, name="port_gross_excess")
    rf = pd.Series(rf_series, index=index, name="rf")
    mkt_excess = pd.Series(mkt_excess_series, index=index, name="mkt_excess")
    msci_world_ret = pd.Series(msci_world_series, index=index, name="msci_world_ret")

    # Calculate NET performance metrics using realized returns (indexed by ret_eom)
    sh_net = sharpe_excess(port_excess)
    port_total_net = port_excess + rf
    cagr_net = cagr_from_total(port_total_net)
    ir_net = ir_vs_market(port_excess, mkt_excess)
    # CAPM alpha is calculated using returns indexed by ret_eom (realization date)
    alpha_ann_net, beta_net = capm_alpha(port_excess, mkt_excess)

    # Calculate additional NET statistics
    ann_std_net = annualized_std(port_excess)
    max_dd_net = max_drawdown(port_total_net)
    max_loss_net = max_monthly_loss(port_total_net)
    te_net = tracking_error(port_excess, mkt_excess)

    # Calculate GROSS performance metrics (before trading costs)
    sh_gross = sharpe_excess(port_gross_excess)
    port_total_gross = port_gross_excess + rf
    cagr_gross = cagr_from_total(port_total_gross)
    ir_gross = ir_vs_market(port_gross_excess, mkt_excess)
    alpha_ann_gross, beta_gross = capm_alpha(port_gross_excess, mkt_excess)

    # Calculate additional GROSS statistics
    ann_std_gross = annualized_std(port_gross_excess)
    max_dd_gross = max_drawdown(port_total_gross)
    max_loss_gross = max_monthly_loss(port_total_gross)
    te_gross = tracking_error(port_gross_excess, mkt_excess)

    avg_turnover = weight_based_turnover(weights_by_reteom)  # Weight-based turnover method

    total_gross_list, long_gross_list, short_gross_list, holdings_count = [], [], [], []
    for _, w_m in weights_by_reteom.items():
        lg = w_m[w_m > 0].sum()
        sg = -w_m[w_m < 0].sum()
        total_gross_list.append(lg + sg)
        long_gross_list.append(lg)
        short_gross_list.append(sg)
        holdings_count.append((w_m.abs() > 1e-9).sum())

    # Calculate average cap composition
    if cap_composition_rows:
        cap_comp_df = pd.DataFrame(cap_composition_rows)
        avg_large_cap = float(cap_comp_df['large_cap_gross'].mean())
        avg_small_cap = float(cap_comp_df['small_cap_gross'].mean())
        avg_unknown_cap = float(cap_comp_df['unknown_cap_gross'].mean())
        total_cap_exposure = avg_large_cap + avg_small_cap + avg_unknown_cap

        # Calculate percentages
        if total_cap_exposure > 0:
            pct_large_cap = avg_large_cap / total_cap_exposure
            pct_small_cap = avg_small_cap / total_cap_exposure
            pct_unknown_cap = avg_unknown_cap / total_cap_exposure
        else:
            pct_large_cap = pct_small_cap = pct_unknown_cap = 0.0
    else:
        avg_large_cap = avg_small_cap = avg_unknown_cap = 0.0
        pct_large_cap = pct_small_cap = pct_unknown_cap = 0.0

    # Calculate market total return (used for both 3-year beat and market capture)
    mkt_total = mkt_excess + rf

    # Calculate 3-year beat benchmark percentage (net returns only)
    three_yr_beat_pct = three_year_beat_benchmark(port_total_net, mkt_total)

    metrics = {
        # NET metrics (after trading costs)
        "net_avg_annualized_return": cagr_net,
        "net_annualized_std_deviation": ann_std_net,
        "net_sharpe_ratio_annualized": sh_net,
        "net_capm_alpha_annualized": alpha_ann_net,
        "net_information_ratio_annualized": ir_net,
        "net_maximum_drawdown": max_dd_net,
        "net_maximum_monthly_loss": max_loss_net,
        "net_tracking_error_annualized": te_net,
        "net_capm_beta": beta_net,

        # GROSS metrics (before trading costs)
        "gross_avg_annualized_return": cagr_gross,
        "gross_annualized_std_deviation": ann_std_gross,
        "gross_sharpe_ratio_annualized": sh_gross,
        "gross_capm_alpha_annualized": alpha_ann_gross,
        "gross_information_ratio_annualized": ir_gross,
        "gross_maximum_drawdown": max_dd_gross,
        "gross_maximum_monthly_loss": max_loss_gross,
        "gross_tracking_error_annualized": te_gross,
        "gross_capm_beta": beta_gross,

        # Portfolio characteristics
        "portfolio_turnover": float(avg_turnover if not np.isnan(avg_turnover) else 0.0),
        "three_year_pct_beat_benchmark": three_yr_beat_pct,
        "avg_total_gross_exposure": float(np.mean(total_gross_list)),
        "avg_long_gross_exposure": float(np.mean(long_gross_list)),
        "avg_short_gross_exposure": float(np.mean(short_gross_list)),
        "avg_holdings": float(np.mean(holdings_count)),
        "avg_large_cap_gross_exposure": avg_large_cap,
        "avg_small_cap_gross_exposure": avg_small_cap,
        "avg_unknown_cap_gross_exposure": avg_unknown_cap,
        "pct_large_cap_of_total": pct_large_cap,
        "pct_small_cap_of_total": pct_small_cap,
        "pct_unknown_cap_of_total": pct_unknown_cap,
    }

    os.makedirs(paths.OUTPUT_DIR, exist_ok=True)
    save_plots(paths.OUTPUT_DIR, port_excess, rf, mkt_excess, msci_world_ret)

    # Calculate market capture ratios for both gross and net returns
    # Create separate DataFrames for net and gross, then combine results
    # Data is already monthly, no frequency conversion needed
    returns_net = pd.DataFrame({
        'Portfolio (Net)': port_total_net,
        'S&P 500 (Net Benchmark)': mkt_total
    })
    returns_gross = pd.DataFrame({
        'Portfolio (Gross)': port_total_gross,
        'S&P 500 (Gross Benchmark)': mkt_total
    })

    market_capture_net = market_capture_ratio(returns_net)
    market_capture_gross = market_capture_ratio(returns_gross)
    # Concatenate results with unique index names
    market_capture_df = pd.concat([market_capture_net, market_capture_gross])

    write_summary(paths.OUTPUT_DIR, metrics, market_capture_df)

    # Generate portfolio returns report (monthly returns with ret_eom dates)
    portfolio_returns_df = pd.DataFrame({
        'ret_eom': port_excess.index,
        'portfolio_excess_return_net': port_excess.values,
        'portfolio_total_return_net': port_total_net.values,
        'portfolio_excess_return_gross': port_gross_excess.values,
        'portfolio_total_return_gross': port_total_gross.values,
        'market_excess_return': mkt_excess.values,
        'market_total_return': (mkt_excess + rf).values,
        'rf_rate': rf.values
    })
    portfolio_returns_df.to_csv(os.path.join(paths.OUTPUT_DIR, "portfolio_returns.csv"), index=False)
    print(f"Portfolio returns CSV written to: {os.path.join(paths.OUTPUT_DIR, 'portfolio_returns.csv')}")

    # Generate yearly returns report (use net returns)
    yearly_rets = returns_by_year(port_total_net)
    write_yearly_returns(paths.OUTPUT_DIR, yearly_rets)

    # Generate sector allocation plot
    if sector_allocation_rows:
        sector_alloc_df = pd.DataFrame(sector_allocation_rows)
        save_sector_allocation_plot(paths.OUTPUT_DIR, sector_alloc_df)

    # Generate rolling 3-year returns plot (use net returns)
    save_rolling_3year_plot(paths.OUTPUT_DIR, port_total_net)

    # Generate AUM evolution reports (using configured initial AUM from config, use net returns)
    aum = calculate_aum_evolution(port_total_net, initial_aum=INITIAL_AUM_USD)

    # Calculate AUM for benchmarks (S&P 500 and MSCI World)
    # Note: mkt_total was calculated earlier for 3-year beat benchmark metric
    sp500_aum = calculate_aum_evolution(mkt_total, initial_aum=INITIAL_AUM_USD)
    msci_world_aum = calculate_aum_evolution(msci_world_ret, initial_aum=INITIAL_AUM_USD)

    save_aum_plot(paths.OUTPUT_DIR, aum, sp500_aum, msci_world_aum)
    write_aum_report(paths.OUTPUT_DIR, aum, initial_aum=INITIAL_AUM_USD)

    # Save cap composition data
    if cap_composition_rows:
        cap_comp_df = pd.DataFrame(cap_composition_rows)
        cap_comp_df.to_csv(os.path.join(paths.OUTPUT_DIR, "cap_composition.csv"), index=False)
        save_cap_composition_plot(paths.OUTPUT_DIR, cap_comp_df)

    if holdings_rows:
        holdings_df = pd.concat(holdings_rows, axis=0, ignore_index=True)
        col_order = [
            "ret_eom", "char_eom", "id", "sector", "sector_cap", "excntry", "country_cap",
            "position", "weight", "abs_weight", "weight_prev", "trade_abs", "trade_cost",
            "market_cap", "cap_class",
            "realized_return_excess", "pnl_contribution_excess",
            "state", "decile", "adj_score", "lightgbm", "beta_60m",
            "rf", "mkt_excess",
            "long_gross", "short_gross", "total_gross", "net_exposure",
            "portfolio_trade_abs_total", "portfolio_trade_cost_total",
        ]
        col_order = [c for c in col_order if c in holdings_df.columns]
        holdings_df = holdings_df[col_order + [c for c in holdings_df.columns if c not in col_order]]
        out_path = os.path.join(paths.OUTPUT_DIR, "holdings.csv")
        holdings_df.sort_values(["ret_eom", "position", "abs_weight"], ascending=[True, False, False]).to_csv(out_path,
                                                                                                              index=False)
        print(f"Holdings CSV written to: {out_path}")

        # Output monthly holdings files
        monthly_dir = os.path.join(paths.OUTPUT_DIR, "monthly")
        os.makedirs(monthly_dir, exist_ok=True)

        # Group by ret_eom and save each month to a separate file
        for ret_eom_date, month_df in holdings_df.groupby('ret_eom'):
            # Format the date as YYYY-MM-DD (e.g., 2017-01-31)
            # ret_eom_date is already a pandas Timestamp from groupby
            date_str = ret_eom_date.strftime('%Y-%m-%d')
            monthly_path = os.path.join(monthly_dir, f"{date_str}.csv")
            # Sort by position (short before long) and abs_weight (largest first)
            # Consistent with main holdings.csv sorting, excluding ret_eom since each file has one date
            month_df.sort_values(["position", "abs_weight"], ascending=[False, False]).to_csv(monthly_path, index=False)

        print(f"Monthly holdings CSVs written to: {monthly_dir} ({len(holdings_df['ret_eom'].unique())} files)")

    print("Backtest complete. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"Outputs saved to {paths.OUTPUT_DIR}")


if __name__ == "__main__":
    run()