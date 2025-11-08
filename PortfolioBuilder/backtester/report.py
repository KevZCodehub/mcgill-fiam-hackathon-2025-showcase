import os
import json
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtester.config import INITIAL_AUM_USD


def annualize_mean_std(monthly_returns: pd.Series):
    mu = monthly_returns.mean()
    sd = monthly_returns.std(ddof=0)
    ann_mu = (1 + mu) ** 12 - 1
    ann_sd = sd * np.sqrt(12)
    return float(ann_mu), float(ann_sd)


def sharpe_excess(monthly_excess: pd.Series):
    # Excess returns already net of rf
    sd = monthly_excess.std(ddof=0)
    if sd <= 0:
        return 0.0
    return float(np.sqrt(12) * monthly_excess.mean() / sd)


def cagr_from_total(total_monthly: pd.Series):
    # total_monthly = rf + excess, we receive directly the total monthly series to compound
    cum = (1 + total_monthly).prod()
    n = len(total_monthly)
    if n == 0:
        return 0.0
    return float(cum ** (12 / n) - 1)


def ir_vs_market(port_excess: pd.Series, mkt_excess: pd.Series):
    active = (port_excess - mkt_excess).dropna()
    sd = active.std(ddof=0)
    if sd <= 0:
        return 0.0
    return float(np.sqrt(12) * active.mean() / sd)


def capm_alpha(port_excess: pd.Series, mkt_excess: pd.Series):
    """
    Calculate CAPM alpha and beta using realized returns.

    Both port_excess and mkt_excess should be indexed by ret_eom (realization date),
    ensuring that alpha is calculated using returns realized at the same time periods.

    Simple OLS: port_excess = alpha + beta * mkt_excess + eps

    Args:
        port_excess: Portfolio excess returns indexed by ret_eom
        mkt_excess: Market excess returns indexed by ret_eom

    Returns:
        Tuple of (annualized_alpha, beta)
    """
    aligned = pd.concat([port_excess, mkt_excess], axis=1, join="inner").dropna()
    if aligned.shape[0] < 3:
        return 0.0, 0.0
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    x1 = np.vstack([np.ones_like(x), x]).T
    # OLS
    beta_hat = np.linalg.lstsq(x1, y, rcond=None)[0]
    alpha_m, beta_m = beta_hat[0], beta_hat[1]
    # Annualize alpha (approx via compounding monthly alpha)
    alpha_ann = alpha_m * 12
    return float(alpha_ann), float(beta_m)


def drawdown_curve(returns: pd.Series):
    growth = (1 + returns).cumprod()
    peak = growth.cummax()
    dd = growth / peak - 1.0
    return dd


def max_drawdown(returns: pd.Series):
    """Calculate maximum drawdown from returns series"""
    dd = drawdown_curve(returns)
    return float(dd.min())


def max_monthly_loss(returns: pd.Series):
    """Calculate maximum one-month loss"""
    return float(returns.min())


def annualized_std(monthly_returns: pd.Series):
    """
    Calculate annualized standard deviation.

    Args:
        monthly_returns: Series of monthly returns

    Returns:
        Annualized standard deviation (monthly_std * sqrt(12))
    """
    sd = monthly_returns.std(ddof=0)
    return float(sd * np.sqrt(12))


def tracking_error(port_excess: pd.Series, mkt_excess: pd.Series):
    """
    Calculate annualized tracking error (std of active returns).

    Args:
        port_excess: Series of portfolio excess returns (monthly)
        mkt_excess: Series of market excess returns (monthly)

    Returns:
        Annualized tracking error (std of active returns * sqrt(12))
    """
    active = (port_excess - mkt_excess).dropna()
    sd = active.std(ddof=0)
    return float(sd * np.sqrt(12))


def returns_by_year(returns: pd.Series):
    """
    Calculate annual returns from monthly returns series.
    Returns a Series indexed by year with annual returns.
    """
    if returns.empty:
        return pd.Series(dtype=float)

    df = pd.DataFrame({'ret': returns})
    df['year'] = df.index.year

    # Compound returns within each year
    annual_rets = df.groupby('year')['ret'].apply(lambda x: (1 + x).prod() - 1)
    return annual_rets


def save_sector_allocation_plot(output_dir: str, sector_allocations: pd.DataFrame):
    """
    Plot sector allocation over time.
    sector_allocations should have columns: date, sector, gross_exposure
    """
    os.makedirs(output_dir, exist_ok=True)

    if sector_allocations.empty:
        return

    # Pivot to get time series by sector
    pivot = sector_allocations.pivot(index='date', columns='sector', values='gross_exposure')
    pivot = pivot.fillna(0.0)

    # Create stacked area chart
    plt.figure(figsize=(12, 6))
    pivot.plot.area(stacked=True, alpha=0.7, ax=plt.gca())
    plt.title("Sector Allocation Over Time (Gross Exposure)")
    plt.ylabel("Gross Exposure")
    plt.xlabel("Date")
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sector_allocation.png"))
    plt.close()


def save_cap_composition_plot(output_dir: str, cap_composition: pd.DataFrame):
    """
    Plot portfolio composition by market cap over time.
    cap_composition should have columns: date, large_cap_gross, small_cap_gross
    """
    os.makedirs(output_dir, exist_ok=True)

    if cap_composition.empty:
        return

    # Create stacked area chart
    plt.figure(figsize=(12, 6))
    cap_composition.set_index('date')[['large_cap_gross', 'small_cap_gross']].plot(
        kind='area', stacked=True, alpha=0.7, ax=plt.gca()
    )
    plt.title("Portfolio Composition by Market Cap")
    plt.ylabel("Gross Exposure")
    plt.xlabel("Date")
    plt.legend(['Large Cap (≥$15B)', 'Small Cap (<$15B)'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cap_composition.png"))
    plt.close()


def save_plots(output_dir: str, port_excess: pd.Series, rf: pd.Series, mkt_excess: pd.Series,
               msci_world_ret: pd.Series = None):
    os.makedirs(output_dir, exist_ok=True)

    # Build total return series (portfolio and S&P 500)
    port_total = (port_excess + rf).dropna()
    mkt_total = (mkt_excess + rf).dropna()  # equals 'ret' from market_data

    # Align on common realized months
    series_to_align = [port_total.rename("Portfolio (Total)"), mkt_total.rename("S&P 500 (Total)")]

    # Add MSCI World if provided
    if msci_world_ret is not None:
        msci_world_total = msci_world_ret.dropna().rename("MSCI World (Total)")
        series_to_align.append(msci_world_total)

    aligned = pd.concat(series_to_align, axis=1, join="inner").dropna()

    # Cumulative curves
    cum = (1 + aligned).cumprod()
    plt.figure(figsize=(10, 5))
    cum["Portfolio (Total)"].plot(label="Portfolio (Total)", linewidth=2)
    cum["S&P 500 (Total)"].plot(label="S&P 500 (Total)", linewidth=2, alpha=0.8)
    if "MSCI World (Total)" in cum.columns:
        cum["MSCI World (Total)"].plot(label="MSCI World (Total)", linewidth=2, alpha=0.8)
    plt.title("Cumulative Total Return (Portfolio vs Benchmarks)")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cum_returns.png"))
    plt.close()

    # Drawdown (portfolio only)
    dd = drawdown_curve(port_total)
    plt.figure(figsize=(10, 4))
    dd.plot(color="firebrick")
    plt.title("Drawdown (Portfolio Total Return)")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drawdown.png"))
    plt.close()

    # Rolling 12m Sharpe (excess)
    roll = port_excess.rolling(12)
    sh = np.sqrt(12) * roll.mean() / (roll.std(ddof=0) + 1e-12)
    plt.figure(figsize=(10, 4))
    sh.plot()
    plt.title("Rolling 12m Sharpe (Excess)")
    plt.ylabel("Sharpe")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling_sharpe_12m.png"))
    plt.close()


def write_summary(output_dir: str, metrics: Dict[str, float], market_capture_df: pd.DataFrame = None):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    lines = [f"{k}: {v:.6f}" for k, v in metrics.items()]

    # Add market capture ratios if provided
    if market_capture_df is not None and not market_capture_df.empty:
        lines.append("")
        lines.append("=" * 60)
        lines.append("Market Capture Ratios:")
        lines.append("=" * 60)

        # Format the market capture DataFrame as text
        for idx in market_capture_df.index:
            lines.append(f"\n{idx}:")
            for col in market_capture_df.columns:
                value = market_capture_df.loc[idx, col]
                lines.append(f"  {col}: {value:.2f}")

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))


def write_yearly_returns(output_dir: str, yearly_returns: pd.Series):
    """Write yearly returns to a CSV and text file"""
    os.makedirs(output_dir, exist_ok=True)

    if yearly_returns.empty:
        return

    # Save to CSV
    df = pd.DataFrame({
        'year': yearly_returns.index,
        'annual_return': yearly_returns.values
    })
    df.to_csv(os.path.join(output_dir, "yearly_returns.csv"), index=False)

    # Save to text file with formatted output
    lines = ["Annual Returns by Year:", "=" * 40]
    for year, ret in yearly_returns.items():
        lines.append(f"{year}: {ret:>10.2%}")
    lines.append("=" * 40)
    lines.append(f"Average: {yearly_returns.mean():>10.2%}")

    with open(os.path.join(output_dir, "yearly_returns.txt"), "w") as f:
        f.write("\n".join(lines))


def rolling_3year_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate rolling 3-year returns from monthly returns series.
    Returns a Series indexed by month with 3-year annualized returns.
    """
    if returns.empty or len(returns) < 36:
        return pd.Series(dtype=float)

    # Calculate rolling 3-year total returns (compounded) using vectorized operation
    rolling_total = returns.rolling(window=36).apply(lambda x: np.prod(1 + x) - 1, raw=True)

    # Annualize: (1 + total_return) ^ (1/3) - 1
    rolling_annualized = (1 + rolling_total) ** (1 / 3) - 1

    return rolling_annualized


def save_rolling_3year_plot(output_dir: str, port_total: pd.Series):
    """
    Plot rolling 3-year annualized returns.
    """
    os.makedirs(output_dir, exist_ok=True)

    rolling_rets = rolling_3year_returns(port_total)

    if rolling_rets.empty:
        return

    plt.figure(figsize=(12, 6))
    rolling_rets.plot(linewidth=2)
    plt.title("Rolling 3-Year Annualized Returns")
    plt.ylabel("Annualized Return")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling_3year_returns.png"))
    plt.close()


def calculate_aum_evolution(returns: pd.Series, initial_aum: float = None) -> pd.Series:
    """
    Calculate AUM evolution starting from initial_aum using monthly returns.

    Args:
        returns: Series of monthly returns
        initial_aum: Starting AUM in dollars (default from config: $500M)

    Returns:
        Series of AUM values indexed by date
    """
    if returns.empty:
        return pd.Series(dtype=float)

    if initial_aum is None:
        initial_aum = INITIAL_AUM_USD

    # Calculate cumulative growth
    growth = (1 + returns).cumprod()
    aum = initial_aum * growth

    return aum


def save_aum_plot(output_dir: str, aum: pd.Series, sp500_aum: pd.Series = None, msci_world_aum: pd.Series = None):
    """
    Plot AUM evolution over time for portfolio and benchmarks.

    Args:
        output_dir: Directory to save plot
        aum: Portfolio AUM series
        sp500_aum: S&P 500 benchmark AUM series (optional)
        msci_world_aum: MSCI World benchmark AUM series (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    if aum.empty:
        return

    plt.figure(figsize=(12, 6))
    aum.plot(linewidth=2, label="Portfolio")

    if sp500_aum is not None and not sp500_aum.empty:
        sp500_aum.plot(linewidth=2, alpha=0.8, label="S&P 500")

    if msci_world_aum is not None and not msci_world_aum.empty:
        msci_world_aum.plot(linewidth=2, alpha=0.8, label="MSCI World")

    plt.title("AUM Evolution")
    plt.ylabel("AUM (USD)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Format y-axis as millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x / 1e6:.0f}M'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aum_evolution.png"))
    plt.close()


def write_aum_report(output_dir: str, aum: pd.Series, initial_aum: float = None):
    """
    Write AUM evolution to CSV and summary statistics.

    Args:
        output_dir: Directory to save outputs
        aum: Series of AUM values over time
        initial_aum: Starting AUM in dollars (default from config: $500M)
    """
    os.makedirs(output_dir, exist_ok=True)

    if aum.empty:
        return

    if initial_aum is None:
        initial_aum = INITIAL_AUM_USD

    # Save full AUM series to CSV
    df = pd.DataFrame({
        'date': aum.index,
        'aum_usd': aum.values
    })
    df.to_csv(os.path.join(output_dir, "aum_evolution.csv"), index=False)

    # Calculate monthly P&L
    monthly_pnl = aum.diff()
    monthly_pnl.iloc[0] = aum.iloc[0] - initial_aum  # First month P&L vs initial

    df_pnl = pd.DataFrame({
        'date': monthly_pnl.index,
        'monthly_pnl_usd': monthly_pnl.values
    })
    df_pnl.to_csv(os.path.join(output_dir, "monthly_pnl.csv"), index=False)

    # Write summary
    lines = [
        "AUM Summary:",
        "=" * 60,
        f"Initial AUM:        ${initial_aum:>15,.0f}",
        f"Final AUM:          ${aum.iloc[-1]:>15,.0f}",
        f"Total P&L:          ${aum.iloc[-1] - initial_aum:>15,.0f}",
        f"Total Return:       {(aum.iloc[-1] / initial_aum - 1):>15.2%}",
        "=" * 60,
        f"Average Monthly P&L: ${monthly_pnl.mean():>15,.0f}",
        f"Max Monthly P&L:     ${monthly_pnl.max():>15,.0f}",
        f"Min Monthly P&L:     ${monthly_pnl.min():>15,.0f}",
    ]

    with open(os.path.join(output_dir, "aum_summary.txt"), "w") as f:
        f.write("\n".join(lines))


def three_year_beat_benchmark(portfolio_returns: pd.Series, market_returns: pd.Series):
    """
    Calculate the percentage of times the portfolio beats the benchmark on a 3-year rolling basis.

    Uses a wealth index approach:
    1. Create wealth index starting at 100
    2. Calculate 3-year rolling returns using pctchange(36)
    3. Compare portfolio vs market
    4. Calculate percentage of times portfolio > market

    Args:
        portfolio_returns: Series of monthly portfolio returns (net)
        market_returns: Series of monthly market returns

    Returns:
        Float representing the percentage (0-100) of times portfolio beats market on 3-year rolling basis
    """
    # Create wealth indices starting at 100
    portfolio_wealth = 100 * (1 + portfolio_returns).cumprod()
    market_wealth = 100 * (1 + market_returns).cumprod()

    # Calculate 3-year rolling returns using pctchange(36)
    portfolio_3yr = portfolio_wealth.pct_change(36)
    market_3yr = market_wealth.pct_change(36)

    # Drop NaN values (first 36 months)
    valid_data = pd.DataFrame({
        'portfolio': portfolio_3yr,
        'market': market_3yr
    }).dropna()

    if len(valid_data) == 0:
        return 0.0

    # Calculate percentage of times portfolio > market
    beat_count = (valid_data['portfolio'] > valid_data['market']).sum()
    total_count = len(valid_data)

    beat_percentage = (beat_count / total_count * 100) if total_count > 0 else 0.0

    return float(beat_percentage)


def market_capture_ratio(returns: pd.DataFrame):
    """
    Calculate the up market capture ratio and down market capture ratio based on monthly returns.

    Market capture ratio measures how well a portfolio performed relative to a benchmark during
    up markets and down markets. A ratio > 100% means the portfolio outperformed during those periods.

    Args:
        returns: DataFrame with paired columns (portfolio, benchmark), indexed by date.
                 Must have an even number of columns where columns are paired as:
                 (portfolio1, benchmark1, portfolio2, benchmark2, ...)
                 Data should be monthly returns.

    Returns:
        DataFrame with up and down market capture ratios for each portfolio-benchmark pair
    """
    if len(returns.columns) % 2 != 0:
        raise ValueError(
            f"DataFrame must have an even number of columns (portfolio-benchmark pairs). Got {len(returns.columns)} columns.")

    df_final = pd.DataFrame()

    for i in range(0, len(returns.columns), 2):
        # create a temp df with the long etf and its index
        df_temp = returns.iloc[:, i:i + 2]  # STOPPED HERE. NEED TO ISOLATE THE COLUMNS
        # get the names of the long_etf and the index
        # Skip if we don't have both columns

        # Get the names of the portfolio and the benchmark
        portfolio = df_temp.columns[0]
        index = df_temp.columns[1]

        # Calculate up market capture (when benchmark >= 0)
        # Use cumulative returns for proper calculation
        up_market = (df_temp[df_temp[index]
                             >= 0]).sum(axis=0)

        up_ratio = (up_market / up_market.iloc[-1] * 100) \
            .round(2)

        up_ratio = up_ratio.to_frame(name='Up Market Capture Ratio')

        # 3) down ratio

        down_market = (df_temp[df_temp[index]
                               < 0]).sum(axis=0)

        down_ratio = (down_market / down_market.iloc[-1] * 100) \
            .round(2)

        down_ratio = down_ratio.to_frame(name='Down Market Capture Ratio')

        # create a final df
        df_ratio = pd.concat([up_ratio, down_ratio], axis=1)
        df_ratio.index = [portfolio, index]

        df_final = pd.concat([df_final, df_ratio])

    return df_final