from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path

# Portfolio configuration constants
INITIAL_AUM_USD = 500_000_000  # Starting AUM: $500 million

# Conversion constants
BPS_TO_DECIMAL = 10000.0  # Basis points to decimal conversion factor

REPO_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_ROOT = REPO_ROOT / "PortfolioBuilder"
MODEL_ROOT = REPO_ROOT / "Model"


@dataclass
class Paths:
    RESULTS_CSV: str = str((MODEL_ROOT / "src" / "results.csv").resolve())
    MARKET_CSV: str = str((PORTFOLIO_ROOT / "backtester" / "market_data.csv").resolve())
    MSCI_WORLD_CSV: str = str((PORTFOLIO_ROOT / "backtester" / "msci world.csv").resolve())
    REGIME_CSV: str = str((PORTFOLIO_ROOT / "regime.csv").resolve())
    OUTPUT_DIR: str = str((PORTFOLIO_ROOT / "artifacts").resolve())


@dataclass
class Constraints:
    # Exposure constraints
    TOTAL_GROSS_MAX: float = 2.00  # sum(|w_i|) <= 2.00
    LONG_GROSS_MAX: float = 1.50  # sum(w_i for w_i>0) <= 1.50
    SHORT_GROSS_MAX: float = 0.50  # sum(-w_i for w_i<0) <= 0.50
    NAME_CAP_ABS: float = 0.10  # |w_i| <= 0.10

    # Country concentration constraint
    COUNTRY_GROSS_CAP: float = 0.75  # sum(|w_i| for country) <= 0.75

    # Holdings constraints
    MIN_HOLDINGS: int = 100
    MAX_HOLDINGS: int = 300

    # Trading costs (one-way, in basis points)
    TRANSACTION_COST_BPS: float = 5.0  # Legacy flat cost (not used with cap-dependent costs)

    # Market cap thresholds and costs (in USD billions)
    LARGE_CAP_THRESHOLD: float = 15.0  # Market cap >= $15B is Large Cap
    LARGE_CAP_COST_BPS: float = 25.0  # 0.25% (25 bps) per turnover for Large Cap
    SMALL_CAP_COST_BPS: float = 100.0  # 1.00% (100 bps) per turnover for Small Cap


@dataclass
class StrategyParams:
    # Feature/selection
    TOP_DECILE: int = 9
    BOTTOM_DECILE: int = 0

    # Turnover control
    TURNOVER_HYSTERESIS_DECILES: Tuple[int, int] = (8, 1)
    SMOOTHING_TO_PREV: float = 0.15

    # Score tilts
    SECTOR_TILT_GAMMA: float = 0.5

    # Enable/disable regime overlays
    ENABLE_REGIME_GROSS_TARGETS: bool = True
    ENABLE_REGIME_SECTOR_GROSS_CAPS: bool = True

    # Default gross targets when regime gross targets are disabled
    DEFAULT_GROSS_TARGETS: Tuple[float, float] = (1.5, 0.25)  # total 1.50 max is enforced downstream

    # Desired gross targets by regime (long, short). More leverage in bullish, more short in bearish.
    REGIME_GROSS_TARGETS: Dict[str, Tuple[float, float]] = None

    # Target holdings count
    TARGET_HOLDINGS: int = 200

    # Minimum absolute weight for any active position
    MIN_STOCK_WEIGHT_ABS: float = 0.0025  # 0.5%

    # Regime-dependent per-sector gross exposure caps: sum(|w_i| for sector) <= cap
    # Keys are GICS sector codes as strings to match df["sector"].astype(str)
    REGIME_SECTOR_GROSS_CAPS: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        # Regime gross targets tuned to your intent:
        # - Expansion/Recovery: long-heavy (more leverage) and small short book
        # - Slowdown/Recession: more short usage (within 0.50 cap) and lower long gross
        if self.REGIME_GROSS_TARGETS is None:
            self.REGIME_GROSS_TARGETS = {
                "Expansion": (1.5, 0.00),  # total 1.50, net 1.20
                "Recovery": (1.25, 0.00),  # total 1.50, net 1.20
                "Slowdown": (1.0, 0.25),  # total 1.40, net 0.60
                "Recession": (0.80, 0.50),  # total 1.35, net 0.35
            }
        # Regime sector caps using your categories:
        # Very Bullish: 0.35, Bullish: 0.30, Bearish: 0.25, Very Bearish: 0.20
        if self.REGIME_SECTOR_GROSS_CAPS is None:
            self.REGIME_SECTOR_GROSS_CAPS = {
                "Expansion": {
                    "40": 0.35,  # Financials
                    "45": 0.35,  # Information Technology
                    "50": 0.30,  # Communication Services
                    "30": 0.25,  # Consumer Staples
                    "35": 0.20,  # Health Care
                    "55": 0.20,  # Utilities
                },
                "Slowdown": {
                    "30": 0.35,  # Consumer Staples
                    "35": 0.35,  # Health Care
                    "20": 0.30,  # Industrials
                    "15": 0.25,  # Materials
                    "25": 0.20,  # Consumer Discretionary
                    "60": 0.20,  # Real Estate
                },
                "Recession": {
                    "30": 0.35,  # Consumer Staples
                    "55": 0.35,  # Utilities
                    "35": 0.30,  # Health Care
                    "50": 0.25,  # Communication Services
                    "45": 0.20,  # Information Technology
                    "60": 0.20,  # Real Estate
                },
                "Recovery": {
                    "25": 0.35,  # Consumer Discretionary
                    "60": 0.35,  # Real Estate
                    "15": 0.30,  # Materials
                    "35": 0.25,  # Health Care
                    "30": 0.20,  # Consumer Staples
                    "55": 0.20,  # Utilities
                },
            }


REGIME_ORDER = ["Recovery", "Expansion", "Slowdown", "Recession"]