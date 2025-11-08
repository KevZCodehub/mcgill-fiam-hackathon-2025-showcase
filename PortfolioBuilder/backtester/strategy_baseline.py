from typing import Dict, Tuple, Set
import numpy as np
import pandas as pd
from backtester.config import Constraints, StrategyParams

class RegimeSectorTiltModel:
    """
    Expanding-window estimator of sector tilts per regime:
    For each (state, sector), keep running mean of stock excess returns up to t-1.
    Convert to z-scores per regime at time t and scale by gamma to adjust scores.
    """
    def __init__(self, gamma: float):
        self.gamma = gamma
        # keys: (state, sector) -> [sum, count]
        self._sum: Dict[Tuple[str, str], float] = {}
        self._cnt: Dict[Tuple[str, str], int] = {}

    def update_history(self, past_df: pd.DataFrame):
        # past_df columns: ['state','sector','stock_ret']
        for (reg, sec), g in past_df.groupby(["state", "sector"]):
            s = float(g["stock_ret"].sum())
            c = int(g["stock_ret"].count())
            key = (reg, sec)
            self._sum[key] = self._sum.get(key, 0.0) + s
            self._cnt[key] = self._cnt.get(key, 0) + c

    def get_sector_multipliers(self, current_regime: str) -> Dict[str, float]:
        """
        For current regime, produce sector multipliers as 1 + gamma * z_score(mean_return_by_sector).
        If insufficient data, return neutral multipliers (1.0).
        """
        sectors = []
        means = []
        for (reg, sec), c in self._cnt.items():
            if reg != current_regime:
                continue
            mu = self._sum[(reg, sec)] / max(1, c)
            sectors.append(sec)
            means.append(mu)
        if not sectors:
            return {}

        means = np.array(means)
        m = means.mean()
        s = means.std(ddof=0)
        if s <= 0:
            z = np.zeros_like(means)
        else:
            z = (means - m) / s

        mult = 1.0 + self.gamma * z
        mult = np.maximum(0.5, mult)  # floor to avoid flipping signs too aggressively

        return {sec: float(mv) for sec, mv in zip(sectors, mult)}

class BaselineSectorTiltStrategy:
    def __init__(self, constraints: Constraints, params: StrategyParams):
        self.constraints = constraints
        self.params = params
        self.sector_tilt_model = RegimeSectorTiltModel(gamma=params.SECTOR_TILT_GAMMA)

    def prepare_scores(self, month_df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Adjust LightGBM scores by regime-conditioned sector multipliers and z-score cross-sectionally.
        Returns df with columns: id, sector, adj_score, decile (and excntry if available)
        """
        cols_to_copy = ["id", "sector", "lightgbm"]
        if "excntry" in month_df.columns:
            cols_to_copy.append("excntry")
        df = month_df[cols_to_copy].copy()

        # Cross-sectional z-score (within this month's cross-section)
        mu = df["lightgbm"].mean()
        sd = df["lightgbm"].std(ddof=0)
        if sd <= 0 or np.isclose(sd, 0.0):
            df["score_z"] = 0.0
        else:
            df["score_z"] = (df["lightgbm"] - mu) / (sd + 1e-12)

        # Apply sector multipliers based on regime
        mult = self.sector_tilt_model.get_sector_multipliers(regime)
        if mult:
            df["sec_mult"] = df["sector"].map(mult).fillna(1.0)
        else:
            df["sec_mult"] = 1.0

        df["adj_score"] = df["score_z"] * df["sec_mult"]

        # Robust deciles using percentile ranks (0..1], then map to 0..9
        ranks_pct = df["adj_score"].rank(method="first", pct=True)
        df["decile"] = np.minimum((ranks_pct * 10).astype(int), 9)

        cols_to_return = ["id", "sector", "adj_score", "decile"]
        if "excntry" in df.columns:
            cols_to_return.append("excntry")
        return df[cols_to_return]

    def update_history(self, past_df: pd.DataFrame):
        # Past realized returns are point-in-time safe
        self.sector_tilt_model.update_history(past_df)