from typing import Dict, Tuple, Set
import numpy as np
import pandas as pd
from backtester.config import Constraints, StrategyParams


def select_long_short_ids(df: pd.DataFrame,
                          prev_ids_long: Set[str],
                          prev_ids_short: Set[str],
                          params: StrategyParams,
                          constraints: Constraints,
                          target_holdings: int) -> Tuple[Set[str], Set[str]]:
    top_mask = df["decile"] >= params.TOP_DECILE
    bot_mask = df["decile"] <= params.BOTTOM_DECILE

    hysteresis_long_band, hysteresis_short_band = params.TURNOVER_HYSTERESIS_DECILES
    keep_long_ids = set(df.loc[df["decile"] >= hysteresis_long_band, "id"]) & prev_ids_long
    keep_short_ids = set(df.loc[df["decile"] <= hysteresis_short_band, "id"]) & prev_ids_short

    long_ids = set(keep_long_ids)
    short_ids = set(keep_short_ids)

    total_target = int(np.clip(target_holdings, constraints.MIN_HOLDINGS, constraints.MAX_HOLDINGS))
    long_target_count = total_target // 2 + total_target // 10
    short_target_count = total_target - long_target_count

    for i in df.loc[top_mask].sort_values("adj_score", ascending=False)["id"]:
        if len(long_ids) >= long_target_count:
            break
        long_ids.add(i)

    for i in df.loc[bot_mask].sort_values("adj_score", ascending=True)["id"]:
        if len(short_ids) >= short_target_count:
            break
        short_ids.add(i)

    def expand_side(side: str, need: int):
        if side == "long":
            for d in [params.TOP_DECILE - 1, params.TOP_DECILE - 2]:
                if d < 0:
                    break
                for i in df.loc[df["decile"] == d].sort_values("adj_score", ascending=False)["id"]:
                    if len(long_ids) >= need: return
                    if i not in long_ids and i not in short_ids:
                        long_ids.add(i)
        else:
            for d in [params.BOTTOM_DECILE + 1, params.BOTTOM_DECILE + 2]:
                if d > 9:
                    break
                for i in df.loc[df["decile"] == d].sort_values("adj_score", ascending=True)["id"]:
                    if len(short_ids) >= need: return
                    if i not in short_ids and i not in long_ids:
                        short_ids.add(i)

    total_now = len(long_ids) + len(short_ids)
    if total_now < constraints.MIN_HOLDINGS:
        need = constraints.MIN_HOLDINGS - total_now
        expand_side("long", len(long_ids) + need // 2 + need % 2)
        expand_side("short", len(short_ids) + need // 2)

    if len(long_ids) + len(short_ids) > constraints.MAX_HOLDINGS:
        df_idx = df.set_index("id")
        long_sorted = sorted(list(long_ids), key=lambda i: df_idx.at[i, "adj_score"], reverse=True)
        short_sorted = sorted(list(short_ids), key=lambda i: df_idx.at[i, "adj_score"])
        cap_long = min(len(long_sorted), constraints.MAX_HOLDINGS // 2 + constraints.MAX_HOLDINGS // 10)
        cap_short = constraints.MAX_HOLDINGS - cap_long
        long_ids = set(long_sorted[:cap_long])
        short_ids = set(short_sorted[:cap_short])

    return long_ids, short_ids


def _scale_to_caps(w: pd.Series, cons: Constraints) -> pd.Series:
    # Per-name cap
    w = w.clip(lower=-cons.NAME_CAP_ABS, upper=cons.NAME_CAP_ABS)

    # Long gross cap
    long_gross = w[w > 0].sum()
    if long_gross > cons.LONG_GROSS_MAX and long_gross > 0:
        w[w > 0] *= cons.LONG_GROSS_MAX / long_gross

    # Short gross cap
    short_gross = -w[w < 0].sum()
    if short_gross > cons.SHORT_GROSS_MAX and short_gross > 0:
        w[w < 0] *= cons.SHORT_GROSS_MAX / short_gross

    # Total gross cap
    total_gross = w.abs().sum()
    if total_gross > cons.TOTAL_GROSS_MAX and total_gross > 0:
        w *= cons.TOTAL_GROSS_MAX / total_gross

    # Re-clip
    w = w.clip(lower=-cons.NAME_CAP_ABS, upper=cons.NAME_CAP_ABS)

    # Recheck long, short, total
    long_gross = w[w > 0].sum()
    if long_gross > cons.LONG_GROSS_MAX and long_gross > 0:
        w[w > 0] *= cons.LONG_GROSS_MAX / long_gross

    short_gross = -w[w < 0].sum()
    if short_gross > cons.SHORT_GROSS_MAX and short_gross > 0:
        w[w < 0] *= cons.SHORT_GROSS_MAX / short_gross

    total_gross = w.abs().sum()
    if total_gross > cons.TOTAL_GROSS_MAX and total_gross > 0:
        w *= cons.TOTAL_GROSS_MAX / total_gross

    return w


def _apply_sector_gross_caps(w: pd.Series,
                             sectors_by_id: pd.Series,
                             sector_caps: Dict[str, float]) -> pd.Series:
    """
    Enforce per-sector gross caps: sum(|w_i| for i in sector s) <= sector_caps[s]
    sector_caps keys are strings matching sectors_by_id (also strings).
    """
    if not sector_caps:
        return w

    # Build a DataFrame for convenience
    dfw = pd.DataFrame({"w": w})
    dfw["sector"] = sectors_by_id.reindex(w.index).astype(str)

    for sec, cap in sector_caps.items():
        mask = dfw["sector"] == str(sec)
        if not mask.any():
            continue
        gross_sec = dfw.loc[mask, "w"].abs().sum()
        if gross_sec > cap and gross_sec > 0:
            scale = cap / gross_sec
            dfw.loc[mask, "w"] *= scale

    return dfw["w"]


def _apply_country_gross_cap(w: pd.Series,
                             countries_by_id: pd.Series,
                             country_cap: float) -> pd.Series:
    """
    Enforce per-country gross cap: sum(|w_i| for i in country c) <= country_cap
    This ensures no single country has more than country_cap total exposure.
    """
    if country_cap <= 0 or country_cap >= 1e10:
        return w

    # Build a DataFrame for convenience
    dfw = pd.DataFrame({"w": w})
    dfw["country"] = countries_by_id.reindex(w.index).astype(str)

    # Calculate gross exposure by country
    # Skip NaN/None countries and the string 'nan' which can result from astype(str) on NaN values
    for country in dfw["country"].unique():
        if pd.isna(country) or str(country).lower() in ('nan', 'none', ''):
            continue
        mask = dfw["country"] == country
        if not mask.any():
            continue
        gross_country = dfw.loc[mask, "w"].abs().sum()
        if gross_country > country_cap and gross_country > 0:
            scale = country_cap / gross_country
            dfw.loc[mask, "w"] *= scale

    return dfw["w"]


def allocate_weights(df: pd.DataFrame,
                     long_ids: Set[str],
                     short_ids: Set[str],
                     prev_w: pd.Series,
                     gross_targets: Tuple[float, float],
                     constraints: Constraints,
                     smoothing_to_prev: float,
                     min_stock_weight_abs: float,
                     sector_gross_caps: Dict[str, float],
                     country_gross_cap: float = None) -> pd.Series:
    """
    Allocate weights within candidate sets using score-proportional scheme,
    then smooth, enforce sector caps, country caps, min weight, and global caps.
    """
    ids = df["id"].tolist()
    scores = df.set_index("id")["adj_score"]
    sectors_by_id = df.set_index("id")["sector"].astype(str)

    # Extract country data if available
    countries_by_id = None
    if "excntry" in df.columns:
        countries_by_id = df.set_index("id")["excntry"].astype(str)

    # Use country cap from constraints if not provided
    if country_gross_cap is None:
        country_gross_cap = constraints.COUNTRY_GROSS_CAP

    w_new = pd.Series(0.0, index=ids, dtype=float)

    long_target, short_target = gross_targets

    # Long side
    if long_ids:
        s_long = scores.loc[list(long_ids)].clip(lower=0.0)
        w_long = pd.Series(1.0, index=s_long.index) if s_long.sum() <= 0 else s_long
        w_long = w_long / w_long.sum() * max(0.0, long_target)
        w_long = w_long.clip(upper=constraints.NAME_CAP_ABS)
        if w_long.sum() > 0:
            w_long = w_long / w_long.sum() * max(0.0, long_target)
        for i, v in w_long.items():
            w_new.at[i] = v

    # Short side
    if short_ids:
        s_short = (-scores.loc[list(short_ids)]).clip(lower=0.0)
        w_short = pd.Series(1.0, index=s_short.index) if s_short.sum() <= 0 else s_short
        w_short = w_short / w_short.sum() * max(0.0, short_target)
        w_short = w_short.clip(upper=constraints.NAME_CAP_ABS)
        if w_short.sum() > 0:
            w_short = w_short / w_short.sum() * max(0.0, short_target)
        for i, v in w_short.items():
            w_new.at[i] = -v

    # Smooth
    prev_w = prev_w.reindex(w_new.index).fillna(0.0)
    w_smooth = (1 - smoothing_to_prev) * w_new + smoothing_to_prev * prev_w

    # Holdings cap pre-check
    nonzero = w_smooth[w_smooth.abs() > 0]
    # If more than allowed, keep top-K by |w|
    # Note: min weight floor can increase holdings; we cap again after.
    if len(nonzero) > constraints.MAX_HOLDINGS:
        top_idx = nonzero.abs().sort_values(ascending=False).head(constraints.MAX_HOLDINGS).index
        w_smooth.loc[~w_smooth.index.isin(top_idx)] = 0.0

    # Enforce sector gross caps (first pass)
    w_sec = _apply_sector_gross_caps(w_smooth.copy(), sectors_by_id, sector_gross_caps)

    # Enforce country gross cap (first pass)
    if countries_by_id is not None:
        w_sec = _apply_country_gross_cap(w_sec, countries_by_id, country_gross_cap)

    # Enforce min weight floor on non-zeros
    small = (w_sec.abs() > 0) & (w_sec.abs() < min_stock_weight_abs)
    w_sec.loc[small] = w_sec.loc[small].apply(lambda x: np.sign(x) * min_stock_weight_abs)

    # Project to global caps
    w_proj = _scale_to_caps(w_sec.copy(), constraints)

    # Final consistency: remove tiny stragglers and re-apply sector, country, and global caps
    tiny = (w_proj.abs() > 0) & (w_proj.abs() < min_stock_weight_abs)
    if tiny.any():
        w_proj.loc[tiny] = 0.0
        # Ensure holdings ≤ MAX
        nz = w_proj[w_proj.abs() > 0]
        if len(nz) > constraints.MAX_HOLDINGS:
            keep = nz.abs().sort_values(ascending=False).head(constraints.MAX_HOLDINGS).index
            w_proj.loc[~w_proj.index.isin(keep)] = 0.0
        w_proj = _apply_sector_gross_caps(w_proj, sectors_by_id, sector_gross_caps)
        if countries_by_id is not None:
            w_proj = _apply_country_gross_cap(w_proj, countries_by_id, country_gross_cap)
        w_proj = _scale_to_caps(w_proj, constraints)

    return w_proj