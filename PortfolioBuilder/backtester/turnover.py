import pandas as pd
import numpy as np
import warnings


def weight_based_turnover(weights_history_by_reteom: dict) -> float:
    """
    Compute average monthly turnover using the formula:

    Turnoverₜ = (1/2) × Σ |wᵢ,ₜ − wᵢ,ₜ₋₁|

    where wᵢ,ₜ is the weight of stock i at time t.

    Args:
        weights_history_by_reteom: dict[ret_eom -> pd.Series(weights indexed by id)]

    Returns:
        Average turnover across all consecutive periods
    """
    if not weights_history_by_reteom or len(weights_history_by_reteom) < 2:
        return 0.0

    dates_sorted = sorted(weights_history_by_reteom.keys())
    turnovers = []

    for i in range(len(dates_sorted) - 1):
        d_t = dates_sorted[i]
        d_tp1 = dates_sorted[i + 1]

        w_t = weights_history_by_reteom[d_t]
        w_tp1 = weights_history_by_reteom[d_tp1]

        # Get union of all ids across both periods
        all_ids = set(w_t.index) | set(w_tp1.index)

        # Align weights, filling missing with 0
        w_t_aligned = w_t.reindex(all_ids).fillna(0.0)
        w_tp1_aligned = w_tp1.reindex(all_ids).fillna(0.0)

        # Calculate turnover for this period
        turnover_t = 0.5 * (w_tp1_aligned - w_t_aligned).abs().sum()
        turnovers.append(turnover_t)

    if not turnovers:
        return 0.0

    return float(np.mean(turnovers))


def combined_turnover_mean(positions_df: pd.DataFrame, date_col: str = "date", id_col: str = "id") -> float:
    """
    Combined turnover as replacement rate on realized-month holdings:

      turnover_t = 1 - |S_ret,t ∩ S_ret,t+1| / |S_ret,t|

    where S_ret,t is the set of held ids keyed by ret_eom for month t (period end).
    Average across consecutive realized months.

    positions_df must have one row per active position per realized month with columns [id_col, date_col].

    .. deprecated::
        This is the old turnover calculation. Use weight_based_turnover instead.
    """
    warnings.warn(
        "combined_turnover_mean is deprecated and will be removed in a future version. "
        "Use weight_based_turnover instead for the new weight-based turnover calculation.",
        DeprecationWarning,
        stacklevel=2
    )

    if positions_df.empty:
        return 0.0

    df = positions_df[[id_col, date_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([date_col, id_col])

    sets_by_date = {d: set(g[id_col].tolist()) for d, g in df.groupby(date_col)}
    dates_sorted = sorted(sets_by_date.keys())
    if len(dates_sorted) < 2:
        return 0.0

    turnovers = []
    for i in range(len(dates_sorted) - 1):
        d_t = dates_sorted[i]
        d_tp1 = dates_sorted[i + 1]
        s_t = sets_by_date[d_t]
        s_tp1 = sets_by_date[d_tp1]
        if len(s_t) == 0:
            continue
        remain = len(s_t.intersection(s_tp1))
        turnovers.append(1.0 - remain / len(s_t))

    if not turnovers:
        return 0.0
    return float(sum(turnovers) / len(turnovers))


def build_positions_table(weights_history_by_reteom):
    """
    weights_history_by_reteom: dict[ret_eom -> pd.Series(weights indexed by id)]
    Returns df with columns ['id','date'] where 'date' is ret_eom for each active position.
    """
    rows = []
    for d, w in weights_history_by_reteom.items():
        nz = w[w.abs() > 1e-9]
        if nz.empty:
            continue
        rows.extend([(i, d) for i in nz.index])
    return pd.DataFrame(rows, columns=["id", "ret_eom"])