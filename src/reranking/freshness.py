from __future__ import annotations
import pandas as pd
import numpy as np


def compute_freshness_bonus(age_hours: pd.Series, max_bonus: float = 0.15) -> pd.Series:
    """
    Newer items get a higher bonus.
    Older items get a smaller bonus.
    """
    age = age_hours.fillna(age_hours.median()).clip(lower=0)

    # Exponential decay: very fresh items get higher bonus
    bonus = max_bonus * np.exp(-age / 24.0)
    return pd.Series(bonus, index=age_hours.index)