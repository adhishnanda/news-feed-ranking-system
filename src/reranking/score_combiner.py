from __future__ import annotations
import pandas as pd

from src.reranking.freshness import compute_freshness_bonus
from src.reranking.diversity import greedy_diversity_rerank


def apply_reranking(
    scored_df: pd.DataFrame,
    top_k: int = 10,
    freshness_weight: float = 1.0,
) -> pd.DataFrame:
    df = scored_df.copy()

    df["freshness_bonus"] = compute_freshness_bonus(df["age_hours"])
    df["preliminary_score"] = df["model_score"] + freshness_weight * df["freshness_bonus"]

    reranked = greedy_diversity_rerank(df, top_k=top_k)

    return reranked