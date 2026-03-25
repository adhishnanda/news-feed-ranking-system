import pandas as pd
import numpy as np


def category_diversity_at_k(df: pd.DataFrame, k: int = 5) -> float:
    topk = df.sort_values("score", ascending=False).head(k)
    if len(topk) == 0:
        return 0.0
    return topk["category"].nunique() / len(topk)


def source_diversity_at_k(df: pd.DataFrame, k: int = 5) -> float:
    topk = df.sort_values("score", ascending=False).head(k)
    if len(topk) == 0:
        return 0.0
    return topk["source"].nunique() / len(topk)


def freshness_at_k(df: pd.DataFrame, k: int = 5) -> float:
    topk = df.sort_values("score", ascending=False).head(k)
    if len(topk) == 0:
        return 0.0

    age = pd.to_numeric(topk["age_hours"], errors="coerce").fillna(0.0)
    freshness = 1.0 / (1.0 + age)
    return float(freshness.mean())


def evaluate_feed_quality(df: pd.DataFrame, k: int = 5) -> dict:
    rows = []

    for _, session_df in df.groupby("session_id"):
        rows.append(
            {
                "category_diversity@k": category_diversity_at_k(session_df, k),
                "source_diversity@k": source_diversity_at_k(session_df, k),
                "freshness@k": freshness_at_k(session_df, k),
            }
        )

    out = pd.DataFrame(rows)
    return out.mean().to_dict()