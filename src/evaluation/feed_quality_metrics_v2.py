from pathlib import Path
import pandas as pd
import numpy as np


def category_diversity_at_k(df: pd.DataFrame, k: int = 5, score_col: str = "target_policy_score") -> float:
    topk = df.sort_values(score_col, ascending=False).head(k)
    if len(topk) == 0:
        return 0.0
    return topk["category"].nunique() / len(topk)


def source_diversity_at_k(df: pd.DataFrame, k: int = 5, score_col: str = "target_policy_score") -> float:
    topk = df.sort_values(score_col, ascending=False).head(k)
    if len(topk) == 0:
        return 0.0
    return topk["source"].nunique() / len(topk)


def freshness_at_k(df: pd.DataFrame, k: int = 5, score_col: str = "target_policy_score") -> float:
    topk = df.sort_values(score_col, ascending=False).head(k)
    if len(topk) == 0:
        return 0.0

    # support both schemas if needed
    if "age_hours" in topk.columns:
        age = pd.to_numeric(topk["age_hours"], errors="coerce").fillna(0.0)
    elif "item_age_hours" in topk.columns:
        age = pd.to_numeric(topk["item_age_hours"], errors="coerce").fillna(0.0)
    else:
        age = pd.Series(np.zeros(len(topk)))

    freshness = 1.0 / (1.0 + age)
    return float(freshness.mean())


def evaluate_feed_quality(df: pd.DataFrame, k: int = 5, score_col: str = "target_policy_score") -> dict:
    rows = []

    for _, session_df in df.groupby("session_id"):
        rows.append(
            {
                "category_diversity@k": category_diversity_at_k(session_df, k, score_col),
                "source_diversity@k": source_diversity_at_k(session_df, k, score_col),
                "freshness@k": freshness_at_k(session_df, k, score_col),
            }
        )

    out = pd.DataFrame(rows)
    return out.mean().to_dict()


def main():
    data_path = Path("data/gold/simulated_logged_policy_v2.parquet")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}")

    df = pd.read_parquet(data_path)

    print("=== Logged Policy Feed Quality Metrics ===")
    logged_metrics = evaluate_feed_quality(df, k=5, score_col="logged_policy_score")
    print(f"Category Diversity@5: {logged_metrics['category_diversity@k']:.4f}")
    print(f"Source Diversity@5:   {logged_metrics['source_diversity@k']:.4f}")
    print(f"Freshness@5:          {logged_metrics['freshness@k']:.4f}")

    print("\n=== Target Policy Feed Quality Metrics ===")
    target_metrics = evaluate_feed_quality(df, k=5, score_col="target_policy_score")
    print(f"Category Diversity@5: {target_metrics['category_diversity@k']:.4f}")
    print(f"Source Diversity@5:   {target_metrics['source_diversity@k']:.4f}")
    print(f"Freshness@5:          {target_metrics['freshness@k']:.4f}")

    print("\n=== Improvement (Target - Logged) ===")
    print(f"Category Diversity@5 Δ: {target_metrics['category_diversity@k'] - logged_metrics['category_diversity@k']:.4f}")
    print(f"Source Diversity@5 Δ:   {target_metrics['source_diversity@k'] - logged_metrics['source_diversity@k']:.4f}")
    print(f"Freshness@5 Δ:          {target_metrics['freshness@k'] - logged_metrics['freshness@k']:.4f}")


if __name__ == "__main__":
    main()