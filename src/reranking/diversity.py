from __future__ import annotations
from collections import Counter
import pandas as pd


def greedy_diversity_rerank(
    df: pd.DataFrame,
    top_k: int = 10,
    category_penalty_weight: float = 0.08,
    source_penalty_weight: float = 0.10,
) -> pd.DataFrame:
    """
    Greedy reranking:
    - start from highest preliminary score
    - penalize repeated categories
    - penalize repeated sources
    """

    working = df.copy().reset_index(drop=True)
    selected_rows = []

    category_counts = Counter()
    source_counts = Counter()

    remaining = working.copy()

    for _ in range(min(top_k, len(remaining))):
        remaining = remaining.copy()

        remaining["category_repeat_penalty"] = remaining["category"].map(
            lambda x: category_counts.get(x, 0) * category_penalty_weight
        )

        remaining["source_repeat_penalty"] = remaining["source"].map(
            lambda x: source_counts.get(x, 0) * source_penalty_weight
        )

        remaining["final_score"] = (
            remaining["preliminary_score"]
            - remaining["category_repeat_penalty"]
            - remaining["source_repeat_penalty"]
        )

        best_idx = remaining["final_score"].idxmax()
        best_row = remaining.loc[best_idx].copy()

        selected_rows.append(best_row)

        category_counts[best_row["category"]] += 1
        source_counts[best_row["source"]] += 1

        remaining = remaining.drop(index=best_idx)

    if not selected_rows:
        return pd.DataFrame()

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result["final_rank"] = range(1, len(result) + 1)
    return result