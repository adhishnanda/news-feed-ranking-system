import numpy as np
import pandas as pd


def precision_at_k(df, k=5):
    df = df.sort_values("score", ascending=False).head(k)
    return df["clicked"].sum() / k


def recall_at_k(df, k=5):
    total_relevant = df["clicked"].sum()
    if total_relevant == 0:
        return 0.0

    df = df.sort_values("score", ascending=False).head(k)
    return df["clicked"].sum() / total_relevant


def dcg_at_k(df, k=5):
    df = df.sort_values("score", ascending=False).head(k)
    gains = df["clicked"].values

    return np.sum([
        gains[i] / np.log2(i + 2) for i in range(len(gains))
    ])


def ndcg_at_k(df, k=5):
    ideal_df = df.sort_values("clicked", ascending=False)

    ideal_dcg = dcg_at_k(ideal_df, k)
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(df, k) / ideal_dcg


def evaluate_ranking(df, k=5):
    """
    Evaluate per session and average
    """
    results = []

    for _, session_df in df.groupby("session_id"):
        results.append({
            "precision@k": precision_at_k(session_df, k),
            "recall@k": recall_at_k(session_df, k),
            "ndcg@k": ndcg_at_k(session_df, k),
        })

    return pd.DataFrame(results).mean().to_dict()