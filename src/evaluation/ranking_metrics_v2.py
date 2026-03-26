import numpy as np
import pandas as pd


def precision_at_k(df, k=5, score_col="target_policy_score"):
    df = df.sort_values(score_col, ascending=False).head(k)
    return df["clicked"].sum() / k


def recall_at_k(df, k=5, score_col="target_policy_score"):
    total_relevant = df["clicked"].sum()
    if total_relevant == 0:
        return 0.0

    df = df.sort_values(score_col, ascending=False).head(k)
    return df["clicked"].sum() / total_relevant


def dcg_at_k(df, k=5, score_col="target_policy_score"):
    df = df.sort_values(score_col, ascending=False).head(k)
    gains = df["clicked"].values
    return np.sum([gains[i] / np.log2(i + 2) for i in range(len(gains))])


def ndcg_at_k(df, k=5, score_col="target_policy_score"):
    ideal_df = df.sort_values("clicked", ascending=False)
    ideal_dcg = dcg_at_k(ideal_df, k, score_col="clicked")

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(df, k, score_col) / ideal_dcg


def evaluate_ranking(df, k=5, score_col="target_policy_score"):
    results = []

    for _, session_df in df.groupby("session_id"):
        results.append(
            {
                "precision@k": precision_at_k(session_df, k, score_col),
                "recall@k": recall_at_k(session_df, k, score_col),
                "ndcg@k": ndcg_at_k(session_df, k, score_col),
            }
        )

    return pd.DataFrame(results).mean().to_dict()


def main():
    df = pd.read_parquet("data/gold/simulated_logged_policy_v2.parquet")

    print("=== Logged Policy Metrics ===")
    logged_metrics = evaluate_ranking(df, k=5, score_col="logged_policy_score")
    print(logged_metrics)

    print("\n=== Target Policy Metrics ===")
    target_metrics = evaluate_ranking(df, k=5, score_col="target_policy_score")
    print(target_metrics)

    print("\n=== Improvement ===")
    print({
    k: target_metrics[k] - logged_metrics[k]
    for k in logged_metrics})


if __name__ == "__main__":
    main()