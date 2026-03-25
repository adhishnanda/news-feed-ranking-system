from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/gold/training_dataset_simulated_v3.parquet")
OUTPUT_PATH = Path("data/gold/simulated_logged_policy_v2.parquet")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def build_logged_policy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Logged policy score: a simple historical production-style policy
    # based on prior CTR + preference match + recency + position.
    item_ctr = pd.to_numeric(out.get("item_ctr_prior_sim", 0.0), errors="coerce").fillna(0.0)
    user_ctr = pd.to_numeric(out.get("user_ctr_prior_sim", 0.0), errors="coerce").fillna(0.0)
    recency = pd.to_numeric(out.get("recency_decay_sim", 0.0), errors="coerce").fillna(0.0)
    cat_match = pd.to_numeric(out.get("preferred_category_match", 0.0), errors="coerce").fillna(0.0)
    src_match = pd.to_numeric(out.get("preferred_source_match", 0.0), errors="coerce").fillna(0.0)
    rank_pos = pd.to_numeric(out.get("rank_position", 999), errors="coerce").fillna(999.0)

    position_bias = 1.0 / rank_pos.clip(lower=1.0)

    logged_policy_score = (
        1.8 * item_ctr
        + 0.8 * user_ctr
        + 0.7 * recency
        + 0.8 * cat_match
        + 0.5 * src_match
        + 1.0 * position_bias
    )

    out["logged_policy_score"] = logged_policy_score

    # session-wise softmax to get propensities
    def softmax_probs(s: pd.Series) -> pd.Series:
        x = s.to_numpy(dtype=float)
        x = x - np.max(x)
        ex = np.exp(x)
        probs = ex / ex.sum()
        return pd.Series(probs, index=s.index)

    out["logged_propensity"] = (
        out.groupby("session_id")["logged_policy_score"]
        .transform(softmax_probs)
    )

    # A candidate target policy score (new policy) that is slightly different
    # from the logged policy and emphasizes more recency + preference alignment.
    target_policy_score = (
        1.2 * item_ctr
        + 0.6 * user_ctr
        + 1.2 * recency
        + 1.0 * cat_match
        + 0.7 * src_match
        + 0.6 * position_bias
    )

    out["target_policy_score"] = target_policy_score
    out["target_propensity"] = (
        out.groupby("session_id")["target_policy_score"]
        .transform(softmax_probs)
    )

    return out


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    out = build_logged_policy(df)
    out.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved logged policy dataset to: {OUTPUT_PATH}")
    print(f"Shape: {out.shape}")
    print("\nColumns added:")
    print(["logged_policy_score", "logged_propensity", "target_policy_score", "target_propensity"])
    print("\nHead:")
    print(out[[
        "session_id",
        "item_id",
        "clicked",
        "logged_policy_score",
        "logged_propensity",
        "target_policy_score",
        "target_propensity",
    ]].head())

if __name__ == "__main__":
    main()