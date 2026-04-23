from pathlib import Path
import pandas as pd

from src.reranking.candidate_generation_stub import (
    load_data,
    get_popular_candidates,
    get_recent_candidates,
    get_category_candidates,
    get_source_candidates,
    get_user_unseen_candidates,
    get_user_preferences,
)


OUTPUT_PATH = Path("data/gold/candidates_v2.parquet")


def standardize_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "item_id",
            "last_seen_time",
            "source",
            "category",
            "retrieval_strategy",
            "retrieval_score",
        ])

    keep_cols = ["item_id", "last_seen_time", "source", "category", "retrieval_strategy", "retrieval_score"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[keep_cols].copy()


def retrieve_candidates_for_user(
    df: pd.DataFrame,
    user_id: str,
    top_k_each: int = 20,
    final_top_k: int = 50,
) -> pd.DataFrame:
    prefs = get_user_preferences(df, user_id)

    candidate_frames = []

    candidate_frames.append(standardize_candidate_frame(get_popular_candidates(df, top_k_each)))
    candidate_frames.append(standardize_candidate_frame(get_recent_candidates(df, top_k_each)))
    candidate_frames.append(standardize_candidate_frame(get_user_unseen_candidates(df, user_id, top_k_each)))

    if prefs["preferred_category"] is not None:
        candidate_frames.append(
            standardize_candidate_frame(get_category_candidates(df, prefs["preferred_category"], top_k_each))
        )

    if prefs["preferred_source"] is not None:
        candidate_frames.append(
            standardize_candidate_frame(get_source_candidates(df, prefs["preferred_source"], top_k_each))
        )

    non_empty = [f for f in candidate_frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    all_candidates = pd.concat(non_empty, ignore_index=True)

    if all_candidates.empty:
        return all_candidates

    # Blend retrieval sources
    strategy_weight_map = {
        "popular": 1.00,
        "recent": 0.90,
        "unseen_recent": 0.95,
        "category_match": 1.10,
        "source_match": 1.05,
    }

    all_candidates["strategy_weight"] = all_candidates["retrieval_strategy"].map(strategy_weight_map).fillna(1.0)
    all_candidates["blended_retrieval_score"] = (
        pd.to_numeric(all_candidates["retrieval_score"], errors="coerce").fillna(0.0)
        * all_candidates["strategy_weight"]
    )

    # keep trace of how many strategies retrieved each item
    strategy_count = (
        all_candidates.groupby("item_id")["retrieval_strategy"]
        .nunique()
        .reset_index()
        .rename(columns={"retrieval_strategy": "retrieval_strategy_count"})
    )

    # aggregate duplicate items
    final_candidates = (
        all_candidates.groupby("item_id", as_index=False)
        .agg(
            blended_retrieval_score=("blended_retrieval_score", "max"),
            retrieval_score_raw=("retrieval_score", "max"),
            last_seen_time=("last_seen_time", "max"),
            source=("source", "last"),
            category=("category", "last"),
        )
    )

    final_candidates = final_candidates.merge(strategy_count, on="item_id", how="left")

    final_candidates = final_candidates.sort_values(
        ["retrieval_strategy_count", "blended_retrieval_score", "last_seen_time"],
        ascending=[False, False, False],
    ).head(final_top_k).reset_index(drop=True)

    final_candidates["user_id"] = str(user_id)

    return final_candidates


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()

    all_users = df["user_id"].astype(str).dropna().unique().tolist()
    output_frames = []

    for user_id in all_users:
        cands = retrieve_candidates_for_user(df, user_id=user_id, top_k_each=10, final_top_k=25)
        output_frames.append(cands)

    final_df = pd.concat(output_frames, ignore_index=True) if output_frames else pd.DataFrame()
    final_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {final_df.shape}")
    print(final_df.head())


if __name__ == "__main__":
    main()