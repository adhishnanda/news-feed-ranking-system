from pathlib import Path
import pandas as pd


IMPRESSIONS_PATH = Path("data/silver/impressions_v2.parquet")


def load_data() -> pd.DataFrame:
    if not IMPRESSIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {IMPRESSIONS_PATH}")

    df = pd.read_parquet(IMPRESSIONS_PATH)
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    return df


def get_popular_candidates(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    out = (
        df.groupby("item_id", as_index=False)
        .agg(
            popularity_clicks=("clicked", "sum"),
            popularity_impressions=("item_id", "count"),
            last_seen_time=("impression_time", "max"),
            source=("source", "last"),
            category=("category", "last"),
        )
        .sort_values(["popularity_clicks", "popularity_impressions", "last_seen_time"], ascending=[False, False, False])
        .head(top_k)
        .copy()
    )
    out["retrieval_strategy"] = "popular"
    out["retrieval_score"] = out["popularity_clicks"]
    return out


def get_recent_candidates(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    out = (
        df.sort_values("impression_time", ascending=False)
        .drop_duplicates(subset=["item_id"])
        .head(top_k)
        .copy()
    )

    keep_cols = ["item_id", "impression_time", "source", "category"]
    out = out[keep_cols].rename(columns={"impression_time": "last_seen_time"})
    out["retrieval_strategy"] = "recent"
    out["retrieval_score"] = range(len(out), 0, -1)
    return out


def get_category_candidates(df: pd.DataFrame, category: str, top_k: int = 20) -> pd.DataFrame:
    filtered = df[df["category"].astype(str) == str(category)].copy()

    if filtered.empty:
        return pd.DataFrame(columns=["item_id", "last_seen_time", "source", "category", "retrieval_strategy", "retrieval_score"])

    out = (
        filtered.groupby("item_id", as_index=False)
        .agg(
            category_clicks=("clicked", "sum"),
            category_impressions=("item_id", "count"),
            last_seen_time=("impression_time", "max"),
            source=("source", "last"),
            category=("category", "last"),
        )
        .sort_values(["category_clicks", "category_impressions", "last_seen_time"], ascending=[False, False, False])
        .head(top_k)
        .copy()
    )
    out["retrieval_strategy"] = "category_match"
    out["retrieval_score"] = out["category_clicks"]
    return out


def get_source_candidates(df: pd.DataFrame, source: str, top_k: int = 20) -> pd.DataFrame:
    filtered = df[df["source"].astype(str) == str(source)].copy()

    if filtered.empty:
        return pd.DataFrame(columns=["item_id", "last_seen_time", "source", "category", "retrieval_strategy", "retrieval_score"])

    out = (
        filtered.groupby("item_id", as_index=False)
        .agg(
            source_clicks=("clicked", "sum"),
            source_impressions=("item_id", "count"),
            last_seen_time=("impression_time", "max"),
            source=("source", "last"),
            category=("category", "last"),
        )
        .sort_values(["source_clicks", "source_impressions", "last_seen_time"], ascending=[False, False, False])
        .head(top_k)
        .copy()
    )
    out["retrieval_strategy"] = "source_match"
    out["retrieval_score"] = out["source_clicks"]
    return out


def get_user_unseen_candidates(df: pd.DataFrame, user_id: str, top_k: int = 20) -> pd.DataFrame:
    user_df = df[df["user_id"].astype(str) == str(user_id)].copy()
    seen_items = set(user_df["item_id"].astype(str).tolist())

    global_recent = (
        df.sort_values("impression_time", ascending=False)
        .drop_duplicates(subset=["item_id"])
        .copy()
    )

    global_recent = global_recent[~global_recent["item_id"].astype(str).isin(seen_items)].head(top_k).copy()

    if global_recent.empty:
        return pd.DataFrame(columns=["item_id", "last_seen_time", "source", "category", "retrieval_strategy", "retrieval_score"])

    out = global_recent[["item_id", "impression_time", "source", "category"]].rename(
        columns={"impression_time": "last_seen_time"}
    )
    out["retrieval_strategy"] = "unseen_recent"
    out["retrieval_score"] = range(len(out), 0, -1)
    return out


def get_user_preferences(df: pd.DataFrame, user_id: str):
    user_df = df[df["user_id"].astype(str) == str(user_id)].copy()

    if user_df.empty:
        return {"preferred_category": None, "preferred_source": None}

    preferred_category = None
    preferred_source = None

    if "preferred_category" in user_df.columns and user_df["preferred_category"].notna().any():
        preferred_category = user_df["preferred_category"].dropna().astype(str).iloc[-1]

    if "preferred_source" in user_df.columns and user_df["preferred_source"].notna().any():
        preferred_source = user_df["preferred_source"].dropna().astype(str).iloc[-1]

    if preferred_category is None and "category" in user_df.columns:
        vc = user_df["category"].dropna().astype(str).value_counts()
        preferred_category = vc.index[0] if len(vc) else None

    if preferred_source is None and "source" in user_df.columns:
        vc = user_df["source"].dropna().astype(str).value_counts()
        preferred_source = vc.index[0] if len(vc) else None

    return {
        "preferred_category": preferred_category,
        "preferred_source": preferred_source,
    }


def main():
    df = load_data()

    sample_user = str(df["user_id"].iloc[0])
    prefs = get_user_preferences(df, sample_user)

    print("Sample user:", sample_user)
    print("Preferences:", prefs)

    print("\nPopular:")
    print(get_popular_candidates(df, top_k=5)[["item_id", "retrieval_strategy", "retrieval_score"]])

    print("\nRecent:")
    print(get_recent_candidates(df, top_k=5)[["item_id", "retrieval_strategy", "retrieval_score"]])

    print("\nCategory:")
    if prefs["preferred_category"] is not None:
        print(get_category_candidates(df, prefs["preferred_category"], top_k=5)[["item_id", "retrieval_strategy", "retrieval_score"]])
    else:
        print("No preferred category available")

    print("\nSource:")
    if prefs["preferred_source"] is not None:
        print(get_source_candidates(df, prefs["preferred_source"], top_k=5)[["item_id", "retrieval_strategy", "retrieval_score"]])
    else:
        print("No preferred source available")

    print("\nUnseen recent:")
    print(get_user_unseen_candidates(df, sample_user, top_k=5)[["item_id", "retrieval_strategy", "retrieval_score"]])


if __name__ == "__main__":
    main()