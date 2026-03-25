from pathlib import Path
import pandas as pd
import numpy as np


IMPRESSIONS_PATH = Path("data/silver/impressions_v2.parquet")
USER_FEATURES_PATH = Path("data/silver/user_features_v2.parquet")
ITEM_FEATURES_PATH = Path("data/silver/item_features_v2.parquet")


def load_impressions() -> pd.DataFrame:
    if not IMPRESSIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {IMPRESSIONS_PATH}")
    df = pd.read_parquet(IMPRESSIONS_PATH)
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    return df


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["user_id", "impression_time"]).reset_index(drop=True)

    # Recompute truly prior user aggregates from event history
    df["user_prev_clicks"] = (
        df.groupby("user_id")["clicked"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0)
    )

    df["user_prev_impressions"] = df.groupby("user_id").cumcount()

    df["user_ctr_prior"] = (
        df["user_prev_clicks"] / df["user_prev_impressions"].replace(0, np.nan)
    ).fillna(0.0)

    # Reuse preferred fields if already present, otherwise derive simple fallback
    if "preferred_category" in df.columns:
        df["user_prev_category"] = df["preferred_category"].fillna("unknown")
    else:
        df["user_prev_category"] = (
            df.groupby("user_id")["category"].shift(1).fillna("unknown")
        )

    if "preferred_source" in df.columns:
        df["user_prev_source"] = df["preferred_source"].fillna("unknown")
    else:
        df["user_prev_source"] = (
            df.groupby("user_id")["source"].shift(1).fillna("unknown")
        )

    # Preserve existing recency features if available
    for col in ["recent_impression_count", "recent_click_count", "recent_save_count", "recent_hide_count"]:
        if col not in df.columns:
            df[col] = 0

    user_features = df[
        [
            "user_id",
            "impression_time",
            "user_prev_clicks",
            "user_prev_impressions",
            "user_ctr_prior",
            "user_prev_category",
            "user_prev_source",
            "recent_impression_count",
            "recent_click_count",
            "recent_save_count",
            "recent_hide_count",
        ]
    ].copy()

    user_features = user_features.rename(columns={"impression_time": "feature_timestamp"})
    return user_features


def build_item_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy().sort_values(["item_id", "impression_time"]).reset_index(drop=True)

    base["item_prev_impressions"] = base.groupby("item_id").cumcount()

    base["item_prev_clicks"] = (
        base.groupby("item_id")["clicked"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0)
    )

    base["item_ctr_prior"] = (
        base["item_prev_clicks"] / base["item_prev_impressions"].replace(0, np.nan)
    ).fillna(0.0)

    item_features = base[
        [
            "item_id",
            "impression_time",
            "item_prev_impressions",
            "item_prev_clicks",
            "item_ctr_prior",
            "item_age_hours",
            "source",
            "category",
            "title_length" if "title_length" in base.columns else None,
            "description_length" if "description_length" in base.columns else None,
            "is_hackernews" if "is_hackernews" in base.columns else None,
            "is_rss" if "is_rss" in base.columns else None,
        ]
    ].copy()

    item_features = item_features[[c for c in item_features.columns if c is not None]]
    item_features = item_features.rename(columns={"impression_time": "feature_timestamp"})
    return item_features


def main():
    USER_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ITEM_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_impressions()

    user_features = build_user_features(df)
    item_features = build_item_features(df)

    user_features.to_parquet(USER_FEATURES_PATH, index=False)
    item_features.to_parquet(ITEM_FEATURES_PATH, index=False)

    print(f"Saved: {USER_FEATURES_PATH} | shape={user_features.shape}")
    print(f"Saved: {ITEM_FEATURES_PATH} | shape={item_features.shape}")


if __name__ == "__main__":
    main()