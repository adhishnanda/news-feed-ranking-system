from pathlib import Path
import pandas as pd
import numpy as np


INPUT_PATH = Path("data/gold/training_dataset.parquet")
OUTPUT_PATH = Path("data/silver/impressions_v2.parquet")


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_impressions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["user_id", "item_id", "timestamp", "clicked"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "timestamp", "clicked"]).copy()

    df["clicked"] = pd.to_numeric(df["clicked"], errors="coerce").fillna(0).astype(int)
    df["clicked"] = df["clicked"].clip(lower=0, upper=1)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep event_id if present, otherwise create impression_id
    if "event_id" in df.columns:
        df["impression_id"] = df["event_id"].astype(str)
    else:
        df["impression_id"] = np.arange(1, len(df) + 1).astype(str)

    # Standardized time/context columns
    df["impression_time"] = df["timestamp"]
    df["hour"] = df["impression_time"].dt.hour
    df["day_of_week"] = df["impression_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Standardize age feature name
    if "age_hours" in df.columns:
        df["item_age_hours"] = pd.to_numeric(df["age_hours"], errors="coerce").fillna(0.0)
        df["item_age_hours"] = df["item_age_hours"].clip(lower=0)
    else:
        df["item_age_hours"] = 0.0

    # Optional defaults
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "source_type" not in df.columns:
        df["source_type"] = "unknown"
    if "category" not in df.columns:
        df["category"] = "unknown"
    if "session_id" not in df.columns:
        df["session_id"] = "unknown"
    if "rank_position" not in df.columns:
        df["rank_position"] = -1
    if "model_version" not in df.columns:
        df["model_version"] = "unknown"
    if "policy_name" not in df.columns:
        df["policy_name"] = "unknown"

    keep_cols = [
        "impression_id",
        "event_id" if "event_id" in df.columns else None,
        "user_id",
        "session_id",
        "item_id",
        "impression_time",
        "clicked",
        "rank_position",
        "model_version",
        "policy_name",
        "source",
        "source_type",
        "category",
        "hour",
        "day_of_week",
        "is_weekend",
        "item_age_hours",
        "title_length" if "title_length" in df.columns else None,
        "description_length" if "description_length" in df.columns else None,
        "hour_published" if "hour_published" in df.columns else None,
        "weekday_published" if "weekday_published" in df.columns else None,
        "is_hackernews" if "is_hackernews" in df.columns else None,
        "is_rss" if "is_rss" in df.columns else None,
        "recent_impression_count" if "recent_impression_count" in df.columns else None,
        "recent_click_count" if "recent_click_count" in df.columns else None,
        "recent_save_count" if "recent_save_count" in df.columns else None,
        "recent_hide_count" if "recent_hide_count" in df.columns else None,
        "preferred_source" if "preferred_source" in df.columns else None,
        "preferred_category" if "preferred_category" in df.columns else None,
        "preferred_source_match" if "preferred_source_match" in df.columns else None,
        "preferred_category_match" if "preferred_category_match" in df.columns else None,
    ]

    keep_cols = [c for c in keep_cols if c is not None]
    return df[keep_cols].copy()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    impressions = build_impressions(df)

    impressions.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {impressions.shape}")
    print(impressions.head())
    print("\nColumns:")
    print(impressions.columns.tolist())


if __name__ == "__main__":
    main()