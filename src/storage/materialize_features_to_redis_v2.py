from pathlib import Path
import pandas as pd

from src.storage.redis_feature_store_v2 import RedisFeatureStoreV2


TRAIN_PATH = Path("data/gold/training_dataset_simulated_v3.parquet")
TTL_SECONDS = 60 * 60 * 24  # 24 hours


def load_data() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")

    df = pd.read_parquet(TRAIN_PATH).copy()

    # Support both schemas:
    # old V2 -> impression_time
    # simulated V3 -> timestamp
    if "impression_time" in df.columns:
        df["feature_time"] = pd.to_datetime(df["impression_time"], errors="coerce")
    elif "timestamp" in df.columns:
        df["feature_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("Expected either 'impression_time' or 'timestamp' column in training dataset")

    return df


def get_user_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        # original V2
        "user_prev_clicks",
        "user_prev_impressions",
        "user_ctr_prior",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
        "user_prev_category",
        "user_prev_source",
        "user_embedding_norm",

        # simulated V3
        "user_prev_clicks_sim",
        "user_prev_impressions_sim",
        "user_ctr_prior_sim",
        "preferred_category",
        "preferred_source",
    ]
    return [c for c in candidates if c in df.columns]


def get_item_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        # original V2
        "item_prev_impressions",
        "item_prev_clicks",
        "item_ctr_prior",
        "item_age_hours",
        "source",
        "source_type",
        "category",
        "title_length",
        "description_length",
        "is_hackernews",
        "is_rss",
        "item_embedding_norm",

        # simulated V3
        "item_prev_impressions_sim",
        "item_prev_clicks_sim",
        "item_ctr_prior_sim",
        "age_hours",
        "hour_published",
        "weekday_published",
    ]
    return [c for c in candidates if c in df.columns]


def normalize_user_payload(payload: dict) -> dict:
    """
    Normalize simulated-v3 and original-v2 fields into a serving-friendly online schema.
    """
    out = dict(payload)

    if "user_prev_clicks_sim" in out and "user_prev_clicks" not in out:
        out["user_prev_clicks"] = out["user_prev_clicks_sim"]

    if "user_prev_impressions_sim" in out and "user_prev_impressions" not in out:
        out["user_prev_impressions"] = out["user_prev_impressions_sim"]

    if "user_ctr_prior_sim" in out and "user_ctr_prior" not in out:
        out["user_ctr_prior"] = out["user_ctr_prior_sim"]

    if "preferred_category" in out and "user_prev_category" not in out:
        out["user_prev_category"] = out["preferred_category"]

    if "preferred_source" in out and "user_prev_source" not in out:
        out["user_prev_source"] = out["preferred_source"]

    return out


def normalize_item_payload(payload: dict) -> dict:
    """
    Normalize simulated-v3 and original-v2 fields into a serving-friendly online schema.
    """
    out = dict(payload)

    if "item_prev_impressions_sim" in out and "item_prev_impressions" not in out:
        out["item_prev_impressions"] = out["item_prev_impressions_sim"]

    if "item_prev_clicks_sim" in out and "item_prev_clicks" not in out:
        out["item_prev_clicks"] = out["item_prev_clicks_sim"]

    if "item_ctr_prior_sim" in out and "item_ctr_prior" not in out:
        out["item_ctr_prior"] = out["item_ctr_prior_sim"]

    if "age_hours" in out and "item_age_hours" not in out:
        out["item_age_hours"] = out["age_hours"]

    return out


def to_clean_dict(row: pd.Series, cols: list[str]) -> dict:
    out = {}
    for col in cols:
        value = row.get(col)
        if pd.isna(value):
            value = None
        out[col] = value
    return out


def main():
    df = load_data()
    store = RedisFeatureStoreV2()

    if not store.ping():
        raise RuntimeError("Redis is not reachable on localhost:6379")

    user_cols = get_user_feature_columns(df)
    item_cols = get_item_feature_columns(df)

    latest_user = (
        df.sort_values("feature_time")
        .drop_duplicates(subset=["user_id"], keep="last")
        .copy()
    )

    latest_item = (
        df.sort_values("feature_time")
        .drop_duplicates(subset=["item_id"], keep="last")
        .copy()
    )

    for _, row in latest_user.iterrows():
        user_id = str(row["user_id"])
        payload = to_clean_dict(row, user_cols)
        payload = normalize_user_payload(payload)
        payload["feature_timestamp"] = str(row["feature_time"])
        store.put_user_features(user_id, payload, ttl_seconds=TTL_SECONDS)

    for _, row in latest_item.iterrows():
        item_id = str(row["item_id"])
        payload = to_clean_dict(row, item_cols)
        payload = normalize_item_payload(payload)
        payload["feature_timestamp"] = str(row["feature_time"])
        store.put_item_features(item_id, payload, ttl_seconds=TTL_SECONDS)

    store.put_metadata(
        "materialization_status",
        {
            "user_count": int(len(latest_user)),
            "item_count": int(len(latest_item)),
            "source_dataset": str(TRAIN_PATH),
        },
        ttl_seconds=TTL_SECONDS,
    )

    print("Materialization complete.")
    print(f"Users materialized: {len(latest_user)}")
    print(f"Items materialized: {len(latest_item)}")
    print("Metadata:", store.get_metadata("materialization_status"))

    if len(latest_user) > 0:
        sample_user = str(latest_user.iloc[0]["user_id"])
        print(f"\nSample user key user:{sample_user}")
        print(store.get_user_features(sample_user))

    if len(latest_item) > 0:
        sample_item = str(latest_item.iloc[0]["item_id"])
        print(f"\nSample item key item:{sample_item}")
        print(store.get_item_features(sample_item))


if __name__ == "__main__":
    main()