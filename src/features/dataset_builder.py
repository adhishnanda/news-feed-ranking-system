from __future__ import annotations
import pandas as pd

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import read_parquet, write_parquet


def build_training_dataset() -> pd.DataFrame:
    cfg = load_yaml("configs/config.yaml")
    db = DuckDBClient(cfg["paths"]["duckdb"])

    item_features = read_parquet(f"{cfg['paths']['gold_dir']}/item_features.parquet")
    user_features = read_parquet(f"{cfg['paths']['gold_dir']}/user_features.parquet")
    context_features = read_parquet(f"{cfg['paths']['gold_dir']}/context_features.parquet")

    impressions = db.query_df("""
        SELECT
            event_id,
            timestamp,
            user_id,
            session_id,
            item_id,
            rank_position,
            model_version,
            policy_name
        FROM events
        WHERE event_type = 'impression'
    """)

    clicks = db.query_df("""
        SELECT DISTINCT
            user_id,
            session_id,
            item_id
        FROM events
        WHERE event_type = 'click'
    """)

    if impressions.empty:
        return pd.DataFrame()

    impressions["clicked"] = 0

    if not clicks.empty:
        clicks["clicked"] = 1
        impressions = impressions.merge(
            clicks,
            on=["user_id", "session_id", "item_id"],
            how="left",
            suffixes=("", "_click")
        )
        impressions["clicked"] = impressions["clicked_click"].fillna(impressions["clicked"]).astype(int)
        impressions = impressions.drop(columns=["clicked_click"])

    dataset = impressions.merge(item_features, on="item_id", how="left")
    dataset = dataset.merge(user_features, on="user_id", how="left")
    dataset = dataset.merge(context_features, on="event_id", how="left")

    dataset["preferred_source_match"] = (
        dataset["source"].fillna("unknown") == dataset["preferred_source"].fillna("unknown")
    ).astype(int)

    dataset["preferred_category_match"] = (
        dataset["category"].fillna("unknown") == dataset["preferred_category"].fillna("unknown")
    ).astype(int)

    dataset["recent_impression_count"] = dataset["recent_impression_count"].fillna(0)
    dataset["recent_click_count"] = dataset["recent_click_count"].fillna(0)
    dataset["recent_save_count"] = dataset["recent_save_count"].fillna(0)
    dataset["recent_hide_count"] = dataset["recent_hide_count"].fillna(0)

    return dataset


def main():
    cfg = load_yaml("configs/config.yaml")
    dataset = build_training_dataset()

    output_path = f"{cfg['paths']['gold_dir']}/training_dataset.parquet"
    write_parquet(dataset, output_path)

    print(f"Saved training dataset: {output_path}")
    print(dataset.head())
    print("\nClicked label distribution:")
    if "clicked" in dataset.columns:
        print(dataset["clicked"].value_counts(dropna=False))


if __name__ == "__main__":
    main()