from __future__ import annotations
import pandas as pd
import numpy as np

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import write_parquet


def build_item_features() -> pd.DataFrame:
    cfg = load_yaml("configs/config.yaml")
    db = DuckDBClient(cfg["paths"]["duckdb"])

    df = db.query_df("""
        SELECT
            item_id,
            source,
            source_type,
            title,
            description,
            published_at,
            category
        FROM content_items
    """)

    now_ts = pd.Timestamp.utcnow()

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    df["age_hours"] = (now_ts - df["published_at"]).dt.total_seconds() / 3600.0
    df["age_hours"] = df["age_hours"].fillna(df["age_hours"].median())

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["source"] = df["source"].fillna("unknown")
    df["source_type"] = df["source_type"].fillna("unknown")
    df["category"] = df["category"].fillna("unknown")

    df["title_length"] = df["title"].str.len()
    df["description_length"] = df["description"].str.len()

    df["hour_published"] = df["published_at"].dt.hour.fillna(-1).astype(int)
    df["weekday_published"] = df["published_at"].dt.weekday.fillna(-1).astype(int)

    df["is_hackernews"] = (df["source_type"] == "hackernews").astype(int)
    df["is_rss"] = (df["source_type"] == "rss").astype(int)

    item_features = df[[
        "item_id",
        "source",
        "source_type",
        "category",
        "age_hours",
        "title_length",
        "description_length",
        "hour_published",
        "weekday_published",
        "is_hackernews",
        "is_rss",
    ]].copy()

    return item_features


def main():
    cfg = load_yaml("configs/config.yaml")
    item_features = build_item_features()

    output_path = f"{cfg['paths']['gold_dir']}/item_features.parquet"
    write_parquet(item_features, output_path)

    print(f"Saved item features: {output_path}")
    print(item_features.head())


if __name__ == "__main__":
    main()