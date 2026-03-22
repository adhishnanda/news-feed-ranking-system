from __future__ import annotations
import pandas as pd

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import write_parquet


def build_context_features() -> pd.DataFrame:
    cfg = load_yaml("configs/config.yaml")
    db = DuckDBClient(cfg["paths"]["duckdb"])

    impressions = db.query_df("""
        SELECT
            event_id,
            user_id,
            session_id,
            item_id,
            rank_position,
            timestamp
        FROM events
        WHERE event_type = 'impression'
    """)

    if impressions.empty:
        return pd.DataFrame(columns=[
            "event_id",
            "hour_of_day",
            "weekday",
            "is_weekend",
        ])

    impressions["timestamp"] = pd.to_datetime(impressions["timestamp"], utc=True, errors="coerce")
    impressions["hour_of_day"] = impressions["timestamp"].dt.hour
    impressions["weekday"] = impressions["timestamp"].dt.weekday
    impressions["is_weekend"] = impressions["weekday"].isin([5, 6]).astype(int)

    context_features = impressions[[
        "event_id",
        "hour_of_day",
        "weekday",
        "is_weekend",
    ]].copy()

    return context_features


def main():
    cfg = load_yaml("configs/config.yaml")
    context_features = build_context_features()

    output_path = f"{cfg['paths']['gold_dir']}/context_features.parquet"
    write_parquet(context_features, output_path)

    print(f"Saved context features: {output_path}")
    print(context_features.head())


if __name__ == "__main__":
    main()