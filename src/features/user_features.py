from __future__ import annotations
import pandas as pd

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import write_parquet


def build_user_features() -> pd.DataFrame:
    cfg = load_yaml("configs/config.yaml")
    db = DuckDBClient(cfg["paths"]["duckdb"])

    events = db.query_df("""
        SELECT
            e.user_id,
            e.event_type,
            e.item_id,
            e.timestamp,
            c.source,
            c.category
        FROM events e
        LEFT JOIN content_items c
            ON e.item_id = c.item_id
    """)

    if events.empty:
        return pd.DataFrame(columns=[
            "user_id",
            "recent_impression_count",
            "recent_click_count",
            "recent_save_count",
            "recent_hide_count",
            "preferred_source",
            "preferred_category",
        ])

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")

    counts = events.pivot_table(
        index="user_id",
        columns="event_type",
        values="item_id",
        aggfunc="count",
        fill_value=0
    ).reset_index()

    counts.columns.name = None

    for col in ["impression", "click", "save", "hide"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts.rename(columns={
        "impression": "recent_impression_count",
        "click": "recent_click_count",
        "save": "recent_save_count",
        "hide": "recent_hide_count",
    })

    click_events = events[events["event_type"] == "click"].copy()

    if click_events.empty:
        pref_source = pd.DataFrame(columns=["user_id", "preferred_source"])
        pref_category = pd.DataFrame(columns=["user_id", "preferred_category"])
    else:
        pref_source = (
            click_events.groupby(["user_id", "source"])
            .size()
            .reset_index(name="n")
            .sort_values(["user_id", "n"], ascending=[True, False])
            .drop_duplicates("user_id")
            [["user_id", "source"]]
            .rename(columns={"source": "preferred_source"})
        )

        pref_category = (
            click_events.groupby(["user_id", "category"])
            .size()
            .reset_index(name="n")
            .sort_values(["user_id", "n"], ascending=[True, False])
            .drop_duplicates("user_id")
            [["user_id", "category"]]
            .rename(columns={"category": "preferred_category"})
        )

    user_features = counts.merge(pref_source, on="user_id", how="left")
    user_features = user_features.merge(pref_category, on="user_id", how="left")

    user_features["preferred_source"] = user_features["preferred_source"].fillna("unknown")
    user_features["preferred_category"] = user_features["preferred_category"].fillna("unknown")

    return user_features[[
        "user_id",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
        "preferred_source",
        "preferred_category",
    ]]


def main():
    cfg = load_yaml("configs/config.yaml")
    user_features = build_user_features()

    output_path = f"{cfg['paths']['gold_dir']}/user_features.parquet"
    write_parquet(user_features, output_path)

    print(f"Saved user features: {output_path}")
    print(user_features.head())


if __name__ == "__main__":
    main()