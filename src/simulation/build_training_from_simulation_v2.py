from pathlib import Path
import pandas as pd
import numpy as np


INPUT_EVENTS_PATH = Path("data/logs/simulated_events_v2.parquet")
OUTPUT_TRAIN_PATH = Path("data/gold/training_dataset_simulated_v2.parquet")


def load_events() -> pd.DataFrame:
    if not INPUT_EVENTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_EVENTS_PATH}")

    df = pd.read_parquet(INPUT_EVENTS_PATH).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["user_prev_clicks_sim"] = (
        df.groupby("user_id")["clicked"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0)
    )

    df["user_prev_impressions_sim"] = df.groupby("user_id").cumcount()

    df["user_ctr_prior_sim"] = (
        df["user_prev_clicks_sim"] / df["user_prev_impressions_sim"].replace(0, np.nan)
    ).fillna(0.0)

    item_df = df.copy().sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    item_df["item_prev_clicks_sim"] = (
        item_df.groupby("item_id")["clicked"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0)
    )
    item_df["item_prev_impressions_sim"] = item_df.groupby("item_id").cumcount()
    item_df["item_ctr_prior_sim"] = (
        item_df["item_prev_clicks_sim"] / item_df["item_prev_impressions_sim"].replace(0, np.nan)
    ).fillna(0.0)

    df = df.merge(
        item_df[
            [
                "event_id",
                "item_prev_clicks_sim",
                "item_prev_impressions_sim",
                "item_ctr_prior_sim",
            ]
        ],
        on="event_id",
        how="left",
    )

    df["recency_decay_sim"] = 1.0 / (1.0 + pd.to_numeric(df["age_hours"], errors="coerce").fillna(0.0))

    return df


def main():
    OUTPUT_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_events()
    out = build_historical_features(df)

    out.to_parquet(OUTPUT_TRAIN_PATH, index=False)

    print(f"Saved simulated training dataset to: {OUTPUT_TRAIN_PATH}")
    print(f"Shape: {out.shape}")
    print("\nClicked distribution:")
    print(out["clicked"].value_counts(dropna=False))
    print("\nColumns:")
    print(out.columns.tolist())


if __name__ == "__main__":
    main()