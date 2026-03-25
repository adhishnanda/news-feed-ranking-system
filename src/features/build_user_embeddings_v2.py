from pathlib import Path
import pandas as pd
import numpy as np


IMPRESSIONS_PATH = Path("data/silver/impressions_v2.parquet")
ITEM_EMB_PATH = Path("data/silver/item_embeddings_v2.parquet")
OUTPUT_PATH = Path("data/silver/user_embeddings_v2.parquet")


def load_data():
    if not IMPRESSIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {IMPRESSIONS_PATH}")
    if not ITEM_EMB_PATH.exists():
        raise FileNotFoundError(f"Missing file: {ITEM_EMB_PATH}")

    impressions = pd.read_parquet(IMPRESSIONS_PATH)
    item_emb = pd.read_parquet(ITEM_EMB_PATH)

    impressions["impression_time"] = pd.to_datetime(impressions["impression_time"])
    return impressions, item_emb


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    impressions, item_emb = load_data()

    emb_cols = [c for c in item_emb.columns if c.startswith("item_emb_")]
    if not emb_cols:
        raise ValueError("No item embedding columns found.")

    # use clicked items to represent user preference
    clicked_df = impressions[impressions["clicked"] == 1].copy()

    if clicked_df.empty:
        # fallback: use all impressions if no clicks exist
        clicked_df = impressions.copy()

    merged = clicked_df.merge(item_emb[["item_id"] + emb_cols], on="item_id", how="left")

    # fill missing embedding rows with zeros
    for col in emb_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    user_emb = (
        merged.groupby("user_id", as_index=False)[emb_cols]
        .mean()
        .copy()
    )

    user_emb_cols = {}
    for col in emb_cols:
        idx = col.replace("item_emb_", "")
        user_emb_cols[col] = f"user_emb_{idx}"

    user_emb = user_emb.rename(columns=user_emb_cols)

    output_df = user_emb.copy()
    output_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved user embeddings to: {OUTPUT_PATH}")
    print(f"Shape: {output_df.shape}")
    print(output_df.head())


if __name__ == "__main__":
    main()