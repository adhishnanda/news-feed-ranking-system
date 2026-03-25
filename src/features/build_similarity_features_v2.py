from pathlib import Path
import pandas as pd
import numpy as np


TRAIN_DATA_PATH = Path("data/gold/train_dataset_v2.parquet")
ITEM_EMB_PATH = Path("data/silver/item_embeddings_v2.parquet")
USER_EMB_PATH = Path("data/silver/user_embeddings_v2.parquet")
OUTPUT_PATH = Path("data/gold/train_dataset_v2_embeddings.parquet")


def load_data():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_DATA_PATH}")
    if not ITEM_EMB_PATH.exists():
        raise FileNotFoundError(f"Missing file: {ITEM_EMB_PATH}")
    if not USER_EMB_PATH.exists():
        raise FileNotFoundError(f"Missing file: {USER_EMB_PATH}")

    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    item_emb = pd.read_parquet(ITEM_EMB_PATH)
    user_emb = pd.read_parquet(USER_EMB_PATH)

    return train_df, item_emb, user_emb


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    denom = a_norm * b_norm
    denom = np.where(denom == 0, 1e-12, denom)

    sim = np.sum(a * b, axis=1) / denom
    return sim


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df, item_emb, user_emb = load_data()

    item_emb_cols = [c for c in item_emb.columns if c.startswith("item_emb_")]
    user_emb_cols = [c for c in user_emb.columns if c.startswith("user_emb_")]

    if not item_emb_cols or not user_emb_cols:
        raise ValueError("Missing embedding columns in item or user embeddings.")

    train_df = train_df.merge(item_emb[["item_id"] + item_emb_cols], on="item_id", how="left")
    train_df = train_df.merge(user_emb[["user_id"] + user_emb_cols], on="user_id", how="left")

    for col in item_emb_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)

    for col in user_emb_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)

    # align user/item vectors by dimension index
    item_matrix = train_df[item_emb_cols].to_numpy(dtype=float)

    aligned_user_cols = []
    for item_col in item_emb_cols:
        idx = item_col.replace("item_emb_", "")
        user_col = f"user_emb_{idx}"
        if user_col in train_df.columns:
            aligned_user_cols.append(user_col)
        else:
            train_df[user_col] = 0.0
            aligned_user_cols.append(user_col)

    user_matrix = train_df[aligned_user_cols].to_numpy(dtype=float)

    train_df["user_item_embedding_cosine"] = cosine_similarity_rows(user_matrix, item_matrix)
    train_df["user_embedding_norm"] = np.linalg.norm(user_matrix, axis=1)
    train_df["item_embedding_norm"] = np.linalg.norm(item_matrix, axis=1)

    # lightweight interaction feature
    train_df["embedding_affinity_bucket"] = pd.cut(
        train_df["user_item_embedding_cosine"],
        bins=[-1.0, 0.2, 0.5, 0.8, 1.0],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True,
    ).astype(str)

    train_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved similarity-enhanced dataset to: {OUTPUT_PATH}")
    print(f"Shape: {train_df.shape}")
    print(train_df[[
        "user_id",
        "item_id",
        "user_item_embedding_cosine",
        "user_embedding_norm",
        "item_embedding_norm",
        "embedding_affinity_bucket",
    ]].head())


if __name__ == "__main__":
    main()