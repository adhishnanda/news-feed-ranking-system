from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


INPUT_PATH = Path("data/silver/impressions_v2.parquet")
OUTPUT_PATH = Path("data/silver/item_embeddings_v2.parquet")
VECTORIZER_PATH = Path("models_artifacts/tfidf_vectorizer_v2.joblib")
SVD_PATH = Path("models_artifacts/tfidf_svd_v2.joblib")


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Since your current dataset does not have raw title/description text,
    # we build a lightweight pseudo-text representation from metadata.
    text_parts = []

    for col in ["source", "source_type", "category"]:
        if col in df.columns:
            text_parts.append(df[col].fillna("unknown").astype(str))

    if "title_length" in df.columns:
        text_parts.append(("titlelen_" + df["title_length"].fillna(0).astype(int).astype(str)))

    if "description_length" in df.columns:
        text_parts.append(("desclen_" + df["description_length"].fillna(0).astype(int).astype(str)))

    if "is_hackernews" in df.columns:
        text_parts.append(("is_hn_" + df["is_hackernews"].fillna(0).astype(int).astype(str)))

    if "is_rss" in df.columns:
        text_parts.append(("is_rss_" + df["is_rss"].fillna(0).astype(int).astype(str)))

    if not text_parts:
        df["embedding_text"] = "unknown"
    else:
        combined = text_parts[0].copy()
        for series in text_parts[1:]:
            combined = combined + " " + series
        df["embedding_text"] = combined

    return df


def build_item_level_table(df: pd.DataFrame) -> pd.DataFrame:
    # one row per item, keeping latest metadata snapshot
    item_df = (
        df.sort_values("impression_time")
        .drop_duplicates(subset=["item_id"], keep="last")
        .copy()
    )
    return item_df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVD_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df["impression_time"] = pd.to_datetime(df["impression_time"])

    df = build_text_column(df)
    item_df = build_item_level_table(df)

    texts = item_df["embedding_text"].fillna("unknown").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=1,
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # keep dimensions small and safe relative to data size
    max_possible_components = max(1, min(16, X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1))
    if max_possible_components < 1:
        max_possible_components = 1

    svd = TruncatedSVD(n_components=max_possible_components, random_state=42)
    X_emb = svd.fit_transform(X_tfidf)

    emb_cols = [f"item_emb_{i}" for i in range(X_emb.shape[1])]
    emb_df = pd.DataFrame(X_emb, columns=emb_cols)

    output_df = pd.concat(
        [
            item_df[["item_id", "source", "source_type", "category"]].reset_index(drop=True),
            emb_df.reset_index(drop=True),
        ],
        axis=1,
    )

    output_df.to_parquet(OUTPUT_PATH, index=False)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)

    print(f"Saved item embeddings to: {OUTPUT_PATH}")
    print(f"Shape: {output_df.shape}")
    print(f"Embedding dimensions: {len(emb_cols)}")
    print(output_df.head())


if __name__ == "__main__":
    main()