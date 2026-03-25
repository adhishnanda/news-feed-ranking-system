from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/gold/candidates_v2.parquet")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head())

    if "user_id" in df.columns:
        print("\nCandidates per user:")
        print(df.groupby("user_id")["item_id"].count())

    if "category" in df.columns:
        print("\nCategory distribution:")
        print(df["category"].value_counts(dropna=False).head(10))

    if "source" in df.columns:
        print("\nSource distribution:")
        print(df["source"].value_counts(dropna=False).head(10))

    if "retrieval_strategy_count" in df.columns:
        print("\nStrategy count distribution:")
        print(df["retrieval_strategy_count"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()