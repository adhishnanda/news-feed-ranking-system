from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/gold/train_dataset_v2.parquet")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head())

    print("\nNull counts:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    if "clicked" in df.columns:
        print("\nClicked distribution:")
        print(df["clicked"].value_counts(dropna=False))


if __name__ == "__main__":
    main()