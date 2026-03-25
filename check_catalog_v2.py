from pathlib import Path
import pandas as pd


CATALOG_PATH = Path("data/silver/item_catalog_expanded_v2.parquet")
EVENTS_PATH = Path("data/logs/simulated_events_v3.parquet")
TRAIN_PATH = Path("data/gold/training_dataset_simulated_v3.parquet")


def show(path: Path, name: str):
    if not path.exists():
        print(f"{name}: missing -> {path}")
        return

    df = pd.read_parquet(path)
    print(f"\n{name}")
    print("=" * len(name))
    print("Path:", path)
    print("Shape:", df.shape)
    print("Head:")
    print(df.head())

    if "item_id" in df.columns:
        print("Unique items:", df["item_id"].nunique())
    if "user_id" in df.columns:
        print("Unique users:", df["user_id"].nunique())
    if "session_id" in df.columns:
        print("Unique sessions:", df["session_id"].nunique())
    if "category" in df.columns:
        print("\nTop categories:")
        print(df["category"].value_counts(dropna=False).head(10))
    if "source" in df.columns:
        print("\nTop sources:")
        print(df["source"].value_counts(dropna=False).head(10))


def main():
    show(CATALOG_PATH, "Expanded Catalog V2")
    show(EVENTS_PATH, "Simulated Events V3")
    show(TRAIN_PATH, "Training Dataset Simulated V3")


if __name__ == "__main__":
    main()