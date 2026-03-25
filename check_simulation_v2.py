from pathlib import Path
import pandas as pd


EVENTS_PATH = Path("data/logs/simulated_events_v2.parquet")
TRAIN_PATH = Path("data/gold/training_dataset_simulated_v2.parquet")


def show_file(path: Path, name: str):
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

    if "user_id" in df.columns:
        print("\nUnique users:", df["user_id"].nunique())
    if "session_id" in df.columns:
        print("Unique sessions:", df["session_id"].nunique())
    if "item_id" in df.columns:
        print("Unique items:", df["item_id"].nunique())
    if "clicked" in df.columns:
        print("\nClicked distribution:")
        print(df["clicked"].value_counts(dropna=False))


def main():
    show_file(EVENTS_PATH, "Simulated Events")
    show_file(TRAIN_PATH, "Simulated Training Dataset")


if __name__ == "__main__":
    main()