from pathlib import Path
import pandas as pd


ITEM_PATH = Path("data/silver/item_embeddings_v2.parquet")
USER_PATH = Path("data/silver/user_embeddings_v2.parquet")
TRAIN_PATH = Path("data/gold/train_dataset_v2_embeddings.parquet")


def show_info(path: Path, name: str):
    if not path.exists():
        print(f"{name}: missing -> {path}")
        return

    df = pd.read_parquet(path)
    print(f"\n{name}")
    print("-" * len(name))
    print("Path:", path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist()[:20], "..." if len(df.columns) > 20 else "")
    print(df.head())


def main():
    show_info(ITEM_PATH, "Item Embeddings")
    show_info(USER_PATH, "User Embeddings")
    show_info(TRAIN_PATH, "Similarity-Enhanced Training Dataset")


if __name__ == "__main__":
    main()