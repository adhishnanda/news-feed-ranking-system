from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/gold/train_dataset_v2.parquet")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df["impression_time"] = pd.to_datetime(df["impression_time"])

    if "user_feature_time" in df.columns:
        df["user_feature_time"] = pd.to_datetime(df["user_feature_time"])
        valid_user = df["user_feature_time"].isna() | (df["user_feature_time"] <= df["impression_time"])
        print("No future user features:", bool(valid_user.all()))
    else:
        print("Column user_feature_time not found")

    if "item_feature_time" in df.columns:
        df["item_feature_time"] = pd.to_datetime(df["item_feature_time"])
        valid_item = df["item_feature_time"].isna() | (df["item_feature_time"] <= df["impression_time"])
        print("No future item features:", bool(valid_item.all()))
    else:
        print("Column item_feature_time not found")

    if "item_age_hours" in df.columns:
        print("Non-negative item age:", bool((df["item_age_hours"] >= 0).all()))
    else:
        print("Column item_age_hours not found")

    if "clicked" in df.columns:
        print("Clicked distribution:")
        print(df["clicked"].value_counts(dropna=False))
    else:
        print("Column clicked not found")


if __name__ == "__main__":
    main()