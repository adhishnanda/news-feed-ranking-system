from pathlib import Path
import json
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.evaluation.ranking_metrics_v2 import evaluate_ranking


DATA_PATH = Path("data/gold/training_dataset_simulated_v2.parquet")
OUTPUT_PATH = Path("data/gold/metrics_simulated_v2.json")


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def time_split(df):
    df = df.sort_values("timestamp")

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:]
    )


def get_features(df):
    numeric_features = [
        "rank_position",
        "age_hours",
        "title_length",
        "description_length",
        "recent_click_count",
        "recent_impression_count",
        "user_prev_clicks_sim",
        "user_prev_impressions_sim",
        "user_ctr_prior_sim",
        "item_prev_clicks_sim",
        "item_prev_impressions_sim",
        "item_ctr_prior_sim",
        "preferred_category_match",
        "preferred_source_match",
        "recency_decay_sim",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    return numeric_features


def train_model(train_df, features):
    X = train_df[features].fillna(0)
    y = train_df["clicked"]

    model = LogisticRegression(max_iter=3000, random_state=42)
    model.fit(X, y)

    return model


def score(df, model, features):
    X = df[features].fillna(0)
    df = df.copy()
    df["score"] = model.predict_proba(X)[:, 1]
    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, test_df = time_split(df)

    features = get_features(df)

    model = train_model(train_df, features)

    val_scored = score(val_df, model, features)
    test_scored = score(test_df, model, features)

    val_metrics = evaluate_ranking(val_scored, k=5)
    test_metrics = evaluate_ranking(test_scored, k=5)

    results = {
        "dataset_size": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "features": features,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()