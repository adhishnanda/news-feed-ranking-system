from pathlib import Path
import json
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.evaluation.ranking_metrics_v2 import evaluate_ranking
from src.evaluation.feed_quality_metrics_v2 import evaluate_feed_quality


DATA_PATH = Path("data/gold/training_dataset_simulated_v3.parquet")
OUTPUT_PATH = Path("data/gold/metrics_simulated_v3.json")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def time_split(df):
    df = df.sort_values("timestamp")
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def get_features(df):
    numeric_features = [
        "rank_position",
        "age_hours",
        "title_length",
        "description_length",
        "recent_click_count",
        "recent_impression_count",
        "recent_save_count",
        "recent_hide_count",
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
    return [c for c in numeric_features if c in df.columns]


def build_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )


def score(df, model, features):
    out = df.copy()
    out["score"] = model.predict_proba(out[features].fillna(0))[:, 1]
    return out


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train_df, val_df, test_df = time_split(df)

    features = get_features(df)
    model = build_model()
    model.fit(train_df[features].fillna(0), train_df["clicked"])

    val_scored = score(val_df, model, features)
    test_scored = score(test_df, model, features)

    val_rank_metrics = evaluate_ranking(val_scored, k=5)
    test_rank_metrics = evaluate_ranking(test_scored, k=5)

    val_feed_metrics = evaluate_feed_quality(val_scored, k=5)
    test_feed_metrics = evaluate_feed_quality(test_scored, k=5)

    results = {
        "dataset_size": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "features": features,
        "val_ranking_metrics": val_rank_metrics,
        "test_ranking_metrics": test_rank_metrics,
        "val_feed_quality_metrics": val_feed_metrics,
        "test_feed_quality_metrics": test_feed_metrics,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()