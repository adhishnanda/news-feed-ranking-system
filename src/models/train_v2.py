from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    LIGHTGBM_AVAILABLE = False


DATA_PATH = Path("data/gold/train_dataset_v2_embeddings.parquet")
METRICS_PATH = Path("data/gold/metrics_v2.json")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    return df


def time_split(df):
    df = df.sort_values("impression_time").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def get_feature_lists(df):
    numeric_features = [
        "rank_position",
        "hour",
        "day_of_week",
        "is_weekend",
        "item_age_hours",
        "title_length",
        "description_length",
        "hour_published",
        "weekday_published",
        "is_hackernews",
        "is_rss",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
        "user_prev_clicks",
        "user_prev_impressions",
        "user_ctr_prior",
        "item_prev_impressions",
        "item_prev_clicks",
        "item_ctr_prior",
        "preferred_source_match",
        "preferred_category_match",
        "user_item_category_match_v2",
        "user_item_source_match_v2",
        "recency_decay",
        "user_item_embedding_cosine",
        "user_embedding_norm",
        "item_embedding_norm",
    ]

    categorical_features = [
        "source",
        "source_type",
        "category",
        "policy_name",
        "model_version",
        "user_prev_category",
        "user_prev_source",
        "embedding_affinity_bucket",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(model, X, y, model_name):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else None
    acc = accuracy_score(y, pred)

    return {
        "model": model_name,
        "auc": auc,
        "accuracy": acc,
    }


def main():
    df = load_data()
    train_df, val_df, test_df = time_split(df)

    target = "clicked"
    numeric_features, categorical_features = get_feature_lists(df)
    feature_cols = numeric_features + categorical_features

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_val = val_df[feature_cols]
    y_val = val_df[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    lr_pipeline.fit(X_train, y_train)
    lr_val_metrics = evaluate_model(lr_pipeline, X_val, y_val, "LogisticRegression_val")
    lr_test_metrics = evaluate_model(lr_pipeline, X_test, y_test, "LogisticRegression_test")

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    if LIGHTGBM_AVAILABLE:
        second_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        second_name_val = "LightGBM_val"
        second_name_test = "LightGBM_test"
    else:
        second_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        )
        second_name_val = "RandomForest_val"
        second_name_test = "RandomForest_test"

    second_model.fit(X_train_t, y_train)
    second_val_metrics = evaluate_model(second_model, X_val_t, y_val, second_name_val)
    second_test_metrics = evaluate_model(second_model, X_test_t, y_test, second_name_test)

    results = {
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
        },
        "features": {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        "metrics": [
            lr_val_metrics,
            lr_test_metrics,
            second_val_metrics,
            second_test_metrics,
        ],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    print(f"\nSaved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()