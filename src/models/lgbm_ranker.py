from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_lgbm_pipeline(params: dict | None = None):
    numeric_features = [
        "age_hours",
        "title_length",
        "description_length",
        "hour_published",
        "weekday_published",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
        "hour_of_day",
        "weekday",
        "is_weekend",
        "preferred_source_match",
        "preferred_category_match",
    ]

    categorical_features = [
        "source",
        "source_type",
        "category",
        "preferred_source",
        "preferred_category",
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LGBMClassifier(**(params or {}))

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline