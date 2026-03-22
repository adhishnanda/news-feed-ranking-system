from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessing_pipeline():
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

    return preprocessor


def build_logistic_model():
    preprocessor = build_preprocessing_pipeline()

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline