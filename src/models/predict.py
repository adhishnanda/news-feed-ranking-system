from __future__ import annotations
import joblib
import pandas as pd

from src.utils.config import load_yaml


def load_model(model_name: str | None = None):
    cfg = load_yaml("configs/config.yaml")
    model_cfg = load_yaml("configs/model.yaml")

    if model_name is None:
        model_name = model_cfg["training"].get("model_name", "lightgbm")

    if model_name == "logistic":
        model_path = f"{cfg['paths']['model_dir']}/logistic_model.joblib"
    elif model_name == "lightgbm":
        model_path = f"{cfg['paths']['model_dir']}/lightgbm_model.joblib"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = joblib.load(model_path)
    return model, model_name


def predict_scores(model, features_df: pd.DataFrame) -> pd.DataFrame:
    scored = features_df.copy()
    scored["model_score"] = model.predict_proba(features_df)[:, 1]
    return scored