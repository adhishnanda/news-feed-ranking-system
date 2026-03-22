from __future__ import annotations
import joblib
import pandas as pd

from src.utils.config import load_yaml


def load_model():
    cfg = load_yaml("configs/config.yaml")
    model_path = f"{cfg['paths']['model_dir']}/logistic_model.joblib"
    model = joblib.load(model_path)
    return model


def predict_scores(model, features_df: pd.DataFrame) -> pd.DataFrame:
    scored = features_df.copy()
    scored["model_score"] = model.predict_proba(features_df)[:, 1]
    return scored