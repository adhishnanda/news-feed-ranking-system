from __future__ import annotations
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from src.utils.config import load_yaml
from src.models.baseline_ranker import build_logistic_model
from src.models.lgbm_ranker import build_lgbm_pipeline


def load_dataset():
    cfg = load_yaml("configs/config.yaml")
    path = f"{cfg['paths']['gold_dir']}/training_dataset.parquet"
    return pd.read_parquet(path)


def evaluate_binary_model(model, X_test, y_test) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_pred_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    return metrics


def main():
    cfg = load_yaml("configs/config.yaml")
    model_cfg = load_yaml("configs/model.yaml")

    model_dir = cfg["paths"]["model_dir"]
    default_model_name = model_cfg["training"].get("model_name", "lightgbm")
    lgbm_params = model_cfg["models"]["lightgbm"]["params"]

    df = load_dataset()

    if df.empty:
        print("Dataset is empty. Cannot train.")
        return

    target = "clicked"

    drop_cols = [
        "event_id",
        "timestamp",
        "user_id",
        "session_id",
        "item_id",
        "model_version",
        "policy_name",
    ]

    X = df.drop(columns=drop_cols + [target], errors="ignore")
    y = df[target]

    if y.nunique() < 2:
        print("Not enough label diversity to train model.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    logistic_model = build_logistic_model()
    logistic_model.fit(X_train, y_train)
    logistic_metrics = evaluate_binary_model(logistic_model, X_test, y_test)

    print("\nLogistic Regression metrics:")
    for k, v in logistic_metrics.items():
        print(f"  {k}: {v:.4f}")

    joblib.dump(logistic_model, f"{model_dir}/logistic_model.joblib")

    lgbm_model = build_lgbm_pipeline(lgbm_params)
    lgbm_model.fit(X_train, y_train)
    lgbm_metrics = evaluate_binary_model(lgbm_model, X_test, y_test)

    print("\nLightGBM metrics:")
    for k, v in lgbm_metrics.items():
        print(f"  {k}: {v:.4f}")

    joblib.dump(lgbm_model, f"{model_dir}/lightgbm_model.joblib")

    all_metrics = {
        "logistic_regression": logistic_metrics,
        "lightgbm": lgbm_metrics,
        "default_serving_model": default_model_name,
    }

    with open(f"{model_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nSaved:")
    print(f"- {model_dir}/logistic_model.joblib")
    print(f"- {model_dir}/lightgbm_model.joblib")
    print(f"- {model_dir}/metrics.json")


if __name__ == "__main__":
    main()