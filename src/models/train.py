from __future__ import annotations
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.utils.config import load_yaml
from src.models.baseline_ranker import build_logistic_model


def load_dataset():
    cfg = load_yaml("configs/config.yaml")
    path = f"{cfg['paths']['gold_dir']}/training_dataset.parquet"
    return pd.read_parquet(path)


def main():
    cfg = load_yaml("configs/config.yaml")
    model_dir = cfg["paths"]["model_dir"]

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

    model = build_logistic_model()

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"AUC: {auc:.4f}")

    joblib.dump(model, f"{model_dir}/logistic_model.joblib")

    print("Model saved.")


if __name__ == "__main__":
    main()