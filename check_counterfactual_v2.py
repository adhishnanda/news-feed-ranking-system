from pathlib import Path
import json
import pandas as pd


DATA_PATH = Path("data/gold/simulated_logged_policy_v2.parquet")
METRICS_PATH = Path("data/gold/counterfactual_metrics_v2.json")


def main():
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        print("Logged policy dataset")
        print("===================")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist()[-8:])
        print(df[[
            "session_id",
            "item_id",
            "clicked",
            "logged_propensity",
            "target_propensity",
        ]].head())
    else:
        print(f"Missing dataset: {DATA_PATH}")

    print()

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        print("Counterfactual metrics")
        print("======================")
        print(json.dumps(metrics, indent=2))
    else:
        print(f"Missing metrics file: {METRICS_PATH}")


if __name__ == "__main__":
    main()