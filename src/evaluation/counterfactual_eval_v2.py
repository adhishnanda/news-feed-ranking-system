from pathlib import Path
import json
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/gold/simulated_logged_policy_v2.parquet")
OUTPUT_PATH = Path("data/gold/counterfactual_metrics_v2.json")


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH).copy()
    return df


def ips(rewards: np.ndarray, logged_propensity: np.ndarray, target_propensity: np.ndarray) -> float:
    weights = target_propensity / np.clip(logged_propensity, 1e-12, None)
    return float(np.mean(weights * rewards))


def snips(rewards: np.ndarray, logged_propensity: np.ndarray, target_propensity: np.ndarray) -> float:
    weights = target_propensity / np.clip(logged_propensity, 1e-12, None)
    denom = np.sum(weights)
    if denom == 0:
        return 0.0
    return float(np.sum(weights * rewards) / denom)


def clipped_ips(
    rewards: np.ndarray,
    logged_propensity: np.ndarray,
    target_propensity: np.ndarray,
    clip_value: float = 10.0,
) -> float:
    weights = target_propensity / np.clip(logged_propensity, 1e-12, None)
    weights = np.clip(weights, 0.0, clip_value)
    return float(np.mean(weights * rewards))


def direct_method_baseline(rewards: np.ndarray) -> float:
    return float(np.mean(rewards))


def evaluate(df: pd.DataFrame) -> dict:
    rewards = pd.to_numeric(df["clicked"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    logged_prop = pd.to_numeric(df["logged_propensity"], errors="coerce").fillna(1e-12).to_numpy(dtype=float)
    target_prop = pd.to_numeric(df["target_propensity"], errors="coerce").fillna(1e-12).to_numpy(dtype=float)

    results = {
        "n_rows": int(len(df)),
        "observed_ctr": direct_method_baseline(rewards),
        "ips_estimate": ips(rewards, logged_prop, target_prop),
        "snips_estimate": snips(rewards, logged_prop, target_prop),
        "clipped_ips_estimate": clipped_ips(rewards, logged_prop, target_prop, clip_value=10.0),
        "mean_logged_propensity": float(np.mean(logged_prop)),
        "mean_target_propensity": float(np.mean(target_prop)),
    }

    return results


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    results = evaluate(df)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()