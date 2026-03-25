from pathlib import Path
import random
import uuid
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/silver/impressions_v2.parquet")
OUTPUT_PATH = Path("data/silver/item_catalog_expanded_v2.parquet")

RANDOM_SEED = 42
TARGET_ITEMS = 300


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_base_items() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH).copy()
    df["impression_time"] = pd.to_datetime(df["impression_time"], errors="coerce")

    catalog = (
        df.sort_values("impression_time")
        .drop_duplicates(subset=["item_id"], keep="last")
        .copy()
    )

    defaults = {
        "source": "unknown",
        "source_type": "unknown",
        "category": "unknown",
        "item_age_hours": 24.0,
        "title_length": 50.0,
        "description_length": 150.0,
        "hour_published": 12.0,
        "weekday_published": 2.0,
        "is_hackernews": 0,
        "is_rss": 1,
    }

    for col, default in defaults.items():
        if col not in catalog.columns:
            catalog[col] = default

    return catalog.reset_index(drop=True)


def perturb_numeric(value, scale=0.1, min_value=0.0):
    try:
        value = float(value)
    except Exception:
        value = 0.0

    noise = np.random.normal(loc=0.0, scale=max(1e-6, abs(value) * scale + 1.0))
    out = value + noise
    return max(min_value, out)


def create_variant_row(base_row: pd.Series, variant_idx: int) -> dict:
    row = base_row.to_dict()

    base_item_id = str(row["item_id"])
    row["item_id"] = f"{base_item_id}_aug_{variant_idx}_{uuid.uuid4().hex[:8]}"

    # keep source/category mostly stable but allow some variation
    if np.random.rand() < 0.15:
        row["source"] = random.choice(["hackernews", "bbc_world", "techcrunch", "ars", "venturebeat", "rss_misc"])

    if np.random.rand() < 0.20:
        row["category"] = random.choice(["tech", "world", "business", "science", "politics", "sports"])

    row["source_type"] = "rss" if row["source"] != "hackernews" else "hn"

    row["item_age_hours"] = perturb_numeric(row.get("item_age_hours", 24.0), scale=0.35, min_value=0.0)
    row["title_length"] = perturb_numeric(row.get("title_length", 50.0), scale=0.20, min_value=5.0)
    row["description_length"] = perturb_numeric(row.get("description_length", 150.0), scale=0.25, min_value=20.0)
    row["hour_published"] = int(np.clip(round(perturb_numeric(row.get("hour_published", 12.0), scale=0.5, min_value=0.0)), 0, 23))
    row["weekday_published"] = int(np.clip(round(perturb_numeric(row.get("weekday_published", 2.0), scale=0.5, min_value=0.0)), 0, 6))

    row["is_hackernews"] = int(str(row["source"]) == "hackernews")
    row["is_rss"] = 1 - row["is_hackernews"]

    return row


def main():
    set_seed()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    base_catalog = load_base_items()
    n_base = len(base_catalog)

    if n_base == 0:
        raise ValueError("Base catalog is empty.")

    rows = []
    rows.extend(base_catalog.to_dict(orient="records"))

    needed = max(0, TARGET_ITEMS - n_base)
    for i in range(needed):
        base_row = base_catalog.iloc[i % n_base]
        rows.append(create_variant_row(base_row, i))

    expanded = pd.DataFrame(rows)

    expanded = expanded.drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    # optional synthetic publish timestamp proxy
    expanded["synthetic_popularity_prior"] = np.random.beta(a=2, b=6, size=len(expanded))
    expanded["synthetic_quality_score"] = np.random.beta(a=3, b=3, size=len(expanded))

    expanded.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved expanded catalog to: {OUTPUT_PATH}")
    print(f"Base items: {n_base}")
    print(f"Expanded items: {len(expanded)}")
    print("\nCategory distribution:")
    print(expanded["category"].value_counts(dropna=False).head(10))
    print("\nSource distribution:")
    print(expanded["source"].value_counts(dropna=False).head(10))
    print("\nHead:")
    print(expanded.head())


if __name__ == "__main__":
    main()