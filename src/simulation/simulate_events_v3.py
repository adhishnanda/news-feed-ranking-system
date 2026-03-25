from pathlib import Path
import uuid
import random
from datetime import timedelta

import numpy as np
import pandas as pd


CATALOG_PATH = Path("data/silver/item_catalog_expanded_v2.parquet")
OUTPUT_EVENTS_PATH = Path("data/logs/simulated_events_v3.parquet")
OUTPUT_TRAIN_PATH = Path("data/gold/training_dataset_simulated_v3.parquet")

RANDOM_SEED = 42
N_USERS = 100
SESSIONS_PER_USER = 25
CANDIDATES_PER_SESSION = 12


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CATALOG_PATH}")
    return pd.read_parquet(CATALOG_PATH).copy()


def build_user_profiles(catalog: pd.DataFrame, n_users: int) -> pd.DataFrame:
    categories = sorted(catalog["category"].fillna("unknown").astype(str).unique().tolist())
    sources = sorted(catalog["source"].fillna("unknown").astype(str).unique().tolist())

    rows = []
    for i in range(1, n_users + 1):
        rows.append(
            {
                "user_id": f"user_sim_v3_{i}",
                "preferred_category": random.choice(categories),
                "secondary_category": random.choice(categories),
                "preferred_source": random.choice(sources),
                "activity_level": random.choice([0.8, 1.0, 1.2]),
                "novelty_preference": random.choice([0.7, 1.0, 1.3]),
                "quality_preference": random.choice([0.8, 1.0, 1.2]),
            }
        )

    return pd.DataFrame(rows)


def sample_candidates(catalog: pd.DataFrame, user_profile: dict, k: int) -> pd.DataFrame:
    working = catalog.copy()

    category = working["category"].astype(str)
    source = working["source"].astype(str)

    category_match = (category == str(user_profile["preferred_category"])).astype(float)
    secondary_match = (category == str(user_profile["secondary_category"])).astype(float)
    source_match = (source == str(user_profile["preferred_source"])).astype(float)

    recency = 1.0 / (1.0 + pd.to_numeric(working.get("item_age_hours", 24.0), errors="coerce").fillna(24.0))
    quality = pd.to_numeric(working.get("synthetic_quality_score", 0.5), errors="coerce").fillna(0.5)
    popularity = pd.to_numeric(working.get("synthetic_popularity_prior", 0.2), errors="coerce").fillna(0.2)

    weights = (
        1.0
        + 1.8 * category_match
        + 0.8 * secondary_match
        + 1.0 * source_match
        + user_profile["novelty_preference"] * 0.6 * recency
        + user_profile["quality_preference"] * 0.5 * quality
        + 0.4 * popularity
    )

    weights = np.clip(weights, 1e-6, None)
    probs = weights / weights.sum()

    sample_size = min(k, len(working))
    chosen_idx = np.random.choice(working.index, size=sample_size, replace=False, p=probs)
    chosen = working.loc[chosen_idx].copy()

    chosen = chosen.sample(frac=1.0, random_state=np.random.randint(0, 1_000_000)).reset_index(drop=True)
    chosen["rank_position"] = np.arange(1, len(chosen) + 1)

    return chosen


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_session_events(session_time: pd.Timestamp, user_profile: dict, shown_items: pd.DataFrame) -> list[dict]:
    events = []

    recent_impression_count = 0
    recent_click_count = 0
    recent_save_count = 0
    recent_hide_count = 0

    for _, row in shown_items.iterrows():
        category = str(row.get("category", "unknown"))
        source = str(row.get("source", "unknown"))
        rank_position = int(row["rank_position"])
        age_hours = float(pd.to_numeric(row.get("item_age_hours", 24.0), errors="coerce"))
        quality = float(pd.to_numeric(row.get("synthetic_quality_score", 0.5), errors="coerce"))
        popularity = float(pd.to_numeric(row.get("synthetic_popularity_prior", 0.2), errors="coerce"))

        preferred_match = 1.0 if category == str(user_profile["preferred_category"]) else 0.0
        secondary_match = 1.0 if category == str(user_profile["secondary_category"]) else 0.0
        source_match = 1.0 if source == str(user_profile["preferred_source"]) else 0.0
        position_bias = 1.0 / rank_position
        recency = 1.0 / (1.0 + max(age_hours, 0.0))

        logit = (
            -2.6
            + 1.2 * preferred_match
            + 0.5 * secondary_match
            + 0.7 * source_match
            + 0.9 * recency
            + 1.1 * position_bias
            + 0.5 * quality
            + 0.3 * popularity
            + 0.2 * user_profile["activity_level"]
        )

        click_prob = float(sigmoid(logit))
        clicked = int(np.random.rand() < click_prob)

        save_prob = min(0.6, 0.05 + 0.30 * clicked + 0.20 * quality)
        saved = int(np.random.rand() < save_prob)

        hide_prob = min(0.35, 0.03 + 0.12 * (1 - preferred_match) + 0.08 * (1 - source_match))
        hidden = int(np.random.rand() < hide_prob)

        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": session_time + timedelta(seconds=int(rank_position * 4)),
                "user_id": user_profile["user_id"],
                "session_id": f"{user_profile['user_id']}_session_{session_time.strftime('%Y%m%d%H%M%S')}",
                "item_id": str(row["item_id"]),
                "rank_position": rank_position,
                "model_version": "sim_v3",
                "policy_name": "simulator_policy_v3",
                "clicked": clicked,
                "saved": saved,
                "hidden": hidden,
                "source": source,
                "source_type": str(row.get("source_type", "unknown")),
                "category": category,
                "age_hours": age_hours,
                "title_length": float(pd.to_numeric(row.get("title_length", 50.0), errors="coerce")),
                "description_length": float(pd.to_numeric(row.get("description_length", 150.0), errors="coerce")),
                "hour_published": float(pd.to_numeric(row.get("hour_published", 12.0), errors="coerce")),
                "weekday_published": float(pd.to_numeric(row.get("weekday_published", 2.0), errors="coerce")),
                "is_hackernews": int(pd.to_numeric(row.get("is_hackernews", 0), errors="coerce")),
                "is_rss": int(pd.to_numeric(row.get("is_rss", 1), errors="coerce")),
                "recent_impression_count": recent_impression_count,
                "recent_click_count": recent_click_count,
                "recent_save_count": recent_save_count,
                "recent_hide_count": recent_hide_count,
                "preferred_source": user_profile["preferred_source"],
                "preferred_category": user_profile["preferred_category"],
                "hour_of_day": session_time.hour,
                "weekday": session_time.dayofweek,
                "is_weekend": int(session_time.dayofweek in [5, 6]),
                "preferred_source_match": int(source == str(user_profile["preferred_source"])),
                "preferred_category_match": int(category == str(user_profile["preferred_category"])),
            }
        )

        recent_impression_count += 1
        recent_click_count += clicked
        recent_save_count += saved
        recent_hide_count += hidden

    return events


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df["user_prev_clicks_sim"] = (
        df.groupby("user_id")["clicked"].transform(lambda s: s.cumsum().shift(1)).fillna(0)
    )
    df["user_prev_impressions_sim"] = df.groupby("user_id").cumcount()
    df["user_ctr_prior_sim"] = (
        df["user_prev_clicks_sim"] / df["user_prev_impressions_sim"].replace(0, np.nan)
    ).fillna(0.0)

    item_df = df.copy().sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    item_df["item_prev_clicks_sim"] = (
        item_df.groupby("item_id")["clicked"].transform(lambda s: s.cumsum().shift(1)).fillna(0)
    )
    item_df["item_prev_impressions_sim"] = item_df.groupby("item_id").cumcount()
    item_df["item_ctr_prior_sim"] = (
        item_df["item_prev_clicks_sim"] / item_df["item_prev_impressions_sim"].replace(0, np.nan)
    ).fillna(0.0)

    df = df.merge(
        item_df[["event_id", "item_prev_clicks_sim", "item_prev_impressions_sim", "item_ctr_prior_sim"]],
        on="event_id",
        how="left",
    )

    df["recency_decay_sim"] = 1.0 / (1.0 + pd.to_numeric(df["age_hours"], errors="coerce").fillna(0.0))
    return df


def main():
    set_seed()
    OUTPUT_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    users = build_user_profiles(catalog, N_USERS)

    base_time = pd.Timestamp("2026-01-01 08:00:00")
    all_events = []

    for _, user_row in users.iterrows():
        user_profile = user_row.to_dict()

        for _ in range(SESSIONS_PER_USER):
            session_time = base_time + timedelta(
                days=np.random.randint(0, 90),
                hours=np.random.randint(0, 14),
                minutes=np.random.randint(0, 60),
            )

            shown = sample_candidates(catalog, user_profile, CANDIDATES_PER_SESSION)
            events = simulate_session_events(session_time, user_profile, shown)
            all_events.extend(events)

    events_df = pd.DataFrame(all_events).sort_values("timestamp").reset_index(drop=True)
    events_df.to_parquet(OUTPUT_EVENTS_PATH, index=False)

    train_df = build_training_features(events_df)
    train_df.to_parquet(OUTPUT_TRAIN_PATH, index=False)

    print(f"Saved events to: {OUTPUT_EVENTS_PATH}")
    print(f"Saved training dataset to: {OUTPUT_TRAIN_PATH}")
    print(f"\nShape events: {events_df.shape}")
    print(f"Shape training: {train_df.shape}")
    print(f"Users: {events_df['user_id'].nunique()}")
    print(f"Sessions: {events_df['session_id'].nunique()}")
    print(f"Items: {events_df['item_id'].nunique()}")
    print("\nClicked distribution:")
    print(events_df["clicked"].value_counts(dropna=False))
    print("\nCategory distribution:")
    print(events_df["category"].value_counts(dropna=False).head(10))
    print("\nSource distribution:")
    print(events_df["source"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()