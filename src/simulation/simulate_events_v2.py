from pathlib import Path
import uuid
import random
from datetime import timedelta

import numpy as np
import pandas as pd


INPUT_ITEMS_PATH = Path("data/silver/impressions_v2.parquet")
OUTPUT_EVENTS_PATH = Path("data/logs/simulated_events_v2.parquet")


RANDOM_SEED = 42
N_USERS = 50
SESSIONS_PER_USER = 20
CANDIDATES_PER_SESSION = 10


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_items() -> pd.DataFrame:
    if not INPUT_ITEMS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_ITEMS_PATH}")

    df = pd.read_parquet(INPUT_ITEMS_PATH).copy()
    df["impression_time"] = pd.to_datetime(df["impression_time"], errors="coerce")
    return df


def build_item_catalog(df: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        df.sort_values("impression_time")
        .drop_duplicates(subset=["item_id"], keep="last")
        .copy()
    )

    required_defaults = {
        "source": "unknown",
        "source_type": "unknown",
        "category": "unknown",
        "item_age_hours": 0.0,
        "title_length": 0.0,
        "description_length": 0.0,
        "hour_published": 0.0,
        "weekday_published": 0.0,
        "is_hackernews": 0,
        "is_rss": 0,
    }

    for col, default in required_defaults.items():
        if col not in catalog.columns:
            catalog[col] = default

    return catalog.reset_index(drop=True)


def build_user_profiles(catalog: pd.DataFrame, n_users: int) -> pd.DataFrame:
    categories = sorted(catalog["category"].fillna("unknown").astype(str).unique().tolist())
    sources = sorted(catalog["source"].fillna("unknown").astype(str).unique().tolist())

    rows = []
    for i in range(1, n_users + 1):
        user_id = f"user_sim_{i}"

        preferred_category = random.choice(categories) if categories else "unknown"
        preferred_source = random.choice(sources) if sources else "unknown"

        rows.append(
            {
                "user_id": user_id,
                "preferred_category": preferred_category,
                "preferred_source": preferred_source,
                "activity_level": random.choice([0.8, 1.0, 1.2]),
                "novelty_preference": random.choice([0.8, 1.0, 1.2]),
            }
        )

    return pd.DataFrame(rows)


def sample_candidates(catalog: pd.DataFrame, user_profile: dict, k: int) -> pd.DataFrame:
    working = catalog.copy()

    # heuristic sampling weights
    weights = np.ones(len(working), dtype=float)

    category_match = (working["category"].astype(str) == str(user_profile["preferred_category"])).astype(float)
    source_match = (working["source"].astype(str) == str(user_profile["preferred_source"])).astype(float)

    weights += 1.5 * category_match
    weights += 1.0 * source_match

    # fresher items slightly more likely
    age = pd.to_numeric(working["item_age_hours"], errors="coerce").fillna(0.0)
    recency = 1.0 / (1.0 + age)
    weights += 0.5 * recency

    weights = np.clip(weights, 1e-6, None)
    probs = weights / weights.sum()

    sample_size = min(k, len(working))
    chosen_idx = np.random.choice(working.index, size=sample_size, replace=False, p=probs)
    chosen = working.loc[chosen_idx].copy()

    # randomize displayed order
    chosen = chosen.sample(frac=1.0, random_state=np.random.randint(0, 1_000_000)).reset_index(drop=True)
    chosen["rank_position"] = np.arange(1, len(chosen) + 1)

    return chosen


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_session_events(
    session_time: pd.Timestamp,
    user_profile: dict,
    shown_items: pd.DataFrame,
) -> list[dict]:
    events = []

    recent_impression_count = 0
    recent_click_count = 0
    recent_save_count = 0
    recent_hide_count = 0

    for _, row in shown_items.iterrows():
        item_id = row["item_id"]
        category = str(row.get("category", "unknown"))
        source = str(row.get("source", "unknown"))
        rank_position = int(row["rank_position"])

        age_hours = float(pd.to_numeric(row.get("item_age_hours", 0.0), errors="coerce"))
        recency = 1.0 / (1.0 + max(age_hours, 0.0))

        category_match = 1.0 if category == str(user_profile["preferred_category"]) else 0.0
        source_match = 1.0 if source == str(user_profile["preferred_source"]) else 0.0
        position_bias = 1.0 / rank_position

        # click probability
        logit = (
            -2.2
            + 1.0 * category_match
            + 0.7 * source_match
            + 1.2 * recency
            + 1.1 * position_bias
            + 0.3 * user_profile["activity_level"]
        )
        click_prob = float(sigmoid(logit))
        clicked = int(np.random.rand() < click_prob)

        # save probability conditioned on click
        save_prob = min(0.7, 0.15 + 0.35 * clicked + 0.15 * category_match)
        saved = int(np.random.rand() < save_prob)

        # hide probability higher when mismatch
        hide_prob = min(0.5, 0.05 + 0.15 * (1 - category_match) + 0.10 * (1 - source_match))
        hidden = int(np.random.rand() < hide_prob)

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": session_time + timedelta(seconds=int(rank_position * 3)),
            "user_id": user_profile["user_id"],
            "session_id": f"{user_profile['user_id']}_session_{session_time.strftime('%Y%m%d%H%M%S')}",
            "item_id": str(item_id),
            "rank_position": rank_position,
            "model_version": "sim_v2",
            "policy_name": "simulator_policy_v2",
            "clicked": clicked,
            "saved": saved,
            "hidden": hidden,
            "source": source,
            "source_type": str(row.get("source_type", "unknown")),
            "category": category,
            "age_hours": age_hours,
            "title_length": float(pd.to_numeric(row.get("title_length", 0.0), errors="coerce")),
            "description_length": float(pd.to_numeric(row.get("description_length", 0.0), errors="coerce")),
            "hour_published": float(pd.to_numeric(row.get("hour_published", 0.0), errors="coerce")),
            "weekday_published": float(pd.to_numeric(row.get("weekday_published", 0.0), errors="coerce")),
            "is_hackernews": int(pd.to_numeric(row.get("is_hackernews", 0), errors="coerce")),
            "is_rss": int(pd.to_numeric(row.get("is_rss", 0), errors="coerce")),
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

        recent_impression_count += 1
        recent_click_count += clicked
        recent_save_count += saved
        recent_hide_count += hidden

        events.append(event)

    return events


def main():
    set_seed()
    OUTPUT_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = load_items()
    catalog = build_item_catalog(raw_df)
    users = build_user_profiles(catalog, N_USERS)

    base_time = pd.Timestamp("2026-01-01 08:00:00")
    all_events = []

    for _, user_row in users.iterrows():
        user_profile = user_row.to_dict()

        for session_idx in range(SESSIONS_PER_USER):
            session_time = base_time + timedelta(
                days=np.random.randint(0, 60),
                hours=np.random.randint(0, 14),
                minutes=np.random.randint(0, 60),
            )

            shown_items = sample_candidates(catalog, user_profile, CANDIDATES_PER_SESSION)
            session_events = simulate_session_events(session_time, user_profile, shown_items)
            all_events.extend(session_events)

    sim_df = pd.DataFrame(all_events).sort_values("timestamp").reset_index(drop=True)
    sim_df.to_parquet(OUTPUT_EVENTS_PATH, index=False)

    print(f"Saved simulated events to: {OUTPUT_EVENTS_PATH}")
    print(f"Shape: {sim_df.shape}")
    print("\nUsers:", sim_df['user_id'].nunique())
    print("Sessions:", sim_df['session_id'].nunique())
    print("Items:", sim_df['item_id'].nunique())
    print("\nClicked distribution:")
    print(sim_df["clicked"].value_counts(dropna=False))
    print("\nHead:")
    print(sim_df.head())


if __name__ == "__main__":
    main()