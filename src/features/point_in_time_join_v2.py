from pathlib import Path
import pandas as pd


IMPRESSIONS_PATH = Path("data/silver/impressions_v2.parquet")
USER_FEATURES_PATH = Path("data/silver/user_features_v2.parquet")
ITEM_FEATURES_PATH = Path("data/silver/item_features_v2.parquet")
OUTPUT_PATH = Path("data/gold/train_dataset_v2.parquet")


def load_data():
    impressions = pd.read_parquet(IMPRESSIONS_PATH)
    user_features = pd.read_parquet(USER_FEATURES_PATH)
    item_features = pd.read_parquet(ITEM_FEATURES_PATH)

    impressions["impression_time"] = pd.to_datetime(impressions["impression_time"])
    user_features["feature_timestamp"] = pd.to_datetime(user_features["feature_timestamp"])
    item_features["feature_timestamp"] = pd.to_datetime(item_features["feature_timestamp"])

    return impressions, user_features, item_features


def asof_join_by_group(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    group_col: str,
    left_time_col: str,
    right_time_col: str,
    right_cols: list,
    right_time_output_col: str,
) -> pd.DataFrame:
    """
    Safe point-in-time join by processing each group separately.
    This avoids pandas merge_asof global sorting issues.
    """
    output_parts = []

    left_df = left_df.copy()
    right_df = right_df.copy()

    # Ensure required columns exist
    needed_right_cols = [group_col, right_time_col] + right_cols
    right_df = right_df[needed_right_cols].copy()

    for group_value, left_group in left_df.groupby(group_col, sort=False):
        left_group = left_group.sort_values(left_time_col).reset_index(drop=True)

        right_group = right_df[right_df[group_col] == group_value].copy()
        right_group = right_group.sort_values(right_time_col).reset_index(drop=True)

        if right_group.empty:
            # Add empty columns if no matching right-side group exists
            for col in right_cols:
                left_group[col] = pd.NA
            left_group[right_time_output_col] = pd.NaT
            output_parts.append(left_group)
            continue

        merged = pd.merge_asof(
            left_group,
            right_group,
            left_on=left_time_col,
            right_on=right_time_col,
            direction="backward",
            allow_exact_matches=True,
        )

        merged = merged.rename(columns={right_time_col: right_time_output_col})

        # Drop duplicate group column created by merge if present
        dup_group_col = f"{group_col}_y"
        if dup_group_col in merged.columns:
            merged = merged.drop(columns=[dup_group_col])

        # If merge created group_col_x, rename back
        left_group_col = f"{group_col}_x"
        if left_group_col in merged.columns:
            merged = merged.rename(columns={left_group_col: group_col})

        output_parts.append(merged)

    result = pd.concat(output_parts, ignore_index=True)
    return result


def point_in_time_join(impressions, user_features, item_features):
    impressions = impressions.copy().sort_values("impression_time").reset_index(drop=True)

    # -----------------------------
    # 1) User point-in-time join
    # -----------------------------
    user_right_cols = [
        "user_prev_clicks",
        "user_prev_impressions",
        "user_ctr_prior",
        "user_prev_category",
        "user_prev_source",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
    ]
    user_right_cols = [c for c in user_right_cols if c in user_features.columns]

    joined = asof_join_by_group(
        left_df=impressions,
        right_df=user_features,
        group_col="user_id",
        left_time_col="impression_time",
        right_time_col="feature_timestamp",
        right_cols=user_right_cols,
        right_time_output_col="user_feature_time",
    )

    # -----------------------------
    # 2) Item point-in-time join
    # -----------------------------
    item_right_cols = [
        "item_prev_impressions",
        "item_prev_clicks",
        "item_ctr_prior",
    ]
    item_right_cols = [c for c in item_right_cols if c in item_features.columns]

    joined = asof_join_by_group(
        left_df=joined,
        right_df=item_features,
        group_col="item_id",
        left_time_col="impression_time",
        right_time_col="feature_timestamp",
        right_cols=item_right_cols,
        right_time_output_col="item_feature_time",
    )

    # Fill numeric nulls
    numeric_fill = {
        "user_prev_clicks": 0.0,
        "user_prev_impressions": 0.0,
        "user_ctr_prior": 0.0,
        "recent_impression_count": 0.0,
        "recent_click_count": 0.0,
        "recent_save_count": 0.0,
        "recent_hide_count": 0.0,
        "item_prev_impressions": 0.0,
        "item_prev_clicks": 0.0,
        "item_ctr_prior": 0.0,
        "item_age_hours": 0.0,
        "title_length": 0.0,
        "description_length": 0.0,
        "is_hackernews": 0.0,
        "is_rss": 0.0,
        "rank_position": -1.0,
        "preferred_source_match": 0.0,
        "preferred_category_match": 0.0,
    }

    for col, value in numeric_fill.items():
        if col in joined.columns:
            joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(value)

    # Fill categorical nulls
    for col in [
        "source",
        "source_type",
        "category",
        "policy_name",
        "model_version",
        "user_prev_category",
        "user_prev_source",
        "preferred_source",
        "preferred_category",
    ]:
        if col in joined.columns:
            joined[col] = joined[col].fillna("unknown").astype(str)

    # Derived V2 features
    if "category" in joined.columns and "user_prev_category" in joined.columns:
        joined["user_item_category_match_v2"] = (
            joined["category"].astype(str) == joined["user_prev_category"].astype(str)
        ).astype(int)
    else:
        joined["user_item_category_match_v2"] = 0

    if "source" in joined.columns and "user_prev_source" in joined.columns:
        joined["user_item_source_match_v2"] = (
            joined["source"].astype(str) == joined["user_prev_source"].astype(str)
        ).astype(int)
    else:
        joined["user_item_source_match_v2"] = 0

    joined["recency_decay"] = 1.0 / (1.0 + joined["item_age_hours"].fillna(0.0))

    # Final sort for downstream training
    joined = joined.sort_values("impression_time").reset_index(drop=True)

    return joined


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    impressions, user_features, item_features = load_data()
    joined = point_in_time_join(impressions, user_features, item_features)

    joined.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {joined.shape}")
    print(joined.head())
    print("\nColumns:")
    print(joined.columns.tolist())


if __name__ == "__main__":
    main()