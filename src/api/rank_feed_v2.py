from src.storage.redis_feature_store_v2 import RedisFeatureStoreV2
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import joblib

from fastapi import APIRouter, HTTPException

from src.api.schemas_v2 import RankFeedV2Request, RankFeedV2Response
from src.reranking.retrieve_candidates_v2 import retrieve_candidates_for_user
from src.reranking.candidate_generation_stub import load_data as load_impressions_data


router = APIRouter()


TRAIN_DATA_PATH = Path("data/gold/train_dataset_v2_embeddings.parquet")
CANDIDATES_CACHE_PATH = Path("data/gold/candidates_v2.parquet")
METRICS_PATH = Path("data/gold/metrics_v2.json")
MODEL_ARTIFACT_PATH = Path("models_artifacts/logistic_model.joblib")

def enrich_with_redis_online_features(scoring_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull latest user/item features from Redis and overwrite local frame where available.
    If Redis is unavailable, return original frame unchanged.
    """
    out = scoring_df.copy()

    try:
        store = RedisFeatureStoreV2()
        if not store.ping():
            return out
    except Exception:
        return out

    user_feature_cols = [
        "user_prev_clicks",
        "user_prev_impressions",
        "user_ctr_prior",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
        "user_prev_category",
        "user_prev_source",
        "user_embedding_norm",
    ]

    item_feature_cols = [
        "item_prev_impressions",
        "item_prev_clicks",
        "item_ctr_prior",
        "item_age_hours",
        "source",
        "source_type",
        "category",
        "title_length",
        "description_length",
        "is_hackernews",
        "is_rss",
        "item_embedding_norm",
    ]

    for idx, row in out.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])

        user_payload = store.get_user_features(user_id)
        item_payload = store.get_item_features(item_id)

        for col in user_feature_cols:
            if col in user_payload and user_payload[col] is not None:
                out.at[idx, col] = user_payload[col]

        for col in item_feature_cols:
            if col in item_payload and item_payload[col] is not None:
                out.at[idx, col] = item_payload[col]

    # recompute derived features after online overwrite
    if "category" in out.columns and "user_prev_category" in out.columns:
        out["preferred_category_match"] = (
            out["category"].astype(str) == out["user_prev_category"].astype(str)
        ).astype(int)
        out["user_item_category_match_v2"] = out["preferred_category_match"]

    if "source" in out.columns and "user_prev_source" in out.columns:
        out["preferred_source_match"] = (
            out["source"].astype(str) == out["user_prev_source"].astype(str)
        ).astype(int)
        out["user_item_source_match_v2"] = out["preferred_source_match"]

    if "item_age_hours" in out.columns:
        age = pd.to_numeric(out["item_age_hours"], errors="coerce").fillna(0.0)
        out["recency_decay"] = 1.0 / (1.0 + age)

    return out


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def load_training_frame() -> pd.DataFrame:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_DATA_PATH}")

    df = pd.read_parquet(TRAIN_DATA_PATH)
    if "impression_time" in df.columns:
        df["impression_time"] = pd.to_datetime(df["impression_time"], errors="coerce")
    return df


def load_model():
    if MODEL_ARTIFACT_PATH.exists():
        try:
            return joblib.load(MODEL_ARTIFACT_PATH)
        except Exception:
            return None
    return None


def build_feature_frame_for_candidates(
    train_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    user_id: str,
) -> pd.DataFrame:
    """
    Build a scoring frame for candidate items by combining:
    - latest user-side row for this user
    - latest item-side row for each candidate item
    - retrieval features
    """
    if candidates_df.empty:
        return pd.DataFrame()

    working = candidates_df.copy()

    # Get latest known user row
    user_hist = train_df[train_df["user_id"].astype(str) == str(user_id)].copy()
    if user_hist.empty:
        # fallback: use latest global row schema
        user_latest = train_df.sort_values("impression_time").tail(1).copy()
        user_latest["user_id"] = str(user_id)
    else:
        user_latest = user_hist.sort_values("impression_time").tail(1).copy()

    user_latest = user_latest.iloc[0].to_dict()

    # Build row per candidate using latest item snapshot + latest user snapshot
    rows = []
    item_hist_all = train_df.sort_values("impression_time").copy()

    for _, cand in working.iterrows():
        item_id = cand["item_id"]
        item_hist = item_hist_all[item_hist_all["item_id"].astype(str) == str(item_id)].copy()

        if item_hist.empty:
            item_latest = {}
        else:
            item_latest = item_hist.tail(1).iloc[0].to_dict()

        row = {}

        # start with user features
        for k, v in user_latest.items():
            row[k] = v

        # overwrite with latest item features where available
        for k, v in item_latest.items():
            row[k] = v

        # force current serving identifiers
        row["user_id"] = str(user_id)
        row["item_id"] = str(item_id)

        # retrieval outputs
        row["blended_retrieval_score"] = safe_float(cand.get("blended_retrieval_score", 0.0))
        row["retrieval_score_raw"] = safe_float(cand.get("retrieval_score_raw", 0.0))
        row["retrieval_strategy_count"] = safe_float(cand.get("retrieval_strategy_count", 0.0))

        # safe categorical overwrites
        if "source" in cand and pd.notna(cand["source"]):
            row["source"] = str(cand["source"])
        if "category" in cand and pd.notna(cand["category"]):
            row["category"] = str(cand["category"])

        rows.append(row)

    scoring_df = pd.DataFrame(rows)

    # make sure key serving-time derived features exist
    if "preferred_category_match" not in scoring_df.columns:
        scoring_df["preferred_category_match"] = (
            scoring_df.get("category", "unknown").astype(str)
            == scoring_df.get("user_prev_category", "unknown").astype(str)
        ).astype(int)

    if "preferred_source_match" not in scoring_df.columns:
        scoring_df["preferred_source_match"] = (
            scoring_df.get("source", "unknown").astype(str)
            == scoring_df.get("user_prev_source", "unknown").astype(str)
        ).astype(int)

    if "user_item_category_match_v2" not in scoring_df.columns:
        scoring_df["user_item_category_match_v2"] = (
            scoring_df.get("category", "unknown").astype(str)
            == scoring_df.get("user_prev_category", "unknown").astype(str)
        ).astype(int)

    if "user_item_source_match_v2" not in scoring_df.columns:
        scoring_df["user_item_source_match_v2"] = (
            scoring_df.get("source", "unknown").astype(str)
            == scoring_df.get("user_prev_source", "unknown").astype(str)
        ).astype(int)

    if "recency_decay" not in scoring_df.columns:
        item_age = pd.to_numeric(scoring_df.get("item_age_hours", 0.0), errors="coerce").fillna(0.0)
        scoring_df["recency_decay"] = 1.0 / (1.0 + item_age)

    if "embedding_affinity_bucket" not in scoring_df.columns:
        cosine = pd.to_numeric(scoring_df.get("user_item_embedding_cosine", 0.0), errors="coerce").fillna(0.0)
        scoring_df["embedding_affinity_bucket"] = pd.cut(
            cosine,
            bins=[-1.0, 0.2, 0.5, 0.8, 1.0],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True,
        ).astype(str)

    return scoring_df


def get_feature_lists(df: pd.DataFrame):
    numeric_features = [
        "rank_position",
        "hour",
        "day_of_week",
        "is_weekend",
        "item_age_hours",
        "title_length",
        "description_length",
        "hour_published",
        "weekday_published",
        "is_hackernews",
        "is_rss",
        "recent_impression_count",
        "recent_click_count",
        "recent_save_count",
        "recent_hide_count",
        "user_prev_clicks",
        "user_prev_impressions",
        "user_ctr_prior",
        "item_prev_impressions",
        "item_prev_clicks",
        "item_ctr_prior",
        "preferred_source_match",
        "preferred_category_match",
        "user_item_category_match_v2",
        "user_item_source_match_v2",
        "recency_decay",
        "user_item_embedding_cosine",
        "user_embedding_norm",
        "item_embedding_norm",
        "blended_retrieval_score",
        "retrieval_score_raw",
        "retrieval_strategy_count",
    ]

    categorical_features = [
        "source",
        "source_type",
        "category",
        "policy_name",
        "model_version",
        "user_prev_category",
        "user_prev_source",
        "embedding_affinity_bucket",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features


def score_candidates(scoring_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use saved model if available.
    If saved model is missing/incompatible, fallback to heuristic scoring.
    """
    out = scoring_df.copy()

    model = load_model()
    numeric_features, categorical_features = get_feature_lists(out)
    feature_cols = numeric_features + categorical_features

    # fill expected columns
    for col in numeric_features:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    for col in categorical_features:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)

    if model is not None:
        try:
            probs = model.predict_proba(out[feature_cols])[:, 1]
            out["model_score"] = probs
            return out
        except Exception:
            pass

    # fallback heuristic if artifact is absent or incompatible
    out["model_score"] = (
        0.35 * pd.to_numeric(out.get("item_ctr_prior", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out.get("user_ctr_prior", 0.0), errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(out.get("recency_decay", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out.get("preferred_category_match", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out.get("preferred_source_match", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out.get("user_item_embedding_cosine", 0.0), errors="coerce").fillna(0.0)
    )

    return out


def apply_reranking(scored_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Lightweight reranking:
    - reward retrieval confidence
    - reward freshness
    - penalize repeated category/source in final slate
    """
    df = scored_df.copy()

    if df.empty:
        return df

    df["retrieval_bonus"] = 0.05 * pd.to_numeric(df.get("blended_retrieval_score", 0.0), errors="coerce").fillna(0.0)
    df["freshness_bonus"] = 0.20 * pd.to_numeric(df.get("recency_decay", 0.0), errors="coerce").fillna(0.0)
    df["base_final_score"] = df["model_score"] + df["retrieval_bonus"] + df["freshness_bonus"]

    # greedy reranking
    selected_rows = []
    category_counts = {}
    source_counts = {}

    working = df.sort_values("base_final_score", ascending=False).reset_index(drop=True)

    while len(selected_rows) < min(limit, len(working)):
        best_idx = None
        best_score = None

        for idx, row in working.iterrows():
            category = str(row.get("category", "unknown"))
            source = str(row.get("source", "unknown"))

            category_penalty = 0.05 * category_counts.get(category, 0)
            source_penalty = 0.05 * source_counts.get(source, 0)

            final_score = safe_float(row["base_final_score"]) - category_penalty - source_penalty

            if best_score is None or final_score > best_score:
                best_score = final_score
                best_idx = idx

        chosen = working.loc[best_idx].copy()
        chosen["final_score"] = best_score
        selected_rows.append(chosen)

        chosen_category = str(chosen.get("category", "unknown"))
        chosen_source = str(chosen.get("source", "unknown"))

        category_counts[chosen_category] = category_counts.get(chosen_category, 0) + 1
        source_counts[chosen_source] = source_counts.get(chosen_source, 0) + 1

        working = working.drop(index=best_idx).reset_index(drop=True)

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


@router.post("/rank-feed-v2", response_model=RankFeedV2Response)
def rank_feed_v2(request: RankFeedV2Request):
    try:
        impressions_df = load_impressions_data()
        train_df = load_training_frame()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load serving data: {e}")

    try:
        candidates_df = retrieve_candidates_for_user(
            impressions_df,
            user_id=request.user_id,
            top_k_each=max(10, request.limit * 2),
            final_top_k=max(20, request.limit * 3),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candidate retrieval failed: {e}")

    if candidates_df.empty:
        return RankFeedV2Response(
            user_id=request.user_id,
            session_id=request.session_id,
            candidate_count=0,
            returned_count=0,
            items=[],
        )

    try:
        scoring_df = build_feature_frame_for_candidates(train_df, candidates_df, request.user_id)
        scoring_df = enrich_with_redis_online_features(scoring_df)
        scored_df = score_candidates(scoring_df)
        ranked_df = apply_reranking(scored_df, request.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking pipeline failed: {e}")

    items = []
    for _, row in ranked_df.iterrows():
        items.append(
            {
                "item_id": str(row.get("item_id")),
                "source": None if pd.isna(row.get("source")) else str(row.get("source")),
                "category": None if pd.isna(row.get("category")) else str(row.get("category")),
                "blended_retrieval_score": safe_float(row.get("blended_retrieval_score"), 0.0),
                "model_score": safe_float(row.get("model_score"), 0.0),
                "final_score": safe_float(row.get("final_score"), 0.0),
                "rank": int(row.get("rank", 0)),
            }
        )

    return RankFeedV2Response(
        user_id=request.user_id,
        session_id=request.session_id,
        candidate_count=int(len(candidates_df)),
        returned_count=int(len(items)),
        items=items,
    )