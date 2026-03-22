from __future__ import annotations
import pandas as pd
from datetime import datetime, timezone

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.storage.parquet_io import read_parquet
from src.models.predict import load_model, predict_scores
from src.reranking.score_combiner import apply_reranking


class RankingService:
    def __init__(self):
        self.cfg = load_yaml("configs/config.yaml")
        self.db = DuckDBClient(self.cfg["paths"]["duckdb"])
        self.model = load_model()

        self.item_features = read_parquet(f"{self.cfg['paths']['gold_dir']}/item_features.parquet")
        self.user_features = read_parquet(f"{self.cfg['paths']['gold_dir']}/user_features.parquet")

    def get_user_features(self, user_id: str) -> pd.DataFrame:
        user_df = self.user_features[self.user_features["user_id"] == user_id].copy()

        if user_df.empty:
            user_df = pd.DataFrame([{
                "user_id": user_id,
                "recent_impression_count": 0,
                "recent_click_count": 0,
                "recent_save_count": 0,
                "recent_hide_count": 0,
                "preferred_source": "unknown",
                "preferred_category": "unknown",
            }])

        return user_df

    def get_candidates(self, limit: int = 50) -> pd.DataFrame:
        query = f"""
            SELECT
                item_id,
                title,
                description,
                source,
                source_type,
                category,
                published_at,
                url
            FROM content_items
            WHERE title IS NOT NULL
              AND TRIM(title) <> ''
            ORDER BY published_at DESC NULLS LAST
            LIMIT {limit}
        """
        return self.db.query_df(query)

    def build_scoring_frame(self, user_id: str, candidates: pd.DataFrame) -> pd.DataFrame:
        user_df = self.get_user_features(user_id)

        scoring_df = candidates.merge(
            self.item_features,
            on="item_id",
            how="left",
            suffixes=("", "_item")
        )

        user_row = user_df.iloc[0].to_dict()
        for col, val in user_row.items():
            if col != "user_id":
                scoring_df[col] = val

        now_ts = datetime.now(timezone.utc)
        scoring_df["hour_of_day"] = now_ts.hour
        scoring_df["weekday"] = now_ts.weekday()
        scoring_df["is_weekend"] = int(now_ts.weekday() in [5, 6])

        scoring_df["preferred_source_match"] = (
            scoring_df["source"].fillna("unknown") == scoring_df["preferred_source"].fillna("unknown")
        ).astype(int)

        scoring_df["preferred_category_match"] = (
            scoring_df["category"].fillna("unknown") == scoring_df["preferred_category"].fillna("unknown")
        ).astype(int)

        scoring_df = scoring_df[[
            "item_id",
            "title",
            "source",
            "source_type",
            "category",
            "url",
            "published_at",
            "age_hours",
            "title_length",
            "description_length",
            "hour_published",
            "weekday_published",
            "recent_impression_count",
            "recent_click_count",
            "recent_save_count",
            "recent_hide_count",
            "preferred_source",
            "preferred_category",
            "hour_of_day",
            "weekday",
            "is_weekend",
            "preferred_source_match",
            "preferred_category_match",
        ]].copy()

        return scoring_df

    def rank_feed(self, user_id: str, session_id: str, limit: int = 10) -> dict:
        candidates = self.get_candidates(limit=50)
        scoring_df = self.build_scoring_frame(user_id, candidates)

        feature_df = scoring_df.drop(columns=[
            "item_id", "title", "url", "published_at"
        ], errors="ignore")

        scored = predict_scores(self.model, feature_df)

        result = scoring_df.copy()
        result["model_score"] = scored["model_score"]

        reranked = apply_reranking(result, top_k=limit)

        items = []
        for _, row in reranked.iterrows():
            items.append({
                "item_id": row["item_id"],
                "title": row["title"],
                "source": row["source"],
                "source_type": row["source_type"],
                "category": row["category"],
                "url": row["url"],
                "published_at": str(row["published_at"]) if pd.notnull(row["published_at"]) else None,
                "model_score": float(row["model_score"]),
                "freshness_bonus": float(row["freshness_bonus"]) if pd.notnull(row.get("freshness_bonus")) else None,
                "final_rank": int(row["final_rank"]) if pd.notnull(row.get("final_rank")) else None,
            })
        
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "model_version": "logistic_v1",
            "items": items,
        }