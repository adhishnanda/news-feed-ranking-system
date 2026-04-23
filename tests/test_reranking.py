"""Unit tests for candidate retrieval and reranking utilities."""
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest

from src.reranking.diversity import greedy_diversity_rerank
from src.reranking.freshness import compute_freshness_bonus
from src.reranking.score_combiner import apply_reranking
from src.reranking.candidate_generation_stub import (
    get_popular_candidates,
    get_recent_candidates,
    get_user_unseen_candidates,
    get_user_preferences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_impressions(n=20, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["tech", "world", "sports", "science"]
    sources = ["bbc", "hn", "techcrunch", "ars"]
    return pd.DataFrame({
        "item_id": [f"item_{i}" for i in range(n)],
        "user_id": [f"user_{i % 3}" for i in range(n)],
        "clicked": rng.integers(0, 2, n).tolist(),
        "impression_time": pd.date_range("2025-01-01", periods=n, freq="h"),
        "source": [sources[i % len(sources)] for i in range(n)],
        "category": [categories[i % len(categories)] for i in range(n)],
        "preferred_category": [categories[i % len(categories)] for i in range(n)],
        "preferred_source": [sources[i % len(sources)] for i in range(n)],
    })


def make_scored_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    categories = (["tech", "world", "sports", "science"] * 3)[:n]
    sources = (["bbc", "hn", "techcrunch", "ars"] * 3)[:n]
    return pd.DataFrame({
        "item_id": [f"item_{i}" for i in range(n)],
        "model_score": rng.random(n).tolist(),
        "age_hours": rng.uniform(1, 100, n).tolist(),
        "category": categories,
        "source": sources,
    })


# ---------------------------------------------------------------------------
# Greedy diversity reranking (requires preliminary_score column)
# ---------------------------------------------------------------------------

class TestGreedyDiversityRerank:
    def make_df(self, n=10):
        df = make_scored_df(n)
        df["preliminary_score"] = df["model_score"]  # required column name
        return df

    def test_returns_correct_count(self):
        result = greedy_diversity_rerank(self.make_df(10), top_k=5)
        assert len(result) == 5

    def test_top_k_larger_than_pool(self):
        result = greedy_diversity_rerank(self.make_df(5), top_k=20)
        assert len(result) == 5

    def test_all_same_category_does_not_crash(self):
        df = self.make_df(10)
        df["category"] = "tech"
        result = greedy_diversity_rerank(df, top_k=5)
        assert len(result) == 5

    def test_output_has_final_rank_column(self):
        result = greedy_diversity_rerank(self.make_df(6), top_k=3)
        assert "final_rank" in result.columns
        assert list(result["final_rank"]) == [1, 2, 3]

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["item_id", "preliminary_score", "category", "source"])
        result = greedy_diversity_rerank(df, top_k=5)
        assert result.empty


# ---------------------------------------------------------------------------
# Freshness bonus
# ---------------------------------------------------------------------------

class TestFreshnessBonus:
    def test_younger_items_get_higher_bonus(self):
        ages = pd.Series([1.0, 100.0])
        bonuses = compute_freshness_bonus(ages)
        assert bonuses.iloc[0] > bonuses.iloc[1]

    def test_zero_age_gives_maximum_bonus(self):
        ages = pd.Series([0.0])
        bonuses = compute_freshness_bonus(ages, max_bonus=0.15)
        assert bonuses.iloc[0] == pytest.approx(0.15)

    def test_output_length_matches_input(self):
        ages = pd.Series([5.0, 10.0, 20.0])
        assert len(compute_freshness_bonus(ages)) == 3

    def test_all_bonuses_non_negative(self):
        ages = pd.Series([0.0, 1.0, 24.0, 200.0])
        assert (compute_freshness_bonus(ages) >= 0).all()


# ---------------------------------------------------------------------------
# Full reranking pipeline via score_combiner
# ---------------------------------------------------------------------------

class TestApplyReranking:
    def test_returns_top_k_items(self):
        df = make_scored_df(10)
        result = apply_reranking(df, top_k=5)
        assert len(result) == 5

    def test_output_has_required_columns(self):
        df = make_scored_df(6)
        result = apply_reranking(df, top_k=3)
        for col in ("final_rank", "freshness_bonus", "preliminary_score", "final_score"):
            assert col in result.columns

    def test_ranks_are_sequential(self):
        df = make_scored_df(5)
        result = apply_reranking(df, top_k=5)
        assert list(result["final_rank"]) == list(range(1, len(result) + 1))


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

class TestCandidateGeneration:
    def setup_method(self):
        self.df = make_impressions(20)

    def test_popular_candidates_count(self):
        result = get_popular_candidates(self.df, top_k=5)
        assert len(result) <= 5

    def test_popular_candidates_has_required_cols(self):
        result = get_popular_candidates(self.df, top_k=5)
        for col in ("item_id", "retrieval_strategy", "retrieval_score"):
            assert col in result.columns
        assert (result["retrieval_strategy"] == "popular").all()

    def test_recent_candidates_count(self):
        result = get_recent_candidates(self.df, top_k=5)
        assert len(result) <= 5

    def test_unseen_candidates_excludes_seen(self):
        seen_user = "user_0"
        seen = set(self.df[self.df["user_id"] == seen_user]["item_id"].tolist())
        result = get_user_unseen_candidates(self.df, seen_user, top_k=20)
        if not result.empty:
            returned = set(result["item_id"].tolist())
            assert returned.isdisjoint(seen)

    def test_user_preferences_returns_dict(self):
        prefs = get_user_preferences(self.df, "user_0")
        assert isinstance(prefs, dict)
        assert "preferred_category" in prefs
        assert "preferred_source" in prefs

    def test_unknown_user_returns_none_preferences(self):
        prefs = get_user_preferences(self.df, "ghost_user_99")
        assert prefs["preferred_category"] is None
        assert prefs["preferred_source"] is None
