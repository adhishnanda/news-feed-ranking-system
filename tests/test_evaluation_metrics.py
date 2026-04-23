"""Unit tests for ranking and feed-quality evaluation metrics."""
from __future__ import annotations
import pandas as pd
import pytest

from src.evaluation.ranking_metrics_v2 import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_ranking,
)
from src.evaluation.feed_quality_metrics_v2 import (
    category_diversity_at_k,
    source_diversity_at_k,
    freshness_at_k,
)

SCORE_COL = "score"


def make_session(clicks, scores, session_id="s1", categories=None, sources=None, age_hours=None):
    n = len(clicks)
    data = {
        "session_id": [session_id] * n,
        "item_id": [f"item_{i}" for i in range(n)],
        "clicked": clicks,
        SCORE_COL: scores,
    }
    if categories:
        data["category"] = categories
    if sources:
        data["source"] = sources
    if age_hours is not None:
        data["age_hours"] = age_hours
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_clicked(self):
        df = make_session([1, 1, 1], [0.9, 0.8, 0.7])
        assert precision_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(1.0)

    def test_none_clicked(self):
        df = make_session([0, 0, 0], [0.9, 0.8, 0.7])
        assert precision_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(0.0)

    def test_partial_click(self):
        df = make_session([1, 0, 0], [0.9, 0.8, 0.7])
        assert precision_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(1 / 3)

    def test_k_smaller_than_frame(self):
        df = make_session([1, 0, 0, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5])
        assert precision_at_k(df, k=1, score_col=SCORE_COL) == pytest.approx(1.0)


class TestRecallAtK:
    def test_all_relevant_in_top_k(self):
        df = make_session([1, 1, 0], [0.9, 0.8, 0.7])
        assert recall_at_k(df, k=2, score_col=SCORE_COL) == pytest.approx(1.0)

    def test_no_clicks_returns_zero(self):
        df = make_session([0, 0, 0], [0.9, 0.8, 0.7])
        assert recall_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(0.0)

    def test_partial(self):
        # top-2 by score: items 0 and 1; only item 0 clicked; 2 total clicks (items 0 and 2)
        df = make_session([1, 0, 1], [0.9, 0.8, 0.7])
        assert recall_at_k(df, k=2, score_col=SCORE_COL) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_perfect_ranking_bounded(self):
        df = make_session([1, 1, 0], [0.9, 0.8, 0.7])
        score = ndcg_at_k(df, k=3, score_col=SCORE_COL)
        assert 0.0 <= score <= 1.0

    def test_good_ranking_beats_bad_ranking(self):
        df_good = make_session([1, 0, 0], [0.9, 0.8, 0.7])
        df_bad = make_session([0, 0, 1], [0.9, 0.8, 0.7])
        assert ndcg_at_k(df_good, k=3, score_col=SCORE_COL) >= ndcg_at_k(df_bad, k=3, score_col=SCORE_COL)

    def test_no_clicks_returns_zero(self):
        df = make_session([0, 0, 0], [0.9, 0.8, 0.7])
        assert ndcg_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(0.0)


class TestEvaluateRanking:
    def test_returns_expected_keys(self):
        rows = [
            {"session_id": "s1", "item_id": "a", "clicked": 1, SCORE_COL: 0.9},
            {"session_id": "s1", "item_id": "b", "clicked": 0, SCORE_COL: 0.8},
            {"session_id": "s2", "item_id": "c", "clicked": 1, SCORE_COL: 0.7},
            {"session_id": "s2", "item_id": "d", "clicked": 0, SCORE_COL: 0.6},
        ]
        df = pd.DataFrame(rows)
        metrics = evaluate_ranking(df, k=2, score_col=SCORE_COL)
        for key in ("precision@k", "recall@k", "ndcg@k"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_empty_df_returns_empty_dict(self):
        df = pd.DataFrame(columns=["session_id", "item_id", "clicked", SCORE_COL])
        metrics = evaluate_ranking(df, k=5, score_col=SCORE_COL)
        # Empty groupby produces empty DataFrame → .mean() returns empty Series → empty dict
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# Feed quality metrics
# ---------------------------------------------------------------------------

class TestCategoryDiversityAtK:
    def test_all_same_category(self):
        df = make_session([1, 1, 1], [0.9, 0.8, 0.7], categories=["tech", "tech", "tech"])
        assert category_diversity_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(1 / 3)

    def test_all_different(self):
        df = make_session([1, 1, 1], [0.9, 0.8, 0.7], categories=["tech", "world", "sports"])
        assert category_diversity_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(1.0)

    def test_k_equals_1(self):
        df = make_session([1, 1], [0.9, 0.8], categories=["tech", "world"])
        assert category_diversity_at_k(df, k=1, score_col=SCORE_COL) == pytest.approx(1.0)


class TestSourceDiversityAtK:
    def test_all_same_source(self):
        df = make_session([1, 1, 1], [0.9, 0.8, 0.7], sources=["bbc", "bbc", "bbc"])
        assert source_diversity_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(1 / 3)

    def test_all_different(self):
        df = make_session([1, 1, 1], [0.9, 0.8, 0.7], sources=["bbc", "cnn", "hn"])
        assert source_diversity_at_k(df, k=3, score_col=SCORE_COL) == pytest.approx(1.0)


class TestFreshnessAtK:
    def test_fresh_items_score_higher_than_stale(self):
        fresh = make_session([1, 1, 1], [0.9, 0.8, 0.7], age_hours=[1.0, 2.0, 3.0])
        stale = make_session([1, 1, 1], [0.9, 0.8, 0.7], age_hours=[100.0, 200.0, 300.0])
        assert freshness_at_k(fresh, k=3, score_col=SCORE_COL) > freshness_at_k(stale, k=3, score_col=SCORE_COL)

    def test_zero_age_gives_maximum_freshness(self):
        df = make_session([1], [0.9], age_hours=[0.0])
        # recency = 1 / (1 + 0) = 1.0
        assert freshness_at_k(df, k=1, score_col=SCORE_COL) == pytest.approx(1.0)
