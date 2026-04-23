from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import requests

from src.ui.feedback_handler import log_interaction, init_user_session


st.set_page_config(
    page_title="News Feed Ranking System",
    page_icon="📰",
    layout="wide",
)

API_BASE_URL = "http://127.0.0.1:8000"
METRICS_V2_PATH = PROJECT_ROOT / "data" / "gold" / "metrics_v2.json"
COUNTERFACTUAL_PATH = PROJECT_ROOT / "data" / "gold" / "counterfactual_metrics_v2.json"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_ranked_feed(user_id: str, session_id: str, limit: int = 10) -> tuple[pd.DataFrame, str]:
    payload = {"user_id": user_id, "session_id": session_id, "limit": limit}
    response = requests.post(f"{API_BASE_URL}/rank-feed", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    model_version = data.get("model_version", "unknown")
    return pd.DataFrame(data.get("items", [])), model_version


def check_api_status() -> dict:
    status = {"api": "offline", "redis": "unknown"}
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        status["api"] = "online" if r.status_code == 200 else f"error ({r.status_code})"
    except Exception:
        pass
    try:
        r2 = requests.post(
            f"{API_BASE_URL}/rank-feed-v2",
            json={"user_id": "probe", "session_id": "probe", "limit": 1},
            timeout=5,
        )
        if r2.status_code == 200:
            status["redis"] = "connected" if r2.json().get("candidate_count", 0) > 0 else "available"
        else:
            status["redis"] = "unavailable"
    except Exception:
        status["redis"] = "unavailable"
    return status


def load_metrics() -> dict | None:
    if METRICS_V2_PATH.exists():
        with open(METRICS_V2_PATH) as f:
            return json.load(f)
    return None


def load_counterfactual_metrics() -> dict | None:
    if COUNTERFACTUAL_PATH.exists():
        with open(COUNTERFACTUAL_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, int]:
    st.sidebar.title("📰 Feed Controls")

    user_id = st.sidebar.selectbox(
        "User",
        ["user_1", "user_2", "user_3"],
        index=0,
        help="Simulated user profile for personalization",
    )
    st.session_state.user_id = user_id

    limit = st.sidebar.slider("Items to show", min_value=5, max_value=20, value=10)

    # --- System status ---
    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    if st.sidebar.button("Check status", use_container_width=True):
        with st.sidebar:
            with st.spinner("Checking…"):
                status = check_api_status()
        st.session_state.system_status = status

    status = st.session_state.get("system_status", {})
    api_color = "green" if status.get("api") == "online" else "red"
    redis_color = "green" if status.get("redis") in ("connected", "available") else "orange"
    st.sidebar.markdown(
        f"- API: :{api_color}[{status.get('api', 'unknown')}]  \n"
        f"- Redis: :{redis_color}[{status.get('redis', 'unknown')}]"
    )

    # --- Model metrics ---
    st.sidebar.divider()
    st.sidebar.subheader("Model Metrics (V2)")
    metrics = load_metrics()
    if metrics:
        dataset = metrics.get("dataset", {})
        model_metrics = metrics.get("metrics", [])
        lgbm_val = next((m for m in model_metrics if m.get("model") == "LightGBM_val"), None)
        lr_val = next((m for m in model_metrics if m.get("model") == "LogisticRegression_val"), None)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            auc = lgbm_val.get("auc") if lgbm_val else None
            st.metric("LightGBM AUC", f"{auc:.3f}" if auc is not None else "—")
        with col2:
            auc_lr = lr_val.get("auc") if lr_val else None
            st.metric("LogReg AUC", f"{auc_lr:.3f}" if auc_lr is not None else "—")
        st.sidebar.caption(
            f"Train: {dataset.get('rows_train', '?'):,} "
            f"| Val: {dataset.get('rows_val', '?'):,} "
            f"| Test: {dataset.get('rows_test', '?'):,}"
        )
    else:
        st.sidebar.caption("metrics_v2.json not found")

    # --- Counterfactual ---
    cf = load_counterfactual_metrics()
    if cf:
        st.sidebar.divider()
        st.sidebar.subheader("Counterfactual Eval")
        ips = cf.get("ips_estimate")
        snips = cf.get("snips_estimate")
        if ips is not None:
            st.sidebar.metric("IPS estimate", f"{ips:.4f}")
        if snips is not None:
            st.sidebar.metric("SNIPS estimate", f"{snips:.4f}")

    return user_id, limit


# ---------------------------------------------------------------------------
# Item card
# ---------------------------------------------------------------------------

def render_item(row: dict, rank_position: int):
    item_id = row.get("item_id", "")
    title = row.get("title", "(no title)")
    source = row.get("source", "—")
    source_type = row.get("source_type", "—")
    category = row.get("category", "—")
    published_at = row.get("published_at")
    url = row.get("url", "")
    model_score = row.get("model_score")
    freshness_bonus = row.get("freshness_bonus")
    final_rank = row.get("final_rank")
    user_id = st.session_state.get("user_id", "user_1")
    session_id = st.session_state.get("session_id", "")

    with st.container(border=True):
        header_col, rank_col = st.columns([9, 1])
        with header_col:
            if url:
                st.markdown(f"### [{title}]({url})")
            else:
                st.markdown(f"### {title}")
        with rank_col:
            st.markdown(f"<div style='text-align:right;padding-top:12px;font-size:1.4em;color:gray'>#{rank_position}</div>",
                        unsafe_allow_html=True)

        st.caption(
            f"**{source}** &nbsp;·&nbsp; {source_type} &nbsp;·&nbsp; `{category}` "
            + (f"&nbsp;·&nbsp; {published_at}" if published_at else "")
        )

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("Model score", f"{model_score:.4f}" if model_score is not None else "N/A")
        with col_meta2:
            st.metric("Freshness bonus", f"{freshness_bonus:.4f}" if freshness_bonus is not None else "N/A")
        with col_meta3:
            st.metric("Final rank", int(final_rank) if final_rank is not None else rank_position)

        # Log impression once
        if item_id not in st.session_state.logged_impressions:
            log_interaction(
                user_id=user_id,
                session_id=session_id,
                event_type="impression",
                item_id=item_id,
                rank_position=rank_position,
                score=model_score,
                policy_name="reranked_feed_v1",
                metadata='{"surface":"streamlit_api_feed"}',
            )
            st.session_state.logged_impressions.add(item_id)

        act_col1, act_col2, act_col3 = st.columns([1, 1, 4])

        with act_col1:
            if st.button("Save", key=f"save_{item_id}"):
                log_interaction(
                    user_id=user_id, session_id=session_id, event_type="save",
                    item_id=item_id, rank_position=rank_position, score=model_score,
                    policy_name="reranked_feed_v1", metadata='{"surface":"streamlit_api_feed"}',
                )
                st.success("Saved")

        with act_col2:
            if st.button("Hide", key=f"hide_{item_id}"):
                log_interaction(
                    user_id=user_id, session_id=session_id, event_type="hide",
                    item_id=item_id, rank_position=rank_position, score=model_score,
                    policy_name="reranked_feed_v1", metadata='{"surface":"streamlit_api_feed"}',
                )
                st.warning("Hidden")

        with act_col3:
            if url:
                st.link_button("Open article", url, use_container_width=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_user_session()

    st.title("Personalized News Feed Ranking System")
    st.caption("End-to-end ML ranking: LightGBM scoring · greedy diversity reranking · Redis online feature store")

    user_id, limit = render_sidebar()

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.write(f"User: `{user_id}` &nbsp; Session: `{st.session_state.session_id[:8]}…`")
    with top_col2:
        if st.button("New session", use_container_width=True):
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.logged_impressions = set()
            st.rerun()

    with st.expander("How this works", expanded=False):
        st.markdown(
            """
**Pipeline overview**

1. **Candidate retrieval** — blends popular, recent, category-match, source-match, and unseen-item strategies
2. **Feature enrichment** — merges user history and item features; optionally overrides with Redis online store
3. **LightGBM scoring** — click-through probability estimate per candidate
4. **Greedy diversity reranking** — freshness bonus + category/source penalty to improve slate diversity

Interactions (clicks, saves, hides) are logged to DuckDB for offline evaluation.
"""
        )

    st.divider()

    try:
        feed_df, model_version = fetch_ranked_feed(
            user_id=user_id,
            session_id=st.session_state.session_id,
            limit=limit,
        )

        log_interaction(
            user_id=user_id,
            session_id=st.session_state.session_id,
            event_type="feed_request",
            item_id="feed_request",
            rank_position=0,
            score=None,
            policy_name="reranked_feed_v1",
            metadata=f'{{"surface":"streamlit_api_feed","limit":{limit}}}',
        )

        st.subheader("Ranked Feed", anchor=False)
        st.caption(f"Model: `{model_version}` · {len(feed_df)} items returned")

        if feed_df.empty:
            st.warning("No items returned from the API. Make sure DuckDB has been populated (`make ingest`).")
            return

        for i, row in enumerate(feed_df.to_dict(orient="records"), start=1):
            render_item(row, rank_position=i)

    except Exception as e:
        st.error(f"Failed to fetch feed: {e}")
        st.info(
            "Make sure the FastAPI server is running:  \n"
            "```\nmake run-api\n```"
        )


if __name__ == "__main__":
    main()
