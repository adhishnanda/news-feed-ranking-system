from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import requests

from src.ui.feedback_handler import log_interaction, init_user_session


st.set_page_config(page_title="News Feed Ranking System", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"


def fetch_ranked_feed(user_id: str, session_id: str, limit: int = 10) -> pd.DataFrame:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "limit": limit,
    }

    response = requests.post(f"{API_BASE_URL}/rank-feed", json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    items = data.get("items", [])
    return pd.DataFrame(items)


def render_sidebar():
    st.sidebar.title("Feed Controls")

    user_id = st.sidebar.selectbox(
        "Choose user",
        ["user_1", "user_2", "user_3"],
        index=0
    )
    st.session_state.user_id = user_id

    limit = st.sidebar.slider("Number of items", min_value=5, max_value=20, value=10)

    api_status = "unknown"
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            api_status = "online"
        else:
            api_status = f"error ({health.status_code})"
    except Exception:
        api_status = "offline"

    st.sidebar.write(f"API status: **{api_status}**")
    return limit


def render_item(row, rank_position: int):
    item_id = row["item_id"]
    title = row["title"]
    source = row["source"]
    source_type = row["source_type"]
    category = row["category"]
    published_at = row.get("published_at")
    url = row["url"]
    model_score = row.get("model_score")
    freshness_bonus = row.get("freshness_bonus")
    final_rank = row.get("final_rank")

    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(
            f"Source: {source} | Type: {source_type} | Category: {category} | "
            f"Published: {published_at}"
        )

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("Model score", f"{model_score:.4f}" if model_score is not None else "N/A")
        with col_meta2:
            st.metric("Freshness bonus", f"{freshness_bonus:.4f}" if freshness_bonus is not None else "N/A")
        with col_meta3:
            st.metric("Final rank", int(final_rank) if final_rank is not None else rank_position)

        if item_id not in st.session_state.logged_impressions:
            log_interaction(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                event_type="impression",
                item_id=item_id,
                rank_position=rank_position,
                score=model_score,
                policy_name="reranked_feed_v1",
                metadata='{"surface":"streamlit_api_feed"}'
            )
            st.session_state.logged_impressions.add(item_id)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("Save", key=f"save_{item_id}"):
                log_interaction(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                    event_type="save",
                    item_id=item_id,
                    rank_position=rank_position,
                    score=model_score,
                    policy_name="reranked_feed_v1",
                    metadata='{"surface":"streamlit_api_feed"}'
                )
                st.success(f"Saved {item_id}")

        with col2:
            if st.button("Hide", key=f"hide_{item_id}"):
                log_interaction(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                    event_type="hide",
                    item_id=item_id,
                    rank_position=rank_position,
                    score=model_score,
                    policy_name="reranked_feed_v1",
                    metadata='{"surface":"streamlit_api_feed"}'
                )
                st.warning(f"Hidden {item_id}")

        with col3:
            if st.button("Open / Click", key=f"click_{item_id}"):
                log_interaction(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                    event_type="click",
                    item_id=item_id,
                    rank_position=rank_position,
                    score=model_score,
                    policy_name="reranked_feed_v1",
                    metadata='{"surface":"streamlit_api_feed"}'
                )
                st.markdown(f"[Open article]({url})")


def main():
    init_user_session()

    st.title("Personalized News Feed Ranking System")
    st.write("Model-ranked and reranked feed served via FastAPI")

    limit = render_sidebar()

    if st.button("Start New Session"):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.logged_impressions = set()
        st.rerun()

    st.write(f"Current user: `{st.session_state.user_id}`")
    st.write(f"Session ID: `{st.session_state.session_id}`")

    try:
        feed_df = fetch_ranked_feed(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            limit=limit,
        )

        log_interaction(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            event_type="feed_request",
            item_id="feed_request",
            rank_position=0,
            score=None,
            policy_name="reranked_feed_v1",
            metadata=f'{{"surface":"streamlit_api_feed","limit":{limit}}}'
        )

        st.subheader("Ranked Feed")
        st.write(f"Items returned: {len(feed_df)}")

        if feed_df.empty:
            st.warning("No items returned from API.")
            return

        for i, row in enumerate(feed_df.to_dict(orient="records"), start=1):
            render_item(row, rank_position=i)

    except Exception as e:
        st.error(f"Failed to fetch ranked feed from API: {e}")
        st.info("Make sure FastAPI is running on http://127.0.0.1:8000")


if __name__ == "__main__":
    main()