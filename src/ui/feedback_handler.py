from __future__ import annotations
import streamlit as st

from src.events.logger import EventLogger
from src.events.schemas import EventPayload


def log_interaction(
    user_id: str,
    session_id: str,
    event_type: str,
    item_id: str,
    rank_position: int,
    score: float | None = None,
    policy_name: str = "manual_feed",
    metadata: str = "{}"
):
    logger = EventLogger()

    payload = EventPayload(
        timestamp=logger.now_utc(),
        user_id=user_id,
        session_id=session_id,
        event_type=event_type,
        item_id=item_id,
        rank_position=rank_position,
        score=score,
        policy_name=policy_name,
        metadata=metadata,
    )
    logger.log_event(payload)


def init_user_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_1"

    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    if "logged_impressions" not in st.session_state:
        st.session_state.logged_impressions = set()