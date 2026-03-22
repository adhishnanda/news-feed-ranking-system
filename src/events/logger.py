from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

from src.utils.config import load_yaml
from src.storage.duckdb_client import DuckDBClient
from src.events.schemas import EventPayload


class EventLogger:
    def __init__(self):
        cfg = load_yaml("configs/config.yaml")
        self.db = DuckDBClient(cfg["paths"]["duckdb"])
        self.db.create_tables()

    def log_event(self, payload: EventPayload) -> None:
        df = pd.DataFrame([{
            "event_id": payload.event_id,
            "timestamp": payload.timestamp,
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "event_type": payload.event_type,
            "item_id": payload.item_id,
            "rank_position": payload.rank_position,
            "model_version": payload.model_version,
            "score": payload.score,
            "policy_name": payload.policy_name,
            "propensity": payload.propensity,
            "dwell_time": payload.dwell_time,
            "device_type": payload.device_type,
            "metadata": payload.metadata,
        }])

        self.db.insert_df("events", df)

    @staticmethod
    def now_utc():
        return datetime.now(timezone.utc)