from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class EventPayload(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    user_id: str
    session_id: str
    event_type: str
    item_id: str
    rank_position: Optional[int] = None
    model_version: Optional[str] = "v0_manual"
    score: Optional[float] = None
    policy_name: Optional[str] = "manual_feed"
    propensity: Optional[float] = None
    dwell_time: Optional[float] = None
    device_type: Optional[str] = "desktop"
    metadata: Optional[str] = "{}"