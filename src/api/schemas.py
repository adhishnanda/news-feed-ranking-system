from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List


class RankFeedRequest(BaseModel):
    user_id: str
    session_id: str
    limit: int = 10


class RankedItem(BaseModel):
    item_id: str
    title: str
    source: str
    source_type: str
    category: str
    url: str
    published_at: Optional[str] = None
    model_score: float
    freshness_bonus: Optional[float] = None
    final_rank: Optional[int] = None


class RankFeedResponse(BaseModel):
    user_id: str
    session_id: str
    model_version: str
    items: List[RankedItem]