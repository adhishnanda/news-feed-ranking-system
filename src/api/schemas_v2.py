from pydantic import BaseModel, Field


class RankFeedV2Request(BaseModel):
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    limit: int = Field(10, ge=1, le=50, description="Number of items to return")


class RankedItemV2(BaseModel):
    item_id: str
    source: str | None = None
    category: str | None = None
    blended_retrieval_score: float | None = None
    model_score: float | None = None
    final_score: float | None = None
    rank: int


class RankFeedV2Response(BaseModel):
    user_id: str
    session_id: str
    candidate_count: int
    returned_count: int
    items: list[RankedItemV2]