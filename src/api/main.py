from __future__ import annotations
from fastapi import FastAPI

from src.api.schemas import RankFeedRequest, RankFeedResponse
from src.api.ranking_service import RankingService

app = FastAPI(title="News Feed Ranking API")
ranking_service = RankingService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rank-feed", response_model=RankFeedResponse)
def rank_feed(request: RankFeedRequest):
    response = ranking_service.rank_feed(
        user_id=request.user_id,
        session_id=request.session_id,
        limit=request.limit,
    )
    return response