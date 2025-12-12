
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from services.ranking_engine import RankingEngine
from auth import get_current_user

router = APIRouter(prefix="/ranking", tags=["Lead Ranking"])

class RankRequest(BaseModel):
    hits: List[Dict[str, Any]]
    weights: Optional[Dict[str, float]] = None

@router.post("/rank-hits")
async def rank_hits(request: RankRequest, current_user: dict = Depends(get_current_user)):
    """
    Ranks a list of docked compounds based on consensus scoring.
    """
    try:
        engine = RankingEngine(weights=request.weights)
        ranked_results = engine.rank_hits(request.hits)
        return {"status": "success", "ranked_hits": ranked_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")
