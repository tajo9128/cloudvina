
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from services.ml_scorer import MLScorer, rank_compounds
from auth import get_current_user

router = APIRouter(prefix="/ranking", tags=["Lead Ranking"])

class RankRequest(BaseModel):
    hits: List[Dict[str, Any]]
    profile: Optional[str] = 'balanced'  # 'speed', 'accuracy', 'balanced', 'novelty'
    weights: Optional[Dict[str, float]] = None

@router.post("/rank-hits")
async def rank_hits(request: RankRequest, current_user: dict = Depends(get_current_user)):
    """
    Ranks a list of docked compounds using ML-based consensus scoring.
    
    **Profiles:**
    - `speed`: Prioritizes fast docking scores (60% docking)
    - `accuracy`: Prioritizes MM-GBSA ΔG (35% MM-GBSA)
    - `balanced`: Default balanced weights
    - `novelty`: Highlights unique binding modes
    
    **Custom Weights (optional):**
    ```json
    {"docking": 0.35, "md_stability": 0.25, "mmgbsa": 0.25, "admet": 0.15}
    ```
    
    Returns ranked list with:
    - `consensus_score`: 0-1 weighted score
    - `confidence`: Low/Medium/High prediction confidence
    - `feature_breakdown`: Contribution from each component
    - `rank`: Position in ranked list
    """
    try:
        scorer = MLScorer(profile=request.profile, custom_weights=request.weights)
        ranked_results = scorer.rank_hits(request.hits)
        
        # Add explanations for top 10
        for hit in ranked_results[:10]:
            hit['explanation'] = scorer.explain_ranking(hit)
        
        return {
            "status": "success",
            "profile": request.profile,
            "weights_used": scorer.weights,
            "ranked_hits": ranked_results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

@router.get("/profiles")
async def get_ranking_profiles():
    """
    Get available ranking profiles and their weight configurations.
    """
    return {
        "profiles": MLScorer.PROFILE_WEIGHTS,
        "default": "balanced"
    }
