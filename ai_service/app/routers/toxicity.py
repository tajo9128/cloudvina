from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.toxicity_engine import ToxicityEngine
from app.routers.auth import get_current_user

router = APIRouter(prefix="/toxicity", tags=["Toxicity"])

# Initialize Engine
toxicity_engine = ToxicityEngine()

class SmilesList(BaseModel):
    smiles: List[str]

@router.post("/predict")
def predict_toxicity(
    payload: SmilesList,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Predict toxicity alerts (PAINS, Brenk, NIH) for a list of SMILES.
    """
    try:
        results = toxicity_engine.predict(payload.smiles)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
