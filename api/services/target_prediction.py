from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import httpx
import re
from typing import List, Optional

router = APIRouter(prefix="/predict/target", tags=["prediction"])

class TargetPredictionRequest(BaseModel):
    smiles: str

class TargetResult(BaseModel):
    target: str
    common_name: str
    uniprot_id: str
    probability: float

@router.post("/", response_model=List[TargetResult])
async def predict_targets(batch: TargetPredictionRequest):
    """
    Predict protein targets for a given SMILES string using ChEMBL Similarity Search.
    
    Strategy:
    1. Query ChEMBL for similar molecules (>70% similarity).
    2. Retrieve high-confidence bioactivities (IC50/Ki < 1uM) for those molecules.
    3. Aggregate and rank potential targets by probability.
    """
    # Validation
    if not batch.smiles or len(batch.smiles) < 2:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")

    try:
        from services.chembl_service import ChEMBLService
        service = ChEMBLService()
        
        # Run prediction
        results = await service.predict_targets(batch.smiles, similarity_cutoff=70) # 70% cutoff for broader hits
        
        return results
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Target prediction failed: {str(e)}")
