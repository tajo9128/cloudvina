from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
from app.auth import get_current_user
from app.services.chemberta import QSARService
from app.services.storage import upload_file_to_hub, download_file_from_hub
import uuid
import os
import pickle

router = APIRouter()

# Services
qsar_service = QSARService()

# Schemas
class TrainRequest(BaseModel):
    project_id: str
    model_name: str
    target_column: str
    smiles: List[str]
    targets: List[float]

class PredictRequest(BaseModel):
    model_path: str # HF path
    smiles: List[str]

@router.post("/train")
def train_model(
    payload: TrainRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Train a QSAR model and upload artifacts to HF Hub.
    """
    try:
        # 1. Train Model (Mocked/Real)
        result = qsar_service.train_model(payload.smiles, payload.targets)
        
        # 2. Save Artifacts locally
        model_id = str(uuid.uuid4())
        filename = f"model_{model_id}.pkl"
        with open(filename, "wb") as f:
            f.write(result["model_blob"])
        
        # 3. Upload to HF Hub
        hf_path = f"models/{user['id']}/{payload.project_id}/{filename}"
        dataset_repo = os.getenv("HF_DATASET_REPO", "tajo9128/biodockify_ai")
        
        upload_url = upload_file_to_hub(
            file_path=filename,
            path_in_repo=hf_path,
            repo_id=dataset_repo, 
            repo_type="space"
        )
        
        os.remove(filename)
        
        return {
            "status": "success",
            "model_id": model_id,
            "metrics": result["metrics"],
            "model_path": hf_path,
            "upload_url": upload_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
def predict_qsar(
    payload: PredictRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    try:
        # 1. Download Model
        dataset_repo = os.getenv("HF_DATASET_REPO", "tajo9128/biodockify_ai")
        local_path = download_file_from_hub(
            path_in_repo=payload.model_path,
            repo_id=dataset_repo,
            repo_type="space"
        )
        
        # 2. Load
        with open(local_path, "rb") as f:
            model_blob = f.read()
            
        # 3. Predict
        predictions = qsar_service.predict(model_blob, payload.smiles)
        
        return {
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# New: Direct disease-based prediction using pre-trained models
class DiseasePredict(BaseModel):
    smiles: List[str]
    disease_target: str = "alzheimers"  # Options: alzheimers, cancer, diabetes, parkinson, cardiovascular


@router.post("/predict/disease")
def predict_by_disease(
    payload: DiseasePredict,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Predict bioactivity using pre-trained ChemBERTa models.
    
    Available disease targets: alzheimers, cancer, diabetes, parkinson, cardiovascular
    """
    try:
        results = qsar_service.predict_activity(
            smiles_list=payload.smiles,
            disease_target=payload.disease_target
        )
        return {"predictions": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch Prediction from CSV
from fastapi import File, UploadFile
import pandas as pd
import io

@router.post("/predict/batch")
async def batch_predict(
    file: UploadFile = File(...),
    disease_target: str = "alzheimers",
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Batch prediction from CSV file.
    
    CSV must have a 'smiles' column (optionally 'compound_name' and 'disease_target').
    If disease_target column exists in CSV, it overrides the query param.
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'smiles' not in df.columns:
            raise HTTPException(400, "CSV must have a 'smiles' column")
        
        # Get SMILES list
        smiles_list = df['smiles'].dropna().tolist()
        
        if len(smiles_list) == 0:
            raise HTTPException(400, "No valid SMILES found in CSV")
        
        if len(smiles_list) > 1000:
            raise HTTPException(400, "Maximum 1000 compounds per batch")
        
        # Determine disease target (column overrides param)
        target = disease_target
        if 'disease_target' in df.columns:
            # Use first non-null value
            target = df['disease_target'].dropna().iloc[0] if len(df['disease_target'].dropna()) > 0 else disease_target
        
        # Run predictions
        results = qsar_service.predict_activity(smiles_list, target)
        
        # Merge with original data if compound_name exists
        if 'compound_name' in df.columns:
            for i, res in enumerate(results):
                if i < len(df):
                    res['compound_name'] = df.iloc[i].get('compound_name', f'compound_{i+1}')
        
        return {
            "total": len(smiles_list),
            "disease_target": target,
            "predictions": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

