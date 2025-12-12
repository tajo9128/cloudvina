from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import os
import io
import pandas as pd
from rdkit import Chem
import database as db

# Initialize App
            smiles = row['smiles']
            name = row.get('name', 'Unknown')
            
            # RDKit Check
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                errors += 1
                continue
                
            # Canonicalize for uniqueness
            canon_smiles = Chem.MolToSmiles(mol)
            
            # Check duplicates
            existing = session.query(db.Compound).filter(db.Compound.smiles == canon_smiles).first()
            if not existing:
                new_compound = db.Compound(
                    smiles=canon_smiles,
                    compound_name=name,
                    source="CSV Upload"
                )
                session.add(new_compound)
                added_count += 1
        
```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import os
import io
import pandas as pd
from rdkit import Chem
import database as db
from pydantic import BaseModel # Added for BaseModel
# Assuming CompoundBatch and ml_engine are defined/imported elsewhere,
# or will be added by subsequent instructions.
# For now, let's assume CompoundBatch is a BaseModel and ml_engine is an instance.

# Initialize App
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the DB session
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

# Placeholder for ML Engine (will be initialized later)
ml_engine = None # This should be initialized properly, e.g., in an app startup event

# Pydantic models for request bodies
class CompoundBatch(BaseModel):
    smiles_list: List[str]

class FeedbackLoop(BaseModel):
    smiles: str
    true_label: int # 1 = Toxic, 0 = Safe

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Toxicity Prediction API"}

# Endpoint to get all compounds
@app.get("/compounds", response_model=List[db.CompoundSchema])
async def get_compounds(db_session: Session = Depends(get_db)):
    compounds = db_session.query(db.Compound).all()
    return compounds

# Endpoint to upload CSV
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), db_session: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'smiles' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'smiles' column.")
            
        added_count = 0
        errors = 0
        
        for index, row in df.iterrows():
            smiles = row['smiles']
            name = row.get('name', 'Unknown')
            
            # RDKit Check
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                errors += 1
                continue
                
            # Canonicalize for uniqueness
            canon_smiles = Chem.MolToSmiles(mol)
            
            # Check duplicates
            existing = session.query(db.Compound).filter(db.Compound.smiles == canon_smiles).first()
            if not existing:
                new_compound = db.Compound(
                    smiles=canon_smiles,
                    compound_name=name,
                    source="CSV Upload"
                )
                session.add(new_compound)
                added_count += 1
        
        session.commit()
        
        return {
            "status": "success",
            "compounds_added": added_count,
            "invalid_smiles": errors,
            "message": f"Successfully ingested {added_count} compounds."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/toxicity")
async def predict_toxicity(payload: CompoundBatch):
    """Predict toxicity for a batch of SMILES"""
    if not ml_engine:
        raise HTTPException(503, "ML Engine not ready")
        
    results = ml_engine.predict_toxicity(payload.smiles_list)
    return {"predictions": results}

@app.post("/train/feedback")
async def active_learning(payload: FeedbackLoop):
    """
    User corrects the AI.
    1. Receive {smiles, true_label}
    2. Retrain model instantly.
    """
    if not ml_engine:
        raise HTTPException(503, "ML Engine not ready")
        
    result = ml_engine.retrain_model(payload.smiles, payload.true_label)
    return result


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is standard for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
```
