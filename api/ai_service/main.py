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
app = FastAPI(
    title="BioDockify AI Microservice",
    description="Zero-Cost AI Inference Engine (Railway Free Tier)",
    version="1.0.0"
)

# Enable CORS (Critical for Vercel Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
@app.on_event("startup")
def startup():
    db.init_db()

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "online", "service": "BioDockify AI", "tier": "Zero-Cost"}

@app.post("/upload")
async def upload_compounds(file: UploadFile = File(...), session: Session = Depends(db.get_db)):
    """
    Phase 1: Ingest Experimental Data (CSV).
    1. Read CSV.
    2. Validate SMILES with RDKit.
    3. Save to SQLite.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files allowed")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validation
        if 'smiles' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have a 'smiles' column")
        
        added_count = 0
        errors = 0
        
        for _, row in df.iterrows():
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


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is standard for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
