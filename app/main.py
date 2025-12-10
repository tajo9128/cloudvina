from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.auth import get_current_user
from app.services.ml_engine import MLEngine # Integrated existing engine
from app.routers import auth, projects, qsar, compounds
from typing import Dict, Any, List
from pydantic import BaseModel

# Initialize Toxicity Engine (Cold Start if needed)
ml_engine = MLEngine(model_dir="data/models")

app = FastAPI(
    title="AI.BioDockify API",
    description="AI/QSAR Microservice running on Hugging Face Spaces. Merged with Phase 4 Toxicity Engine.",
    version="0.5.0",
    root_path="/api/v1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(compounds.router, prefix="/compounds", tags=["compounds"])
app.include_router(qsar.router, prefix="/qsar", tags=["qsar"])

# --- Legacy Phase 4 Endpoints (Toxicity) - To be moved to proper router later ---
class CompoundBatch(BaseModel):
    smiles_list: List[str]

class FeedbackLoop(BaseModel):
    smiles: str
    true_label: int

@app.get("/")
def read_root():
    return {"status": "online", "service": "AI.BioDockify", "version": "0.4.0"}

@app.post("/predict/toxicity", tags=["legacy"])
async def predict_toxicity(
    payload: CompoundBatch,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Predict toxicity for a batch of SMILES (Phase 4 Feature)"""
    if not ml_engine:
        raise HTTPException(503, "ML Engine not ready")
        
    results = ml_engine.predict_toxicity(payload.smiles_list)
    return {"predictions": results}

@app.post("/train/feedback", tags=["legacy"])
async def active_learning(
    payload: FeedbackLoop,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    User corrects the AI (Active Learning).
    """
    if not ml_engine:
        raise HTTPException(503, "ML Engine not ready")
        
    result = ml_engine.retrain_model(payload.smiles, payload.true_label)
    return result
