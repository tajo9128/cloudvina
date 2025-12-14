from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.auth import get_current_user
from app.services.ml_engine import MLEngine # Integrated existing engine (Keep for Auto-QSAR if needed)
from app.routers import auth, projects, qsar, compounds, toxicity # Added toxicity
from typing import Dict, Any, List
from pydantic import BaseModel

# Initialize Toxicity Engine (Cold Start if needed)
ml_engine = MLEngine(model_dir="data/models")

app = FastAPI(
    title="AI.BioDockify API",
    description="AI/QSAR Microservice running on Hugging Face Spaces. Includes Auto-QSAR and Toxicity Prediction.",
    version="0.6.0",
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
app.include_router(toxicity.router, prefix="/toxicity", tags=["toxicity"]) # New Router

@app.get("/")
def read_root():
    return {"status": "online", "service": "AI.BioDockify", "version": "0.6.0"}
