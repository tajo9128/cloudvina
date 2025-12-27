"""
BioDockify API - FastAPI Backend
Handles authentication, job submission, and AWS Batch orchestration
"""
from fastapi import FastAPI, Depends, HTTPException, status, Security, Body, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List
import os
from datetime import datetime
import uuid
import sys

# Force unbuffered output for Render logs
sys.stdout.reconfigure(line_buffering=True)

# Suppress noisy HTTP libraries
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import auth and AWS utilities
from auth import supabase, get_current_user
from aws_services import (
    generate_presigned_upload_urls,
    generate_presigned_download_url,
    submit_batch_job,
    get_batch_job_status
)
from fastapi.responses import StreamingResponse
from services.ai_explainer import AIExplainer
from tools import router as tools_router
from routes.admin import router as admin_router
from routes.evolution import router as evolution_router
from routes.batch import router as batch_router
from routes.jobs import router as jobs_router # NEW: Dedicated Jobs Router
from routes.feedback import router as feedback_router 
from services.cavity_detector import CavityDetector
from services.drug_properties import DrugPropertiesCalculator
# from services.smiles_converter import pdbqt_to_pdb # Moved to function
# from rdkit import Chem # Moved to function

app = FastAPI(
    title="BioDockify API",
    description="Molecular docking as a service with AutoDock Vina",
    version="6.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    print("="*50)
    print("BioDockify API v6.3.0 - Deployed at " + datetime.utcnow().isoformat())
    print("Optimization: Lazy Loading Enabled")
    print("="*50)

    # --- Start Sentinel Monitoring Loop ---
    import asyncio
    
    # 1. Self-Healing Sentinel (5 min interval)
    async def start_sentinel_background():
        """Background loop for Self-Healing"""
        print("🤖 Sentinel: Background Monitor Started (Interval: 5m)")
        while True:
            await asyncio.sleep(300) # Wait 5 minutes
            try:
                from services.sentinel import BioDockifySentinel
                from auth import get_service_client
                
                svc_client = get_service_client()
                sentinel = BioDockifySentinel(svc_client)
                
                # Run Scan
                report = await sentinel.scan_and_heal()
                
                if report['anomalies_detected'] > 0:
                     print(f"🤖 Sentinel: Auto-Healed {report['anomalies_detected']} anomalies.")
                     
            except Exception as e:
                print(f"❌ Sentinel Loop Error: {e}")

    # 2. Zero-Failure Queue Processor (5 sec interval)
    async def start_queue_processor():
        """Aggressive Loop to consume QUEUED jobs immediately"""
        print("⚡ Queue Processor: Started (Interval: 5s)")
        while True:
            await asyncio.sleep(5)
            try:
                from services.queue_processor import QueueProcessor
                from auth import get_service_client
                
                svc_client = get_service_client()
                processor = QueueProcessor(svc_client)
                
                # Consumes one job per loop to start with
                await processor.process_queue()
                
            except Exception as e:
                print(f"❌ Queue Loop Error: {e}")
                
    # Fire and forget tasks
    asyncio.create_task(start_sentinel_background())
    asyncio.create_task(start_queue_processor())

@app.get("/health")
@app.head("/health")
async def health_check():
    """Health check endpoint for UptimeRobot"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Session Middleware for SQLAdmin
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
)

# ============================================================================
# WebSocket Connection Manager
# ============================================================================
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, message: str, job_id: str):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    pass

manager = ConnectionManager()

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.biodockify.com",
        "https://biodockify.com",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_origin_regex="https://.*\.biodockify\.com", # Specific subdomains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(tools_router)
app.include_router(admin_router)
from export_routes import router as export_router
app.include_router(export_router)
app.include_router(evolution_router)
from routes.qsar import router as qsar_router
app.include_router(qsar_router)

app.include_router(batch_router)
app.include_router(jobs_router) # REGISTER JOBS ROUTER
app.include_router(feedback_router)
from services.target_prediction import router as target_prediction_router
app.include_router(target_prediction_router)

from routes.md import router as md_router
app.include_router(md_router)

from routes.md_analysis import router as md_analysis_router
app.include_router(md_analysis_router)

from routes.ranking import router as ranking_router
app.include_router(ranking_router)

from routes.benchmark import router as benchmark_router
app.include_router(benchmark_router)

from routes.user_system import router as user_router
app.include_router(user_router)

from routes.reporting import router as reporting_router
app.include_router(reporting_router)

# ============================================================================
# Configuration
# ============================================================================
security = HTTPBearer()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ============================================================================
# Auth Models & Routes
# ============================================================================

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    designation: str
    organization: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "BioDockify API",
        "version": "6.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest):
    try:
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "designation": request.designation,
                    "organization": request.organization
                }
            }
        })
        return {
            "user": {"id": response.user.id, "email": response.user.email},
            "message": "Account created successfully. Please verify your email."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")

@app.post("/auth/login")
async def login(request: LoginRequest):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        return {
            "access_token": response.session.access_token,
            "token_type": "bearer",
            "user": {"id": response.user.id, "email": response.user.email}
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

# ============================================================================
# Molecule Utilities (Shared)
# ============================================================================
# Kept here as they are generic molecule endpoints, or could move to a molecules.py router later.

@app.post("/molecules/drug-properties")
async def get_drug_properties(
    request: dict,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        smiles = request.get("smiles")
        if not smiles: raise HTTPException(400, "SMILES required")
        properties = DrugPropertiesCalculator().calculate_all(smiles)
        if "error" in properties: raise HTTPException(400, properties["error"])
        return properties
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")

@app.post("/molecules/bioactivity")
async def get_bioactivity(
    request: dict,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        from services.bioactivity_service import get_bioactivity_data
        smiles = request.get("smiles")
        if not smiles: raise HTTPException(400, "SMILES required")
        return get_bioactivity_data(smiles)
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
