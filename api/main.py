"""
BioDockify API - FastAPI Backend
Handles authentication, job submission, and AWS Batch orchestration
"""
from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
import os
from datetime import datetime
import uuid

# Import auth and AWS utilities
from auth import supabase, get_current_user
from aws_services import (
    generate_presigned_upload_urls,
    generate_presigned_download_url,
    submit_batch_job,
    get_batch_job_status
)
from tools import router as tools_router
from admin import router as admin_router

# NEW: Import SQLAdmin setup
from admin_sqladmin import setup_admin

app = FastAPI(
    title="BioDockify API",
    description="Molecular docking as a service with AutoDock Vina",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

@app.on_event("startup")
async def startup_event():
    print("="*50)
    print("🚀 BioDockify API v1.1.0 - Deployed at " + datetime.utcnow().isoformat())
    print("✓ Fix: RateLimiter Admin API call removed")
    print("✓ Fix: Signup handled by DB Triggers")
    print("="*50)

# NEW: Add session middleware for SQLAdmin authentication
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://biodockify.com",
        "https://www.biodockify.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Activity logging middleware for audit trail
from middleware.activity_logger import ActivityLoggerMiddleware
app.add_middleware(ActivityLoggerMiddleware)

# NEW: Setup SQLAdmin (Django-style admin panel at /sqladmin)
admin = setup_admin(app)

app.include_router(tools_router)
app.include_router(admin_router)

# ============================================================================
# Configuration
# ============================================================================

# Security
security = HTTPBearer()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")

# ============================================================================
# Pydantic Models
# ============================================================================

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    designation: str
    organization: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class DockingConfig(BaseModel):
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8

class JobSubmitRequest(BaseModel):
    receptor_filename: str
    ligand_filename: str
    config: Optional[DockingConfig] = DockingConfig()

class JobStartRequest(BaseModel):
    pass  # No body needed, job_id comes from path

# ============================================================================
# Health Check Routes
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "BioDockify API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "checks": {
            "supabase": "configured" if SUPABASE_URL else "missing",
            "aws": "configured" if os.getenv("AWS_ACCESS_KEY_ID") else "missing"
        }
    }

# ============================================================================
# Authentication Routes
# ============================================================================

@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest):
    """
    Create a new user account with verification required
    """
    try:
        # Create auth user
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
        
        if response.user:
            # User setup is now handled by Supabase Database Triggers (handle_new_user)
            # This avoids RLS issues and ensures atomicity
            pass
        
        return {
            "user": {
                "id": response.user.id,
                "email": response.user.email
            },
            "message": "Account created successfully. Please verify your email (check your inbox) before submitting jobs."
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signup failed: {str(e)}"
        )

@app.post("/auth/login")
async def login(request: LoginRequest):
    """
    Login and receive JWT token
    """
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        return {
            "access_token": response.session.access_token,
            "token_type": "bearer",
            "user": {
                "id": response.user.id,
                "email": response.user.email
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Login failed: {str(e)}"
        )

# ============================================================================
# Job Management Routes
# ============================================================================

@app.post("/jobs/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    request: JobSubmitRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security) # Get raw token
):
    """
    Submit a new docking job and get upload URLs
    Includes rate limiting: 3 jobs/day for free users
    Requires email verification
    """
    try:
        # Import rate limiter
        from services.rate_limiter import RateLimiter
        from auth import get_authenticated_client
        
        # Create authenticated client for this request
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Check if user can submit job (verification + rate limit)
        # Use auth_client so RLS policies work for user_profiles
        eligibility = await RateLimiter.check_can_submit(auth_client, current_user)
        
        if not eligibility['allowed']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": eligibility['message'],
                    "reason": eligibility.get('reason'),
                    "jobs_today": eligibility.get('jobs_today'),
                    "limit": eligibility.get('limit')
                }
            )
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Generate pre-signed upload URLs
        receptor_url, ligand_url, receptor_key, ligand_key = generate_presigned_upload_urls(
            job_id,
            request.receptor_filename,
            request.ligand_filename
        )
        
        # Create job record in database
        auth_client.table('jobs').insert({
            'id': job_id,
            'user_id': current_user.id,
            'status': 'PENDING',
            'receptor_s3_key': receptor_key,
            'ligand_s3_key': ligand_key
        }).execute()
        
        # Increment daily usage (for rate limiting)
        await RateLimiter.increment_usage(supabase, current_user.id)
        
        return {
            "job_id": job_id,
            "upload_urls": {
                "receptor": receptor_url,
                "ligand": ligand_url
            },
            "message": "Upload your files to the provided URLs, then call /jobs/{job_id}/start",
            "remaining_jobs_today": eligibility.get('remaining_today')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )

@app.post("/jobs/{job_id}/start")
async def start_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Start the docking job after files are uploaded
    """
    try:
        from auth import get_authenticated_client
        auth_client = get_authenticated_client(credentials.credentials)

        # Verify job belongs to user
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = job_response.data[0]
        
        # Submit to AWS Batch
        batch_job_id = submit_batch_job(
            job_id,
            job['receptor_s3_key'],
            job['ligand_s3_key']
        )
        
        # Update job status
        auth_client.table('jobs').update({
            'status': 'SUBMITTED',
            'batch_job_id': batch_job_id
        }).eq('id', job_id).execute()
        
        return {
            "job_id": job_id,
            "status": "SUBMITTED",
            "message": "Job started successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start job: {str(e)}"
        )

@app.get("/jobs/{job_id}", response_model=dict)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get job status and results
    """
    try:
        from auth import get_authenticated_client
        auth_client = get_authenticated_client(credentials.credentials)

        # Get job from database
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = job_response.data[0]
        
        # If job is running, check AWS Batch status
        if job['status'] in ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING']:
            batch_status = get_batch_job_status(job['batch_job_id'])
            
            # Update DB with latest status
            auth_client.table('jobs').update({
                'status': batch_status['status']
            }).eq('id', job_id).execute()
            
            job['status'] = batch_status['status']
        
        # Generate download URLs if succeeded
        download_urls = {}
        if job['status'] == 'SUCCEEDED':
            download_urls = {
                'output': generate_presigned_download_url(job_id, 'output.pdbqt'),
                'log': generate_presigned_download_url(job_id, 'log.txt')
            }
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "created_at": job['created_at'],
            "completed_at": job.get('completed_at'),
            "binding_affinity": job.get('binding_affinity'),
            "download_urls": download_urls if download_urls else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

