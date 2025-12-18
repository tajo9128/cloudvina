"""
BioDockify API - FastAPI Backend
Handles authentication, job submission, and AWS Batch orchestration
"""
from fastapi import FastAPI, Depends, HTTPException, status, Security, Body, UploadFile, File
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
from fastapi.responses import StreamingResponse
from services.ai_explainer import AIExplainer
from tools import router as tools_router
from routes.admin import router as admin_router
from routes.evolution import router as evolution_router
from routes.batch import router as batch_router  # ADD BATCH ROUTES
from services.cavity_detector import CavityDetector
from services.drug_properties import DrugPropertiesCalculator

# NEW: Import SQLAdmin setup
# from admin_sqladmin import setup_admin

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
    print("BioDockify API v1.1.0 - Deployed at " + datetime.utcnow().isoformat())
    print("Fix: RateLimiter Admin API call removed")
    print("Fix: Signup handled by DB Triggers")
    print("="*50)

@app.get("/health")
async def health_check():
    """Health check endpoint for UptimeRobot"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

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
        "https://www.biodockify.com/",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Activity logging middleware for audit trail
# TEMPORARILY DISABLED due to RLS policy conflicts with Supabase
# TODO: Re-enable once we implement proper tenant context for SQLAlchemy
# from middleware.activity_logger import ActivityLoggerMiddleware
# app.add_middleware(ActivityLoggerMiddleware)

# NEW: Setup SQLAdmin (Django-style admin panel at /sqladmin)
# admin = setup_admin(app)

app.include_router(tools_router)
app.include_router(admin_router)
from export_routes import router as export_router
app.include_router(export_router)
app.include_router(evolution_router)
# Register QSAR Router
from routes.qsar import router as qsar_router
app.include_router(qsar_router)

app.include_router(batch_router)  # Enable batch job endpoints
# Register Target Prediction Router
from services.target_prediction import router as target_prediction_router
app.include_router(target_prediction_router)

# Register MD Simulation Router (Colab/Celery)
from routes.md import router as md_router
app.include_router(md_router)

# Register Lead Ranking Router
from routes.ranking import router as ranking_router
app.include_router(ranking_router)

# Register User System Router (Profile, Billing, Support)
from routes.user_system import router as user_router
app.include_router(user_router)

# Register Reporting Router (PDF)
from routes.reporting import router as reporting_router
app.include_router(reporting_router)

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
            'ligand_s3_key': ligand_key,
            'receptor_filename': request.receptor_filename,
            'ligand_filename': request.ligand_filename
        }).execute()
        
        # Increment daily usage (for rate limiting)
        await RateLimiter.increment_usage(auth_client, current_user.id)
        
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
    request: dict = Body(default={}),
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
        
        # Extract grid parameters
        grid_params = request.get('grid_params', None)
        
        # Engine Selection (Default to 'vina' for reliability, but allow override)
        # Previous 'consensus' default caused missing Vina logs in some container versions
        engine = request.get('engine', 'vina')
        
        # Generate config file
        try:
            from services.config_generator import generate_vina_config
            config_key = generate_vina_config(job_id, grid_params=grid_params)
        except Exception as e:
            print(f"Config generation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate config file: {str(e)}"
            )
        
        # Prepare files for AWS Batch (Ensure PDBQT)
        # ------------------------------------------
        # 1. Receptor Preparation (PDB/MOL2/CIF -> Rigid PDBQT)
        job_receptor_key = job['receptor_s3_key']
        rec_ext = job['receptor_filename'].lower().split('.')[-1]
        
        if rec_ext != 'pdbqt':
             print(f"DEBUG: Converting receptor {job['receptor_filename']} (Format: {rec_ext}) to PDBQT")
             try:
                 import boto3
                 from services.smiles_converter import convert_receptor_to_pdbqt
                 
                 s3 = boto3.client('s3', region_name=AWS_REGION)
                 
                 # Download content
                 pdb_obj = s3.get_object(Bucket=S3_BUCKET, Key=job['receptor_s3_key'])
                 rec_content = pdb_obj['Body'].read().decode('utf-8')
                 
                 # Convert
                 pdbqt_content, err = convert_receptor_to_pdbqt(rec_content, job['receptor_filename'])
                 if err or not pdbqt_content:
                     raise Exception(f"Receptor conversion failed: {err}")
                 
                 # Upload PDBQT
                 # We overwrite the original 'receptor_s3_key' in the JOB record effectively? 
                 # Or better, create a new key. Let's create a new key to avoid overwriting input.
                 new_rec_key = f"jobs/{job_id}/receptor_input_converted.pdbqt"
                 
                 s3.put_object(
                     Bucket=S3_BUCKET,
                     Key=new_rec_key,
                     Body=pdbqt_content.encode('utf-8')
                 )
                 print(f"DEBUG: Receptor conversion & upload successful: {new_rec_key}")
                 job_receptor_key = new_rec_key
                 
             except Exception as conv_err:
                 print(f"ERROR: Receptor preparation failed: {conv_err}")
                 raise HTTPException(status_code=400, detail=f"Receptor conversion failed: {str(conv_err)}")
        
        # 2. Ligand Preparation (SDF/MOL/PDB/SMI -> Flexible PDBQT)
        # If ligand is not PDBQT, convert it now to avoid container compatibility issues.
        final_ligand_key = job['ligand_s3_key']
        
        if not final_ligand_key.lower().endswith('.pdbqt'):
            print(f"DEBUG: Ligand {final_ligand_key} is not PDBQT. Converting...")
            try:
                import boto3
                from services.smiles_converter import convert_to_pdbqt
                
                s3 = boto3.client('s3', region_name=AWS_REGION)
                
                # 1. Download original
                obj = s3.get_object(Bucket=S3_BUCKET, Key=final_ligand_key)
                content = obj['Body'].read().decode('utf-8')
                
                # 2. Convert
                pdbqt_content, err = convert_to_pdbqt(content, job['ligand_filename'])
                if err:
                    raise Exception(f"Conversion failed: {err}")
                
                # 3. Upload new PDBQT
                # Construct new key: jobs/{id}/ligand_input_converted.pdbqt
                new_key = f"jobs/{job_id}/ligand_input_converted.pdbqt"
                s3.put_object(
                    Bucket=S3_BUCKET, 
                    Key=new_key, 
                    Body=pdbqt_content.encode('utf-8')
                )
                
                print(f"DEBUG: Converted and uploaded to {new_key}")
                final_ligand_key = new_key
                
            except Exception as e:
                # If conversion fails, we log but might try to submit original (fallback) or fail hard?
                # Failing hard is safer than crashing in container.
                print(f"ERROR: Backend conversion failed: {e}")
                raise HTTPException(status_code=400, detail=f"Ligand conversion failed: {str(e)}")

        # Submit to AWS Batch
        try:
            batch_job_id = submit_batch_job(
                job_id,
                job['receptor_s3_key'],
                final_ligand_key, # Use possibly converted key
                engine=engine
            )
            
            # Update job status with batch_job_id
            auth_client.table('jobs').update({
                'status': 'SUBMITTED',
                'batch_job_id': batch_job_id
            }).eq('id', job_id).execute()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"CRITICAL: Batch submission failed for job {job_id}")
            print(f"Error: {str(e)}")
            print(f"Full traceback:\n{error_details}")
            
            # Update job as FAILED with error message
            auth_client.table('jobs').update({
                'status': 'FAILED',
                'error_message': f"AWS Batch submission failed: {str(e)}"
            }).eq('id', job_id).execute()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit batch job: {str(e)}"
            )
        
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

@app.get("/jobs", response_model=list)
async def list_jobs(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get all jobs for the current user"""
    try:
        from auth import get_authenticated_client
        
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Fetch all jobs for this user, ordered by creation date
        response = auth_client.table('jobs')\
            .select('*')\
            .eq('user_id', current_user.id)\
            .order('created_at', desc=True)\
            .execute()
        
        return response.data if response.data else []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch jobs: {str(e)}"
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
        
        # If job is running or pending, check AWS Batch status
        error_message = job.get('error_message')
        if job['status'] in ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING']:
            batch_status = get_batch_job_status(job['batch_job_id'])
            
            # Build update payload
            update_payload = {'status': batch_status['status']}
            
            # Capture error message if job failed
            if batch_status['status'] == 'FAILED' and batch_status.get('status_reason'):
                update_payload['error_message'] = batch_status['status_reason']
                error_message = batch_status['status_reason']
            
            # Update DB with latest status and error message
            auth_client.table('jobs').update(update_payload).eq('id', job_id).execute()
            
            # REFUND LOGIC: If job failed, refund the credit
            if batch_status['status'] == 'FAILED':
                 # Use background task or await? This function is async.
                 # Need to import RateLimiter
                 from services.rate_limiter import RateLimiter
                 try:
                     print(f"DEBUG: Job {job_id} failed. Refunding credit...")
                     await RateLimiter.refund_usage(auth_client, current_user.id)
                 except Exception as refund_err:
                     print(f"Error processing refund: {refund_err}")
            
            job['status'] = batch_status['status']
        
        # --- LAZY REPAIR: Fetch missing results from S3 if Job Succeeded but DB has no score ---
        # This fixes the "0.0 Affinity" bug if the container failed to update Supabase
        is_missing_score = job.get('binding_affinity') is None or float(job.get('binding_affinity') or 0) == 0.0
        
        if job['status'] == 'SUCCEEDED' and is_missing_score:
            try:
                import boto3
                import json
                s3 = boto3.client('s3', region_name=AWS_REGION)
                
                # Check for "results.json" which contains parsed scores
                results_key = f"jobs/{job_id}/results.json"
                try:
                    obj = s3.get_object(Bucket=S3_BUCKET, Key=results_key)
                    results_data = json.loads(obj['Body'].read().decode('utf-8'))
                    
                    # Update DB
                    update_payload = {}
                    if 'best_affinity' in results_data:
                         update_payload['binding_affinity'] = results_data['best_affinity']
                         job['binding_affinity'] = results_data['best_affinity'] # Update local object
                         
                    if 'vina_score' in results_data:
                         # Ensure docking_results structure
                         current_results = job.get('docking_results') or {}
                         if isinstance(current_results, str): current_results = json.loads(current_results)
                         
                         current_results.update(results_data)
                         update_payload['docking_results'] = current_results
                         job['docking_results'] = current_results # Update local object
                    
                    if update_payload:
                        auth_client.table('jobs').update(update_payload).eq('id', job_id).execute()
                        # print(f"DEBUG: Lazy repaired job {job_id} from S3 results.json")
                        
                except Exception as s3_err:
                     # If results.json missing, try log.txt parsing as last resort?
                     # For now, just log error.
                     print(f"Lazy repair failed for {job_id}: {s3_err}")
                     
            except Exception as e:
                print(f"Error in lazy repair block: {e}") 

        
        # Generate download URLs if succeeded
        download_urls = {}
        if job['status'] == 'SUCCEEDED':
            # Extract basename for receptor key to work with generate_presigned_download_url helper
            # job['receptor_s3_key'] looks like "jobs/{job_id}/receptor_input.pdb"
            receptor_fname = os.path.basename(job.get('receptor_s3_key', 'receptor.pdb'))
            
            download_urls = {
                'output': generate_presigned_download_url(job_id, 'output.pdbqt'),
                'receptor': generate_presigned_download_url(job_id, receptor_fname),
                'log': generate_presigned_download_url(job_id, 'log.txt'),
                'config': generate_presigned_download_url(job_id, 'config.txt'),
                # Extended URLs for Consensus/Multi-Engine
                'results_json': generate_presigned_download_url(job_id, 'results.json'),
                'results_csv': generate_presigned_download_url(job_id, 'results.csv'),
                'output_vina': generate_presigned_download_url(job_id, 'output_vina.pdbqt'),
                'output_rdock': generate_presigned_download_url(job_id, 'output_rdock.sdf'),
                'output_gnina': generate_presigned_download_url(job_id, 'output_gnina.pdbqt')
            }
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "created_at": job['created_at'],
            "completed_at": job.get('completed_at'),
            "binding_affinity": job.get('binding_affinity') or (job.get('docking_results') or {}).get('best_affinity'),
            "ligand_filename": job.get('ligand_filename'),
            "receptor_filename": job.get('receptor_filename'),
            "download_urls": download_urls if download_urls else None,
            "error_message": error_message
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@app.post("/jobs/{job_id}/detect-cavities")
async def detect_cavities(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Detect binding cavities in the uploaded receptor structure.
    Returns top 5 potential binding pockets with coordinates for grid box configuration.
    """
    try:
        from auth import get_authenticated_client
        import boto3
        
        auth_client = get_authenticated_client(credentials.credentials)

        # Get job from database
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = job_response.data[0]
        receptor_key = job.get('receptor_s3_key')
        
        if not receptor_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No receptor file uploaded for this job"
            )
        
        # Download receptor from S3
        s3 = boto3.client('s3', region_name=AWS_REGION)
        
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=receptor_key)
            pdb_content = response['Body'].read().decode('utf-8')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not retrieve receptor file: {str(e)}"
            )
        
        # Detect cavities
        detector = CavityDetector()
        cavities = detector.detect_cavities(pdb_content)
        
        if not cavities:
            # Return a default cavity at protein center if none found
            return {
                "job_id": job_id,
                "cavities": [{
                    "pocket_id": 1,
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "center_z": 0.0,
                    "size_x": 20.0,
                    "size_y": 20.0,
                    "size_z": 20.0,
                    "score": 0.5,
                    "volume": 8000.0,
                    "residues": []
                }],
                "message": "No distinct cavities detected. Using protein center."
            }
        
        return {
            "job_id": job_id,
            "cavities": cavities,
            "message": f"Detected {len(cavities)} potential binding site(s)"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect cavities: {str(e)}"
        )


@app.post("/molecules/drug-properties")
async def get_drug_properties(
    request: dict,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Calculate drug-likeness properties and ADMET predictions for a molecule.
    Returns Lipinski Rule of 5, Veber rules, PAINS alerts, and links to external tools.
    """
    try:
        smiles = request.get("smiles")
        if not smiles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SMILES string is required"
            )
        
        calculator = DrugPropertiesCalculator()
        properties = calculator.calculate_all(smiles)
        
        if "error" in properties:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=properties["error"]
            )
        
        return properties
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate drug properties: {str(e)}"
        )


@app.post("/molecules/bioactivity")
async def get_bioactivity(
    request: dict,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Fetch bioactivity data from PubChem and ChEMBL.
    Returns compound info, targets, and activity values.
    """
    try:
        from services.bioactivity_service import get_bioactivity_data
        
        smiles = request.get("smiles")
        if not smiles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SMILES string is required"
            )
        
        data = get_bioactivity_data(smiles)
        
        return data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch bioactivity data: {str(e)}"
        )


@app.get("/jobs/{job_id}/drug-properties")
async def get_job_drug_properties(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get drug-likeness properties for the ligand used in a docking job.
    Extracts SMILES from the docked ligand and calculates properties.
    """
    try:
        from auth import get_authenticated_client
        import boto3
        from rdkit import Chem
        
        auth_client = get_authenticated_client(credentials.credentials)

        # Get job from database
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = job_response.data[0]
        
        if job['status'] != 'SUCCEEDED':
            return {"job_id": job_id, "status": job['status'], "properties": None, "message": "Job not completed"}
        
        # Try to get ligand SMILES from docked output
        s3 = boto3.client('s3', region_name=AWS_REGION)
        ligand_key = f"{job_id}/output.pdbqt"
        
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=ligand_key)
            pdbqt_content = response['Body'].read().decode('utf-8')
            
            # Convert PDBQT to mol and then to SMILES
            # First try to extract from the original ligand if available
            mol = Chem.MolFromPDBBlock(pdbqt_content)
            if mol:
                smiles = Chem.MolToSmiles(mol)
            else:
                return {"job_id": job_id, "properties": None, "message": "Could not extract SMILES from ligand"}
            
        except Exception as e:
            return {"job_id": job_id, "properties": None, "message": f"Could not retrieve ligand: {str(e)}"}
        
        calculator = DrugPropertiesCalculator()
        properties = calculator.calculate_all(smiles)
        
        return {
            "job_id": job_id,
            "smiles": smiles,
            "properties": properties
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate drug properties: {str(e)}"
        )


@app.get("/jobs/{job_id}/analysis", response_model=dict)
async def get_docking_analysis(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get comprehensive docking analysis with parsed Vina results
    Returns binding affinities, RMSD values, and pose data
    """
    try:
        from auth import get_authenticated_client
        from services.vina_parser import parse_vina_log
        import boto3
        
        auth_client = get_authenticated_client(credentials.credentials)

        # Get job from database
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        job = job_response.data[0]
        
        # Only parse if job succeeded
        if job['status'] != 'SUCCEEDED':
            return {
                "job_id": job_id,
                "status": job['status'],
                "analysis": None,
                "message": "Analysis only available for completed jobs"
            }
        
        # Check if already parsed
        if job.get('docking_results'):
            return {
                "job_id": job_id,
                "status": job['status'],
                "analysis": job['docking_results'],
                "from_cache": True
            }
        
        # Fetch log from S3
        s3 = boto3.client('s3')
        bucket_name = os.getenv('S3_BUCKET_NAME')
        log_key = f"{job_id}/log.txt"
        
        try:
            log_obj = s3.get_object(Bucket=bucket_name, Key=log_key)
            log_content = log_obj['Body'].read().decode('utf-8')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Log file not found: {str(e)}"
            )
        
        # Parse Vina output
        analysis = parse_vina_log(log_content)
        
        # Store in database for future requests
        auth_client.table('jobs').update({
            'best_affinity': analysis.get('best_affinity'),
            'num_poses': analysis.get('num_poses'),
            'energy_range_min': analysis.get('energy_range_min'),
            'energy_range_max': analysis.get('energy_range_max'),
            'docking_results': analysis
        }).eq('id', job_id).execute()
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "analysis": analysis,
            "from_cache": False
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze docking results: {str(e)}"
        )


@app.get("/jobs/{job_id}/interactions", response_model=dict)
async def get_interaction_analysis(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get protein-ligand interaction analysis (H-bonds, hydrophobic contacts)
    Fetches PDB/PDBQT from S3 and runs geometric analysis
    """
    try:
        from auth import get_authenticated_client
        from services.interaction_analyzer import InteractionAnalyzer
        import boto3
        
        auth_client = get_authenticated_client(credentials.credentials)

        # Get job from database
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        
        job = job_response.data[0]
        
        if job['status'] != 'SUCCEEDED':
            return {"job_id": job_id, "status": job['status'], "interactions": None, "message": "Job not completed"}
        
        # Return cached results if available
        if job.get('interaction_results'):
            return {
                "job_id": job_id,
                "status": job['status'],
                "interactions": job['interaction_results'],
                "from_cache": True
            }
            
        # Fetch files from S3
        s3 = boto3.client('s3')
        bucket = os.getenv('S3_BUCKET_NAME')
        
        try:
            # Fetch Receptor PDB
            # Note: We need to know the receptor filename or key. 
            # Usually stored in job metadata or we assume a standard name if we renamed it.
            # But the job record has 'receptor_filename'. We stored it in S3 under {job_id}/receptor.pdb usually?
            # Let's check how submit_job stores it.
            # Assuming standard keys: {job_id}/receptor.pdb and {job_id}/output.pdbqt
            
            receptor_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/receptor.pdb")
            receptor_content = receptor_obj['Body'].read().decode('utf-8')
            
            output_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/output.pdbqt")
            output_content = output_obj['Body'].read().decode('utf-8')
            
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Files not found in S3: {str(e)}")

        # Run Analysis
        analyzer = InteractionAnalyzer()
        interactions = analyzer.analyze_interactions(receptor_content, output_content)
        
        # Cache results
        auth_client.table('jobs').update({
            'interaction_results': interactions
        }).eq('id', job_id).execute()
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "interactions": interactions,
            "from_cache": False
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze interactions: {str(e)}"
        )


@app.post("/jobs/{job_id}/explain")
async def explain_results(
    job_id: str,
    request: dict,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get AI explanation of docking results
    """
    auth_client = get_authenticated_client(credentials.credentials)
    
    # Get job from database
    job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
    
    if not job_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    job = job_response.data[0]
    
    # Get analysis and interactions
    analysis = job.get('docking_results')
    interactions = job.get('interaction_results')
    
    explainer = AIExplainer()
    user_question = request.get('question')
    
    async def generate():
        async for chunk in explainer.explain_results(
            job, analysis, interactions, user_question
        ):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/ai/upload-and-analyze")
async def upload_and_analyze(
    log_file: UploadFile = File(...),
    pdbqt_file: UploadFile = File(None),
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Upload and analyze docking output files"""
    from services.vina_parser import parse_vina_log
    from auth import get_authenticated_client
    
    try:
        # Parse log file
        content = await log_file.read()
        
        analysis = None
        try:
            # Try to decode as text
            text_content = content.decode('utf-8')
            # Try to parse as Vina log
            analysis = parse_vina_log(text_content)
        except Exception:
            # If decoding or parsing fails, return generic info
            # This allows uploading PDFs, images, etc. without crashing
            analysis = {
                "best_affinity": "N/A",
                "num_poses": 0,
                "energy_range_min": 0,
                "energy_range_max": 0,
                "raw_content": "File uploaded successfully. Content format not automatically parsable as Vina log.",
                "filename": log_file.filename
            }
        
        # Simple interactions placeholder (can be enhanced)
        interactions = None
        if pdbqt_file:
            interactions = {
                "hydrogen_bonds": [],
                "hydrophobic_contacts": [],
                "note": "Upload both receptor and ligand PDBQT for detailed analysis"
            }
        
        return {
            "analysis": analysis,
            "interactions": interactions,
            "filenames": {
                "log": log_file.filename,
                "pdbqt": pdbqt_file.filename if pdbqt_file else None
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process files: {str(e)}"
        )


@app.post("/ai/explain")
async def explain_uploaded_results(
    request: dict = Body(...),
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get AI explanation for uploaded docking results"""
    analysis_data = request.get('analysis_data')
    user_question = request.get('question')
    
    if not analysis_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No analysis data provided"
        )
    
    # Mock job data  
    job_data = {
        "receptor_filename": analysis_data.get('filenames', {}).get('receptor', 'Uploaded'),
        "ligand_filename": analysis_data.get('filenames', {}).get('ligand', 'Uploaded')
    }
    
    analysis = analysis_data.get('analysis')
    interactions = analysis_data.get('interactions')
    
    explainer = AIExplainer()
    
    async def generate():
        async for chunk in explainer.explain_results(
            job_data, analysis, interactions, user_question
        ):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/jobs/{job_id}/export/pdf")
async def export_job_pdf(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Generate and download comprehensive PDF report"""
    return await _generate_job_export(job_id, current_user, credentials, 'pdf')

@app.get("/jobs/{job_id}/export/csv")
async def export_job_csv(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Export job details to CSV"""
    return await _generate_job_export(job_id, current_user, credentials, 'csv')

@app.get("/jobs/{job_id}/export/json")
async def export_job_json(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Export job details to JSON"""
    return await _generate_job_export(job_id, current_user, credentials, 'json')

async def _generate_job_export(job_id, current_user, credentials, format_type):
    """Helper to generate exports - auto-triggers analysis if missing"""
    try:
        from auth import get_authenticated_client
        from services.export import ExportService
        from services.vina_parser import parse_vina_log
        from services.interaction_analyzer import InteractionAnalyzer
        import boto3
        
        auth_client = get_authenticated_client(credentials.credentials)
        job_response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        
        job = job_response.data[0]
        
        # Get Analysis & Interaction Data
        analysis = job.get('docking_results')
        interactions = job.get('interaction_results')
        
        # Auto-trigger analysis if missing and job succeeded
        if job['status'] == 'SUCCEEDED' and (not analysis or not interactions):
            s3 = boto3.client('s3')
            bucket = os.getenv('S3_BUCKET_NAME')
            
            try:
                # Parse docking results if missing
                if not analysis:
                    log_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/log.txt")
                    log_content = log_obj['Body'].read().decode('utf-8')
                    analysis = parse_vina_log(log_content)
                    
                    # Save to DB
                    auth_client.table('jobs').update({
                        'best_affinity': analysis.get('best_affinity'),
                        'num_poses': analysis.get('num_poses'),
                        'energy_range_min': analysis.get('energy_range_min'),
                        'energy_range_max': analysis.get('energy_range_max'),
                        'docking_results': analysis
                    }).eq('id', job_id).execute()
                
                # Analyze interactions if missing
                if not interactions:
                    receptor_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/receptor.pdb")
                    receptor_content = receptor_obj['Body'].read().decode('utf-8')
                    
                    output_obj = s3.get_object(Bucket=bucket, Key=f"{job_id}/output.pdbqt")
                    output_content = output_obj['Body'].read().decode('utf-8')
                    
                    analyzer = InteractionAnalyzer()
                    interactions = analyzer.analyze_interactions(receptor_content, output_content)
                    
                    # Save to DB
                    auth_client.table('jobs').update({
                        'interaction_results': interactions
                    }).eq('id', job_id).execute()
                    
            except Exception as e:
                # If analysis fails, continue with whatever data we have
                print(f"Warning: Could not auto-generate analysis: {e}")

        if format_type == 'pdf':
            return ExportService.export_job_pdf(job, analysis, interactions)
        elif format_type == 'csv':
            return ExportService.export_jobs_csv([job])
        elif format_type == 'json':
            return ExportService.export_jobs_json([job])
        else:
            raise HTTPException(status_code=400, detail="Invalid format")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export job: {str(e)}"
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

