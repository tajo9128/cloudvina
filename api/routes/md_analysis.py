from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from typing import Optional, List
import uuid
import os
from datetime import datetime

# Import Auth & Services
from auth import get_current_user, get_authenticated_client
from aws_services import invoke_md_stability_scorer

# Security & Router Setup
security = HTTPBearer()
router = APIRouter(prefix="/md-analysis", tags=["MD Stability Analysis (Isolated)"])

# --- Pydantic Models (Separate from Docking) ---

class MDAnalysisRequest(BaseModel):
    molecule_name: str
    rmsd: float
    rmsf: float
    docking_job_id: Optional[str] = None

class MDAnalysisResponse(BaseModel):
    job_id: str
    score: float
    status: str
    bucket_used: str

class MDJobRecord(BaseModel):
    id: str
    molecule_name: str
    rmsd: float
    rmsf: float
    md_score: Optional[float]
    status: str
    created_at: str

# --- Endpoints ---

@router.post("/calculate", response_model=MDAnalysisResponse)
async def calculate_stability(
    request: MDAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Invokes the Isolated AWS MD Scorer (Lambda).
    Saves results to the dedicated 'md_stability_jobs' table.
    """
    try:
        # 1. Generate ID
        job_id = str(uuid.uuid4())
        
        # 2. Call AWS Lambda (The Isolated Engine)
        # expected return: {'md_stability_score': 77.2, 'bucket_used': '...', 'status': 'success'}
        aws_result = invoke_md_stability_scorer(request.rmsd, request.rmsf)
        
        score = aws_result.get('md_stability_score')
        bucket = aws_result.get('bucket_used', 'unknown')
        
        if score is None:
            raise Exception("AWS Lambda did not return a score.")

        # 3. Save to Isolated Database (md_stability_jobs)
        auth_client = get_authenticated_client(credentials.credentials)
        
        db_record = {
            'id': job_id,
            'user_id': current_user.id,
            'molecule_name': request.molecule_name,
            'rmsd': request.rmsd,
            'rmsf': request.rmsf,
            'md_score': score,
            'bucket_used': bucket,
            'status': 'SUCCESS',
            'created_at': datetime.utcnow().isoformat(),
            'docking_job_id': request.docking_job_id
        }
        
        # Using the dedicated table as requested
        auth_client.table('md_stability_jobs').insert(db_record).execute()
        
        return {
            "job_id": job_id,
            "score": score,
            "status": "SUCCESS",
            "bucket_used": bucket
        }

    except Exception as e:
        print(f"MD Analysis Failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stability Analysis Failed: {str(e)}"
        )

@router.get("/history", response_model=List[MDJobRecord])
async def get_md_history(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Fetch history from the isolated MD table.
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        response = auth_client.table('md_stability_jobs')\
            .select('*')\
            .eq('user_id', current_user.id)\
            .order('created_at', desc=True)\
            .execute()
            
        return response.data if response.data else []
        

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch MD history: {str(e)}"
        )

# --- AWS Batch Simulation Endpoints ---

class MDSimulationStartRequest(BaseModel):
    molecule_name: str
    pdb_s3_key: str

class MDUploadRequest(BaseModel):
    filename: str

@router.post("/upload-pdb")
async def get_md_upload_url(
    request: MDUploadRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get Presigned URL to upload PDB for MD Simulation.
    """
    try:
        # Generate unique job ID for storage path
        job_id = str(uuid.uuid4())
        
        # We reuse the existing helper from aws_services
        # Note: generate_presigned_upload_urls returns a tuple for Receptor/Ligand.
        # Here we just need one file. We can reuse it or use s3_client directly.
        # For simplicity and isolation, we'll implement a simple one-off here or import a new helper.
        # Let's import the client directly or add a helper in code.
        # Better: Add a simple helper here to keep it self-contained? 
        # Actually, let's use the valid s3_client via aws_services import.
        from aws_services import s3_client, S3_BUCKET
        
        key = f"md-jobs/{job_id}/{request.filename}"
        
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET, 'Key': key, 'ContentType': 'chemical/x-pdb'},
            ExpiresIn=300
        )
        
        return {
            "upload_url": url,
            "s3_key": key,
            "job_id": job_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-simulation")
async def start_md_simulation(
    request: MDSimulationStartRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Trigger the AWS Batch MD Simulation (Separate Docker).
    """
    try:
        from aws_services import submit_md_simulation_job
        
        # 1. Submit to AWS Batch
        batch_job_id = submit_md_simulation_job(str(uuid.uuid4()), request.pdb_s3_key)
        
        # 2. Save Initial Record in DB
        auth_client = get_authenticated_client(credentials.credentials)
        job_id = batch_job_id # Use Batch ID or generating one? 
        # submit_md_simulation_job returns Batch ID. 
        # But we want a tracked ID in our DB.
        
        db_record = {
            'id': batch_job_id, # Using batch ID as primary key? Or UUID?
            # Let's use batch_job_id for tracking simplicity if it's unique enough (it is).
            'user_id': current_user.id,
            'molecule_name': request.molecule_name,
            'rmsd': 0.0, # Placeholder
            'rmsf': 0.0, # Placeholder
            'md_score': None,
            'status': 'SUBMITTED', # New status for Batch
            'created_at': datetime.utcnow().isoformat()
        }
        
        auth_client.table('md_stability_jobs').insert(db_record).execute()
        
        return {
            "job_id": batch_job_id,
            "status": "SUBMITTED",
            "message": "Simulation started on AWS Batch."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
