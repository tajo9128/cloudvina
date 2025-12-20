from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth import get_current_user, get_authenticated_client
from services.accuracy_service import accuracy_service
import csv
import io
import json

router = APIRouter(prefix="/tools/benchmark", tags=["Tools"])
security = HTTPBearer()

@router.post("/analyze")
async def analyze_batch_accuracy(
    batch_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Analyze accuracy of a batch against an uploaded Reference CSV.
    CSV Format: name, value
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        # 1. Read CSV
        content = await file.read()
        decoded = content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded))
        
        # Validate CSV headers
        if not 'name' in reader.fieldnames or not 'value' in reader.fieldnames:
             raise HTTPException(status_code=400, detail="CSV must have 'name' and 'value' columns")
             
        reference_data = list(reader)
        
        # 2. Fetch Batch Jobs
        # We need ALL jobs, not paginated
        jobs_res = auth_client.table('jobs').select('ligand_filename, compound_name, binding_affinity, docking_results, status').eq('batch_id', batch_id).execute()
        batch_jobs = jobs_res.data
        
        if not batch_jobs:
            raise HTTPException(status_code=404, detail="Batch not found or empty")

        # 3. Calculate Metrics
        result = accuracy_service.match_and_analyze(batch_jobs, reference_data)
        
        # 4. Save Analysis to DB
        new_analysis = {
            "user_id": current_user['id'],
            "batch_id": batch_id,
            "name": name,
            "dataset_filename": file.filename,
            "metrics": result['metrics'],
            "plot_data": result['plot_data'] # Warning: Could be large, but usually < 100kb for < 1000 points
        }
        
        db_res = auth_client.table("benchmark_analyses").insert(new_analysis).execute()
        
        return db_res.data[0]
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_benchmark_history(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get past analyses"""
    auth_client = get_authenticated_client(credentials.credentials)
    res = auth_client.table("benchmark_analyses") \
        .select("id, name, batch_id, created_at, metrics, dataset_filename") \
        .eq("user_id", current_user['id']) \
        .order("created_at", desc=True) \
        .execute()
    return res.data

@router.get("/stats")
async def get_accuracy_stats(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get aggregated accuracy statistics for the user"""
    auth_client = get_authenticated_client(credentials.credentials)
    stats = await accuracy_service.get_user_stats(auth_client, current_user['id'])
    return stats

@router.get("/{analysis_id}")
async def get_benchmark_details(
    analysis_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Get full details including plot data"""
    auth_client = get_authenticated_client(credentials.credentials)
    res = auth_client.table("benchmark_analyses") \
        .select("*") \
        .eq("id", analysis_id) \
        .single() \
        .execute()
    return res.data
