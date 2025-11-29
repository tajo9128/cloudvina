from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from auth import get_current_user, get_authenticated_client
from services.export import ExportService

router = APIRouter(tags=["Export"])
security = HTTPBearer()

@router.get("/jobs/{job_id}/export/{format}")
async def export_single_job(
    job_id: str,
    format: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Export a single job to CSV, JSON, or PDF
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Fetch job details
        response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = response.data[0]
        
        if format == 'csv':
            return ExportService.export_jobs_csv([job])
        elif format == 'json':
            return ExportService.export_jobs_json([job])
        elif format == 'pdf':
            return ExportService.export_job_pdf(job)
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use csv, json, or pdf")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{format}")
async def export_all_jobs(
    format: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Export all user jobs to CSV or JSON
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Fetch all jobs
        response = auth_client.table('jobs').select('*').eq('user_id', current_user.id).order('created_at', desc=True).execute()
        jobs = response.data
        
        if format == 'csv':
            return ExportService.export_jobs_csv(jobs)
        elif format == 'json':
            return ExportService.export_jobs_json(jobs)
        elif format == 'pdf':
            raise HTTPException(status_code=400, detail="PDF export is only available for single jobs")
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use csv or json")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
