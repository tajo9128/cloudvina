
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from services.report_generator import ReportGenerator
import logging
import boto3
import os
from auth import get_current_user
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, HTTPBearer
from fastapi import Security

# AWS Config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")
s3_client = boto3.client('s3', region_name=AWS_REGION)
security = HTTPBearer()

router = APIRouter(prefix="/ranking", tags=["ranking"])
logger = logging.getLogger(__name__)

# Re-use schema or define simplified one
class ReportRequest(BaseModel):
    hits: List[Dict[str, Any]]
    project_name: Optional[str] = "BioDockify Project"

@router.post("/report")
async def generate_pdf_report(request: ReportRequest):
    """
    Generate a PDF report for the provided list of hits.
    """
    try:
        generator = ReportGenerator()
        pdf_buffer = generator.generate_report(request.hits, request.project_name)
        
        return StreamingResponse(
            pdf_buffer, 
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report.pdf"}
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}/export/pymol")
async def generate_pymol_export(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Generate a PyMOL script for a specific job.
    """
    try:
        from services.report_generator import ReportGenerator
        from auth import get_authenticated_client
        
        auth_client = get_authenticated_client(credentials.credentials)
        
        # 1. Fetch Job
        job_res = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).single().execute()
        job = job_res.data
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 2. Generate Presigned URLs (Fresh ones)
        # We need read access to the receptor and ligand files
        try:
            receptor_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': job['receptor_s3_key']},
                ExpiresIn=3600
            )
            
            ligand_url = s3_client.generate_presigned_url(
                 'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': job['ligand_s3_key']},
                ExpiresIn=3600
            )
        except Exception as s3_err:
             logger.error(f"S3 URL generation failed: {s3_err}")
             raise HTTPException(status_code=500, detail="Failed to generate file URLs")

        generator = ReportGenerator()
        buffer = generator.generate_pymol_script(job_id, receptor_url, ligand_url)
        
        return StreamingResponse(
            buffer,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=visualization_{job_id}.pml"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PyMOL generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
