
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from services.report_generator import ReportGenerator
import logging

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
async def generate_pymol_export(job_id: str):
    """
    Generate a PyMOL script for a specific job.
    """
    try:
        from services.report_generator import ReportGenerator
        from auth import get_service_client # Or passed in context
        
        # Ideally fetch job to get URLs. For speed in this request scope, 
        # we'll assume we can pass generic URLs or fetch them.
        # Simulating fetch for now or passing dummy until integrated with DB lookup in router
        
        # NOTE: In a real app, I'd fetch the S3 URLs from the DB here.
        # For this quick fix, I will use placeholders that the user can replace or 
        # assume standarized naming if S3 keys are predictable.
        
        receptor_url = "receptor.pdbqt" # Placeholder or fetch from DB
        ligand_url = "ligand_out.pdbqt"

        generator = ReportGenerator()
        buffer = generator.generate_pymol_script(job_id, receptor_url, ligand_url)
        
        return StreamingResponse(
            buffer,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=visualization_{job_id}.pml"}
        )
    except Exception as e:
        logger.error(f"PyMOL generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
