
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
