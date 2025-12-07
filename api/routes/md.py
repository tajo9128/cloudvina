
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from ..auth import get_current_user
from ..celery_app import celery_app

router = APIRouter(prefix="/md", tags=["Molecular Dynamics"])

class MDConfig(BaseModel):
    temperature: float = 300.0
    steps: int = 5000  # Default 10ps (assuming 2fs step)
    forcefield: str = "amber14-all.xml"
    water: str = "amber14/tip3pfb.xml"

class MDJobRequest(BaseModel):
    pdb_content: str
    config: Optional[MDConfig] = MDConfig()

@router.post("/submit")
async def submit_md_job(job: MDJobRequest, current_user: dict = Depends(get_current_user)):
    """
    Submits an MD simulation job to the Redis queue.
    This job will be picked up by an external Colab worker or local worker.
    """
    job_id = str(uuid.uuid4())
    
    # Prepare config dict
    config_dict = job.config.model_dump()
    
    # Send task to Celery "run_openmm_simulation"
    # This matches the name defined in workers/openmm_worker.py
    try:
        task = celery_app.send_task(
            "run_openmm_simulation",
            args=[job_id, job.pdb_content, config_dict],
            task_id=job_id,
            queue="worker" # Send to 'worker' queue
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully. Waiting for a worker (Colab) to pick it up."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

@router.get("/status/{job_id}")
async def get_md_status(job_id: str, current_user: dict = Depends(get_current_user)):
    """
    Checks the status of a specific MD job.
    """
    task_result = celery_app.AsyncResult(job_id)
    
    response = {
        "job_id": job_id,
        "status": task_result.status,
        "info": None
    }
    
    # If the task is running or has custom meta (PROGRESS state)
    if task_result.status == 'PROGRESS':
        response["info"] = task_result.info
    elif task_result.status == 'SUCCESS':
        response["result"] = task_result.result
    elif task_result.status == 'FAILURE':
        response["error"] = str(task_result.result)
        
    return response
