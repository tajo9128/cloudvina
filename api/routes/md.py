
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from auth import get_current_user
from celery_app import celery_app

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
    
    # Validation/Conversion: Frontend sends PDBQT, Worker expects PDB
    # We attempt to convert here to avoid crashing the remote worker
    from services.smiles_converter import pdbqt_to_pdb
    
    pdb_content = job.pdb_content
    # Check if content looks like PDBQT (has ROOT/BRANCH)
    if "ROOT" in pdb_content or "TORSDOF" in pdb_content or "pdbqt" in pdb_content.lower():
         converted, err = pdbqt_to_pdb(pdb_content)
         if converted:
             pdb_content = converted
             # print(f"DEBUG: Converted PDBQT to PDB for job {job_id}")
         else:
             print(f"WARNING: PDBQT conversion failed for MD job {job_id}: {err}")
             # We send original content as fallback, hoping worker handles it or fails gracefully
    
    # Send task to Celery "run_openmm_simulation"
    # This matches the name defined in workers/openmm_worker.py
    try:
        task = celery_app.send_task(
            "run_openmm_simulation",
            args=[job_id, pdb_content, config_dict],
            task_id=job_id,
            queue="colab_gpu" # Send to 'colab_gpu' queue to ensure GPU execution
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully. Waiting for Colab GPU worker (queue: colab_gpu)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

@router.get("/status/{job_id}")
async def get_md_status(job_id: str, current_user: dict = Depends(get_current_user)):
    """
    Checks the status of a specific MD job.
    """
    try:
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
    except Exception as e:
        # Fallback if redis is down or task not found
        return {"job_id": job_id, "status": "UNKNOWN", "error": str(e)}

class BindingEnergyRequest(BaseModel):
    ligand_resname: str = "LIG"
    stride: int = 10  # Analyze every Nth frame

@router.post("/analyze/binding-energy/{job_id}")
async def analyze_binding_energy(job_id: str, request: BindingEnergyRequest, current_user: dict = Depends(get_current_user)):
    """
    Triggers MM-GBSA binding energy calculation for a completed MD job.
    """
    # Create a new task ID for the analysis
    analysis_task_id = f"{job_id}_binding_energy"
    
    try:
        task = celery_app.send_task(
            "calculate_binding_energy",
            args=[job_id, request.ligand_resname, request.stride],
            task_id=analysis_task_id,
            queue="worker"
        )
        
        return {
            "job_id": job_id,
            "analysis_task_id": analysis_task_id,
            "status": "queued",
            "message": "Binding energy calculation started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue analysis: {str(e)}")

