
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
    Submits an MD simulation job to AWS Batch (md-simulation-queue).
    Uses the dedicated Docker image (OpenMM+Amber) for execution.
    """
    job_id = str(uuid.uuid4())
    
    # 1. Prepare PDB Content
    from services.smiles_converter import pdbqt_to_pdb
    pdb_content = job.pdb_content
    # Auto-convert PDBQT -> PDB
    if "ROOT" in pdb_content or "TORSDOF" in pdb_content or "pdbqt" in pdb_content.lower():
         converted, err = pdbqt_to_pdb(pdb_content)
         if converted:
             pdb_content = converted
         else:
             print(f"WARNING: PDBQT conversion failed for MD job {job_id}: {err}")

    # 2. Upload Input to S3 (Required for AWS Batch)
    try:
        from aws_services import s3_client, S3_BUCKET, submit_md_simulation_job
        pdb_key = f"jobs/{job_id}/input.pdb"
        s3_client.put_object(
            Bucket=S3_BUCKET, 
            Key=pdb_key, 
            Body=pdb_content,
            ContentType='chemical/x-pdb'
        )

        # 3. Submit to AWS Batch (MD Queue)
        batch_job_id = submit_md_simulation_job(job_id, pdb_key)
        
        return {
            "job_id": job_id,
            "aws_batch_id": batch_job_id,
            "status": "SUBMITTED",
            "message": "MD Simulation submitted to AWS Batch (GPU/Docker)."
        }
    except Exception as e:
        import traceback
        import sys
        print("❌ MD SUBMISSION ERROR:", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to submit MD job: {str(e)}")

@router.get("/status/{job_id}")
async def get_md_status(
    job_id: str, 
    aws_batch_id: Optional[str] = None, 
    current_user: dict = Depends(get_current_user)
):
    """
    Checks the status of a specific MD job on AWS.
    If aws_batch_id is provided, checks Batch status.
    Otherwise, checks S3 for result artifacts.
    """
    try:
        from aws_services import s3_client, S3_BUCKET, get_batch_job_status, generate_presigned_download_url
        import botocore
        
        # 1. Check if Results exist in S3 (Ultimate Success Check)
        # We look for the main trajectory or report
        result_key = f"jobs/{job_id}/md_log.txt" 
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=result_key)
            # If successful, job is DONE
            
            # Generate URLs for outputs
            return {
                "job_id": job_id,
                "status": "SUCCESS",
                "result": {
                    "trajectory_url": generate_presigned_download_url(job_id, "trajectory.dcd"),
                    "report_url": generate_presigned_download_url(job_id, "MD_analysis_report.pdf"),
                    "log_url": generate_presigned_download_url(job_id, "md_log.txt"),
                    "rmsd_plot_url": generate_presigned_download_url(job_id, "rmsd_plot.png"),
                    "stability_score": 85.5 # TODO: Read from JSON file
                }
            }
        except botocore.exceptions.ClientError:
            pass # Results not ready yet

        # 2. If results not found, check Batch Status (if ID provided)
        status_label = "RUNNING"
        if aws_batch_id:
            try:
                batch_info = get_batch_job_status(aws_batch_id)
                status_label = batch_info.get("status", "UNKNOWN")
            except:
                pass 
        
        return {
            "job_id": job_id,
            "status": status_label,
            "info": {"message": "Simulation in progress on AWS Batch..."}
        }

    except Exception as e:
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

