from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from auth import get_current_user, get_authenticated_client
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
import boto3

router = APIRouter(prefix="/jobs/batch", tags=["Batch Jobs"])
security = HTTPBearer()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")
s3_client = boto3.client('s3', region_name=AWS_REGION)

class BatchSubmitRequest(BaseModel):
    receptor_filename: str
    ligand_filenames: List[str]

class BatchStartRequest(BaseModel):
    grid_params: dict

def generate_batch_urls(batch_id: str, receptor_filename: str, ligand_filenames: List[str]):
    try:
        # Shared Receptor Key
        receptor_ext = os.path.splitext(receptor_filename)[1].lower() or '.pdb'
        receptor_key = f"batches/{batch_id}/receptor{receptor_ext}"
        
        receptor_url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET, 'Key': receptor_key},
            ExpiresIn=600
        )

        ligand_urls = []
        ligand_keys = {}

        for filename in ligand_filenames:
            ligand_ext = os.path.splitext(filename)[1].lower() or '.pdbqt'
            # Use unique path per ligand to avoid collisions if filenames are same (though list implies distinct)
            # But safer to just use filename in the batch folder
            key = f"batches/{batch_id}/ligands/{filename}"
            ligand_keys[filename] = key
            
            url = s3_client.generate_presigned_url(
                'put_object',
                Params={'Bucket': S3_BUCKET, 'Key': key},
                ExpiresIn=600
            )
            ligand_urls.append({"filename": filename, "url": url})

        return receptor_url, receptor_key, ligand_urls, ligand_keys

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate URLs: {str(e)}")


@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_batch(
    request: BatchSubmitRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Submit a batch of ligands for docking against one receptor.
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        batch_id = str(uuid.uuid4())

        # Check limits (MAX 100)
        if len(request.ligand_filenames) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 ligands per batch")

        # Generate URLs
        rec_url, rec_key, lig_urls, lig_keys = generate_batch_urls(
            batch_id, request.receptor_filename, request.ligand_filenames
        )

        # Create Job Records (One per ligand)
        jobs_data = []
        for filename in request.ligand_filenames:
            job_id = str(uuid.uuid4())
            jobs_data.append({
                'id': job_id,
                'user_id': current_user.id,
                'status': 'PENDING',
                'batch_id': batch_id,
                'receptor_s3_key': rec_key, # Shared
                'ligand_s3_key': lig_keys[filename],
                'receptor_filename': request.receptor_filename,
                'ligand_filename': filename
            })

        # Bulk Insert
        # Supabase API might strictly limit bulk insert size, but 100 should be OK.
        result = auth_client.table('jobs').insert(jobs_data).execute()

        return {
            "batch_id": batch_id,
            "upload_urls": {
                "receptor_url": rec_url,
                "ligands": lig_urls
            },
            "job_count": len(jobs_data),
            "message": "Batch created. Upload files to URLs then call /start"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")

@router.post("/{batch_id}/start")
async def start_batch(
    batch_id: str,
    request: BatchStartRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Start all jobs in a batch after upload.
    """
    try:
        from services.config_generator import generate_vina_config
        from aws_services import submit_batch_job as submit_to_aws

        auth_client = get_authenticated_client(credentials.credentials)

        # Fetch jobs
        jobs_res = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', current_user.id).execute()
        jobs = jobs_res.data

        if not jobs:
            raise HTTPException(status_code=404, detail="Batch not found or empty")

        started_count = 0
        grid_params = request.grid_params

        # Iterate and start (Ideally asynchronous background task, but doing sync loop for MVP)
        for job in jobs:
            try:
                # 1. Generate Config
                # Note: Config generation uploads to S3 `jobs/{job_id}/config.txt`.
                # We need to make sure generate_vina_config uses job_id and uploads correctly.
                # It accepts job_id.
                # It generates config based on grid_params.
                generate_vina_config(job['id'], grid_params=grid_params)

                # 2. Submit to AWS
                aws_job_id = submit_to_aws(
                    job['id'],
                    job['receptor_s3_key'],
                    job['ligand_s3_key']
                )

                # 3. Update Status
                auth_client.table('jobs').update({
                    'status': 'SUBMITTED',
                    'batch_job_id': aws_job_id
                }).eq('id', job['id']).execute()

                started_count += 1
            except Exception as e:
                print(f"Failed to start job {job['id']}: {e}")
                # Continue starting others? Yes.

        return {
            "batch_id": batch_id,
            "started": started_count,
            "total": len(jobs),
            "message": "Batch processing started"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch: {str(e)}")
