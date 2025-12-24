from fastapi import APIRouter, Depends, HTTPException, status, Security, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from auth import get_current_user, get_authenticated_client
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
import boto3
import json
import logging
from services.fda_service import fda_service
from services.export import ExportService
from services.vina_parser import parse_vina_log

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
    engine: Optional[str] = "consensus"

class CSVBatchSubmitRequest(BaseModel):
    grid_center_x: float = 0.0
    grid_center_y: float = 0.0
    grid_center_z: float = 0.0
    grid_size_x: float = 20.0
    grid_size_y: float = 20.0
    grid_size_z: float = 20.0
    engine: str = "consensus"

def generate_batch_urls(batch_id: str, receptor_filename: str, ligand_filenames: List[str]):
    try:
        receptor_ext = os.path.splitext(receptor_filename)[1].lower() or '.pdb'
        receptor_key = f"jobs/{batch_id}/receptor_input{receptor_ext}"
        
        receptor_url = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET, 'Key': receptor_key},
            ExpiresIn=600
        )

        ligand_urls = []
        ligand_keys = {}

        for filename in ligand_filenames:
            key = f"jobs/{batch_id}/ligands/{filename}"
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
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        batch_id = str(uuid.uuid4())
        
        if len(request.ligand_filenames) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 ligands per batch")

        rec_url, rec_key, lig_urls, lig_keys = generate_batch_urls(
            batch_id, request.receptor_filename, request.ligand_filenames
        )

        jobs_data = []
        for filename in request.ligand_filenames:
            job_id = str(uuid.uuid4())
            jobs_data.append({
                'id': job_id,
                'user_id': current_user.id,
                'status': 'PENDING',
                'batch_id': batch_id,
                'receptor_s3_key': rec_key,
                'ligand_s3_key': lig_keys[filename],
                'receptor_filename': request.receptor_filename,
                'ligand_filename': filename
            })

        auth_client.table('jobs').insert(jobs_data).execute()

        return {
            "batch_id": batch_id,
            "upload_urls": {
                "receptor_url": rec_url,
                "ligands": lig_urls
            },
            "job_count": len(jobs_data),
            "message": "Batch initialized"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")

@router.post("/{batch_id}/start")
async def start_batch(
    batch_id: str,
    request: BatchStartRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        from services.config_generator import generate_vina_config
        from aws_services import submit_batch_job as submit_to_aws
        from services.smiles_converter import convert_receptor_to_pdbqt, convert_to_pdbqt

        auth_client = get_authenticated_client(credentials.credentials)
        jobs = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', current_user.id).execute().data
        
        if not jobs: raise HTTPException(404, "Batch not found")

        started_count = 0
        
        # NOTE: This loop is synchronous but handles critical preparation logic.
        # Ideally moved to Celery, but keeping simple for now.
        for job in jobs:
            try:
                job_id = job['id']
                job_rec_key = f"jobs/{job_id}/receptor.pdbqt"
                job_lig_key = f"jobs/{job_id}/ligand.pdbqt"
                
                # 1. Prepare Receptor (Check/Convert/Copy)
                if not job['receptor_filename'].endswith('.pdbqt'):
                     # Needs conversion (Expensive - should be cached or async)
                     # For now, we assume if it's NOT pdbqt, we convert.
                     # Optimisation: Check if we already converted for this batch?
                     # Using "Shared" receptor logic would be better: convert ONCE per batch.
                     # But current data model is per-job.
                     pass 
                
                # Simplified Logic:
                # We assume client uploaded to the PRESIGNED URLs which are formatted as:
                # receptor: jobs/{batch_id}/receptor_input.ext
                # ligand: jobs/{batch_id}/ligands/{filename}
                
                # We need to COPY these to job specific API-expected paths:
                # jobs/{job_id}/receptor_input.pdbqt
                # jobs/{job_id}/ligand_input.pdbqt
                
                # Doing this sync for now as per legacy logic, but cleaner.
                
                # Generate Config
                generate_vina_config(job_id, grid_params=request.grid_params)
                
                # Submit
                aws_id = submit_to_aws(job_id, job['receptor_s3_key'], job['ligand_s3_key'], engine=request.engine)
                
                auth_client.table('jobs').update({
                    'status': 'SUBMITTED', 'batch_job_id': aws_id
                }).eq('id', job_id).execute()
                
                started_count += 1
            except Exception as e:
                print(f"Job start failed: {e}")
                auth_client.table('jobs').update({'status': 'FAILED', 'error_message': str(e)}).eq('id', job['id']).execute()

        await fda_service.log_audit_event(auth_client, current_user.id, 'BATCH_STARTED', 'batch', batch_id, {'jobs': started_count})

        return {"batch_id": batch_id, "started": started_count, "message": "Batch started"}

    except Exception as e:
        raise HTTPException(500, f"Start failed: {e}")

@router.post("/submit-csv", status_code=status.HTTP_201_CREATED)
async def submit_csv_batch(
    receptor_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
    grid_params: str = None, # JSON string
    engine: str = "consensus",
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    import pandas as pd
    import io
    from services.smiles_converter import smiles_to_pdbqt
    from aws_services import submit_batch_job as submit_to_aws
    from services.config_generator import generate_vina_config
    
    auth_client = get_authenticated_client(credentials.credentials)
    batch_id = str(uuid.uuid4())
    
    # Grid Params parsing
    gp = {'center_x': 0, 'center_y': 0, 'center_z': 0, 'size_x': 20, 'size_y': 20, 'size_z': 20}
    if grid_params:
        try:
            gp.update(json.loads(grid_params))
        except: pass

    # 1. Handle CSV
    content = await csv_file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except:
        raise HTTPException(400, "Invalid CSV")
    
    smiles_col = next((c for c in df.columns if c.lower() == 'smiles'), None)
    if not smiles_col: raise HTTPException(400, "No 'smiles' column")
    
    smiles_list = df[smiles_col].dropna().tolist()
    if len(smiles_list) > 10: raise HTTPException(400, "Max 10 items")

    # 2. Upload Receptor
    # TODO: Convert if not PDBQT
    rec_key = f"jobs/{batch_id}/receptor.pdbqt"
    rec_content = await receptor_file.read()
    # Assuming PDBQT for simplified async flow, or convert here?
    # Let's convert PDB->PDBQT if needed using threadpool
    if not receptor_file.filename.endswith('.pdbqt'):
        from services.smiles_converter import convert_receptor_to_pdbqt
        # Run conversion in thread to avoid blocking main loop
        pdbqt_content, err = await run_in_threadpool(convert_receptor_to_pdbqt, rec_content.decode('utf-8'), receptor_file.filename)
        if err: raise HTTPException(400, f"Receptor conversion error: {err}")
        rec_content = pdbqt_content.encode('utf-8')

    s3_client.put_object(Bucket=S3_BUCKET, Key=rec_key, Body=rec_content)

    # 3. Process Ligands (Async-ish)
    jobs_created = []
    
    for idx, smiles in enumerate(smiles_list):
        job_id = str(uuid.uuid4())
        compound_name = f"cmpd_{idx}"
        
        # Heavy CPU Task - Run in Threadpool
        pdbqt_content, error = await run_in_threadpool(smiles_to_pdbqt, smiles, compound_name)
        
        if error: continue 
        
        lig_key = f"jobs/{job_id}/ligand.pdbqt"
        s3_client.put_object(Bucket=S3_BUCKET, Key=lig_key, Body=pdbqt_content.encode('utf-8'))
        
        # Copy receptor to job folder (needed for AWS Batch isolation)
        job_rec_key = f"jobs/{job_id}/receptor.pdbqt"
        s3_client.copy_object(Bucket=S3_BUCKET, CopySource={'Bucket': S3_BUCKET, 'Key': rec_key}, Key=job_rec_key)
        
        auth_client.table('jobs').insert({
            'id': job_id,
            'user_id': current_user.id,
            'status': 'PENDING',
            'batch_id': batch_id,
            'receptor_s3_key': job_rec_key,
            'ligand_s3_key': lig_key,
            'receptor_filename': receptor_file.filename,
            'ligand_filename': f"{compound_name}.pdbqt",
            'notes': smiles
        }).execute()
        
        # Submit
        generate_vina_config(job_id, grid_params=gp)
        aid = submit_to_aws(job_id, job_rec_key, lig_key, engine=engine)
        
        auth_client.table('jobs').update({'status': 'SUBMITTED', 'batch_job_id': aid}).eq('id', job_id).execute()
        jobs_created.append(job_id)

    return {"batch_id": batch_id, "jobs_created": len(jobs_created)}


@router.get("/{batch_id}")
async def get_batch_details(
    batch_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        from aws_services import get_batch_job_status
        
        auth_client = get_authenticated_client(credentials.credentials)
        jobs = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', current_user.id).execute().data
        
        if not jobs: raise HTTPException(404, "Batch not found")

        # Sync Logic (Ported from jobs.py)
        # We limit specific checks to avoid timeouts on large batches, 
        # but for decent sizes this is okay for now.
        updated_jobs = []
        for job in jobs:
            modified = False
            
            # 1. Sync Status with AWS Batch
            if job['status'] in ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING']:
                try:
                    # Only check if we have an AWS Batch ID
                    if job.get('batch_job_id'):
                        batch_res = get_batch_job_status(job['batch_job_id'])
                        if batch_res['status'] != job['status']:
                            print(f"[BatchSync] Job {job['id']} status changed: {job['status']} -> {batch_res['status']}")
                            update_data = {'status': batch_res['status']}
                            if batch_res['status'] == 'FAILED':
                                update_data['error_message'] = batch_res.get('status_reason', 'Unknown error')
                            
                            auth_client.table('jobs').update(update_data).eq('id', job['id']).execute()
                            job['status'] = batch_res['status']
                            job['error_message'] = update_data.get('error_message')
                            modified = True
                except Exception as e:
                    print(f"[BatchSync] Error syncing job {job['id']}: {e}")

            # 2. Lazy Repair Scores (if specific score is missing but job succeeded)
            # This handles cases where the webhook/callback missed the update
            # 2. Lazy Repair Scores (if specific score is missing but job succeeded)
            # This handles cases where the webhook/callback missed the update or results.json is missing (Legacy V10 jobs)
            if job['status'] == 'SUCCEEDED' and (not job.get('binding_affinity') or float(job.get('binding_affinity') or 0) == 0.0):
                print(f"[LazyRepair] Checking job {job['id']} for missing scores...")
                try:
                    # Strategy 1: Try results.json (Newer Job Definitions)
                    try:
                        s3_key = f"jobs/{job['id']}/results.json"
                        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
                        res_data = json.loads(obj['Body'].read().decode('utf-8'))
                        
                        updates = {}
                        if 'best_affinity' in res_data:
                            updates['binding_affinity'] = res_data['best_affinity']
                            job['binding_affinity'] = res_data['best_affinity']
                        
                        if 'vina_score' in res_data:
                             updates['vina_score'] = res_data['vina_score']
                             job['vina_score'] = res_data['vina_score']
    
                        if 'docking_score' in res_data: # Gnina
                             updates['docking_score'] = res_data['docking_score']
                             job['docking_score'] = res_data['docking_score']
                        
                        if updates:
                            print(f"[LazyRepair] Restored from results.json: {updates}")
                            auth_client.table('jobs').update(updates).eq('id', job['id']).execute()
                            modified = True
                    except Exception as json_err:
                        print(f"[LazyRepair] results.json failed ({json_err}). Trying log.txt fallback...")
                        # Strategy 2: Fallback to log.txt (Old Job Definitions like v10)
                        try:
                            log_key = f"jobs/{job['id']}/log.txt"
                            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=log_key)
                            log_content = obj['Body'].read().decode('utf-8')
                            parsed = parse_vina_log(log_content)
                            
                            if parsed.get('best_affinity'):
                                print(f"[LazyRepair] Restored from log.txt: {parsed['best_affinity']}")
                                updates = {'binding_affinity': parsed['best_affinity'], 'vina_score': parsed['best_affinity']}
                                auth_client.table('jobs').update(updates).eq('id', job['id']).execute()
                                job['binding_affinity'] = parsed['best_affinity']
                                job['vina_score'] = parsed['best_affinity']
                                modified = True
                        except Exception as log_ex:
                             print(f"[LazyRepair] log.txt failed as well: {log_ex}")
                             pass
                except Exception as e:
                    print(f"[LazyRepair] Critical error: {e}")
            
            updated_jobs.append(job)

        return {"batch_id": batch_id, "jobs": updated_jobs, "stats": {"total": len(updated_jobs)}}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch batch details: {str(e)}")

# NEW ENDPOINT FOR PDF REPORTS
@router.get("/{batch_id}/report-pdf")
async def generate_batch_report(
    batch_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Generate a comprehensive PDF report for the entire batch"""
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        jobs = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', current_user.id).execute().data
        if not jobs: raise HTTPException(404, "Batch not found")

        # Use ExportService
        return ExportService.export_batch_pdf(batch_id, jobs)
        
    except Exception as e:
         raise HTTPException(500, f"Report generation failed: {e}")
