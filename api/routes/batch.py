from fastapi import APIRouter, Depends, HTTPException, status, Security, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from auth import get_current_user, get_authenticated_client, security
# security = HTTPBearer() # Removed local instance to use shared auth.security

import os
import boto3
import json
import logging
import uuid
from typing import List, Optional
from pydantic import BaseModel
from services.fda_service import fda_service
from services.export import ExportService
from services.vina_parser import parse_vina_log

router = APIRouter(prefix="/jobs/batch", tags=["Batch Jobs"])

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
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        # Validate batch exists first
        auth_client = get_authenticated_client(credentials.credentials)
        # Just check one job to verify ownership
        check = auth_client.table('jobs').select('id').eq('batch_id', batch_id).eq('user_id', current_user.id).limit(1).execute()
        if not check.data:
            raise HTTPException(404, "Batch not found or empty")

        # Offload to background
        background_tasks.add_task(
            process_batch_jobs, 
            batch_id, 
            current_user.id, 
            credentials.credentials, 
            request.grid_params, 
            request.engine
        )

        return {"batch_id": batch_id, "started": True, "message": "Batch processing started in background"}

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Start failed: {e}")

async def process_batch_jobs(batch_id: str, user_id: str, token: str, grid_params: dict, engine: str):
    """
    Background worker to process and submit batch jobs.
    Handles PDBQT conversion, S3 upload, and AWS Batch submission.
    """
    try:
        from services.config_generator import generate_vina_config
        from aws_services import submit_batch_job as submit_to_aws
        from services.smiles_converter import convert_to_pdbqt, convert_receptor_to_pdbqt
        import tempfile
        import shutil
        import traceback

        # Re-initialize client for background thread
        auth_client = get_authenticated_client(token)
        
        # Fetch all jobs
        jobs = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', user_id).execute().data
        
        print(f"🔄 [Background] Starting processing for batch {batch_id} ({len(jobs)} jobs)")
        
        started_count = 0
        
        for job in jobs:
            try:
                # Update status to PROCESSING
                auth_client.table('jobs').update({'status': 'PROCESSING', 'notes': 'Starting Preparation...'}).eq('id', job['id']).execute()
                
                job_id = job['id']
                job_rec_key = f"jobs/{job_id}/receptor.pdbqt"
                job_lig_key = f"jobs/{job_id}/ligand.pdbqt"
                
                # 1. Prepare Receptor & Ligand (Check/Convert/Copy)
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Generic function to handle download -> convert -> key
                    def prepare_file(s3_key, filename, target_key, is_receptor=False):
                        print(f"   preparing {filename}...", flush=True)
                        ext = os.path.splitext(filename)[1].lower()
                        local_path = os.path.join(temp_dir, filename)
                        
                        # Download
                        s3_client.download_file(S3_BUCKET, s3_key, local_path)
                        
                        # Convert if needed
                        if ext != '.pdbqt':
                            print(f"   Converting {filename} to PDBQT...", flush=True)
                            with open(local_path, 'r') as f: content = f.read()
                            
                            pdbqt_content = None
                            err = None
                            
                            if is_receptor:
                                pdbqt_content, err = convert_receptor_to_pdbqt(content, filename)
                            else:
                                pdbqt_content, err = convert_to_pdbqt(content, filename)
                                
                            if err or not pdbqt_content:
                                raise Exception(f"Conversion failed: {err}")
                                
                            # Write Converted to temp file
                            local_path = local_path + ".pdbqt" # Update path to point to converted
                            with open(local_path, 'w') as f: f.write(pdbqt_content)
                        
                        # Upload to target
                        s3_client.upload_file(local_path, S3_BUCKET, target_key)
                        
                    # Process Receptor
                    try:
                        auth_client.table('jobs').update({'notes': 'Preparing Receptor...'}).eq('id', job_id).execute()
                        prepare_file(job['receptor_s3_key'], job['receptor_filename'], job_rec_key, is_receptor=True)
                    except Exception as rx:
                         print(f"Receptor Prep Failed for {job_id}: {rx}")
                         raise rx
                    
                    # Process Ligand
                    try:
                        auth_client.table('jobs').update({'notes': 'Preparing Ligand...'}).eq('id', job_id).execute()
                        prepare_file(job['ligand_s3_key'], job['ligand_filename'], job_lig_key, is_receptor=False)
                    except Exception as lx:
                         print(f"Ligand Prep Failed for {job_id}: {lx}")
                         raise lx
                
                # Generate Config
                auth_client.table('jobs').update({'notes': 'Generating Vina Config...'}).eq('id', job_id).execute()
                generate_vina_config(job_id, grid_params=grid_params)
                
                # Submit
                auth_client.table('jobs').update({'notes': 'Submitting to AWS Batch...'}).eq('id', job_id).execute()
                aws_id = submit_to_aws(job_id, job_rec_key, job_lig_key, engine=engine)
                
                auth_client.table('jobs').update({
                    'status': 'SUBMITTED', 'batch_job_id': aws_id, 'notes': 'Batch Submitted'
                }).eq('id', job_id).execute()
                
                print(f"✅ [Background] Job {job_id} submitted. AWS ID: {aws_id}", flush=True)
                started_count += 1
                
            except Exception as e:
                traceback.print_exc()
                print(f"❌ [Background] Job start failed: {e}", flush=True)
                auth_client.table('jobs').update({'status': 'FAILED', 'error_message': str(e)}).eq('id', job['id']).execute()

        await fda_service.log_audit_event(auth_client, user_id, 'BATCH_STARTED', 'batch', batch_id, {'jobs': started_count})
        print(f"🏁 [Background] Batch {batch_id} complete. Started: {started_count}")

    except Exception as outer_e:
        traceback.print_exc()
        print(f"❌ [Background] Critical Batch Failure: {outer_e}", flush=True)

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


# NEW ENDPOINT FOR PDF REPORTS (Moved to top to avoid routing conflicts)
@router.get("/{batch_id}/report-pdf")
async def generate_batch_report(
    batch_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Generate a comprehensive PDF report for the entire batch"""
    print(f"Generating PDF report for batch {batch_id} user {current_user.get('email')}", flush=True)
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        jobs = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', current_user.id).execute().data
        if not jobs: raise HTTPException(404, "Batch not found")

        # Use ExportService
        return ExportService.export_batch_pdf(batch_id, jobs)
        
    except Exception as e:
         print(f"Report generation error: {e}", flush=True)
         raise HTTPException(500, f"Report generation failed: {e}")

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
                            print(f"[BatchSync] Job {job['id']} status changed: {job['status']} -> {batch_res['status']}", flush=True)
                            update_data = {'status': batch_res['status']}
                            if batch_res['status'] == 'FAILED':
                                update_data['error_message'] = batch_res.get('status_reason', 'Unknown error')
                            
                            auth_client.table('jobs').update(update_data).eq('id', job['id']).execute()
                            job['status'] = batch_res['status']
                            job['error_message'] = update_data.get('error_message')
                            modified = True
                except Exception as e:
                    print(f"[BatchSync] Error syncing job {job['id']}: {e}", flush=True)

            updated_jobs.append(job)

        # Lazy Repair Score (Ported from jobs.py)
        for job in updated_jobs:
            if job['status'] == 'SUCCEEDED' and (not job.get('binding_affinity') or float(job.get('binding_affinity') or 0) == 0.0):
                try:
                    s3 = boto3.client('s3', region_name=AWS_REGION)
                    obj = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{job['id']}/results.json")
                    res_data = json.loads(obj['Body'].read().decode('utf-8'))
                    
                    # Extract scores
                    best_score = res_data.get('best_affinity')
                    vina_score = res_data.get('vina_affinity') or res_data.get('vina_score')
                    gnina_score = res_data.get('gnina_affinity') or res_data.get('cnn_score') or res_data.get('docking_score')
                    
                    updates = {}
                    if best_score is not None:
                        updates['binding_affinity'] = best_score
                        job['binding_affinity'] = best_score
                        
                    # Attempt to update DB if columns exist (ignoring errors if they don't)
                    # And blindly populate the response object
                    if vina_score is not None:
                        job['vina_score'] = vina_score
                        # updates['vina_score'] = vina_score # Uncomment if column exists
                        
                    if gnina_score is not None:
                        job['docking_score'] = gnina_score
                        # updates['docking_score'] = gnina_score # Uncomment if column exists

                    if updates:
                        try:
                            auth_client.table('jobs').update(updates).eq('id', job['id']).execute()
                        except Exception as db_err:
                            print(f"[BatchSync] Warning: Could not update some columns in DB: {db_err}", flush=True)

                except Exception as e:
                    # Log error but don't fail the request
                    print(f"Failed to repair score for job {job['id']}: {e}", flush=True)

        return {"batch_id": batch_id, "jobs": updated_jobs, "stats": {"total": len(updated_jobs)}}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch batch details: {str(e)}")


