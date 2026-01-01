from fastapi import APIRouter, Depends, HTTPException, status, Security, Body, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import os
import boto3
import json
from auth import get_current_user, get_authenticated_client
from aws_services import (
    generate_presigned_upload_urls,
    generate_presigned_download_url,
    submit_batch_job,
    get_batch_job_status
)
from services.rate_limiter import RateLimiter
from services.cavity_detector import CavityDetector
from services.drug_properties import DrugPropertiesCalculator
from services.smiles_converter import pdbqt_to_pdb
from services.vina_parser import parse_vina_log
from services.interaction_analyzer import InteractionAnalyzer
from services.ai_explainer import AIExplainer
from services.export import ExportService
from rdkit import Chem

router = APIRouter(prefix="/jobs", tags=["Docking Jobs"])
security = HTTPBearer()

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")

# Models
class DockingConfig(BaseModel):
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 8

class JobSubmitRequest(BaseModel):
    receptor_filename: str
    ligand_filename: str
    config: Optional[DockingConfig] = DockingConfig()

@router.post("/submit", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    request: JobSubmitRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Submit a new docking job and get upload URLs
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Rate Limiting
        eligibility = await RateLimiter.check_can_submit(auth_client, current_user)
        if not eligibility['allowed']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": eligibility['message'],
                    "reason": eligibility.get('reason'),
                    "jobs_today": eligibility.get('jobs_today'),
                    "limit": eligibility.get('limit')
                }
            )
        
        job_id = str(uuid.uuid4())
        
        receptor_url, ligand_url, receptor_key, ligand_key = generate_presigned_upload_urls(
            job_id,
            request.receptor_filename,
            request.ligand_filename
        )
        
        auth_client.table('jobs').insert({
            'id': job_id,
            'user_id': current_user.id,
            'status': 'PENDING',
            'receptor_s3_key': receptor_key,
            'ligand_s3_key': ligand_key,
            'receptor_filename': request.receptor_filename,
            'ligand_filename': request.ligand_filename
        }).execute()
        
        await RateLimiter.increment_usage(auth_client, current_user.id)
        
        return {
            "job_id": job_id,
            "upload_urls": {
                "receptor": receptor_url,
                "ligand": ligand_url
            },
            "message": "Upload files, then call /start",
            "remaining_jobs_today": eligibility.get('remaining_today')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@router.post("/{job_id}/start")
async def start_job(
    job_id: str,
    request: dict = Body(default={}),
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).single().execute()
        
        if not job.data:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = job.data

        grid_params = request.get('grid_params')
        engine = request.get('engine', 'vina')
        
        # Config
        from services.config_generator import generate_vina_config
        generate_vina_config(job_id, grid_params=grid_params)
        
        # Helper for S3 operations
        s3 = boto3.client('s3', region_name=AWS_REGION)

        # 1. Receptor Preparation
        job_receptor_key = job_data['receptor_s3_key']
        rec_ext = job_data['receptor_filename'].lower().split('.')[-1]
        
        if rec_ext != 'pdbqt':
             try:
                 from services.smiles_converter import convert_receptor_to_pdbqt
                 pdb_obj = s3.get_object(Bucket=S3_BUCKET, Key=job_data['receptor_s3_key'])
                 rec_content = pdb_obj['Body'].read().decode('utf-8')
                 pdbqt_content, err = convert_receptor_to_pdbqt(rec_content, job_data['receptor_filename'])
                 if err or not pdbqt_content: raise Exception(f"Receptor conversion failed: {err}")
                 
                 new_rec_key = f"jobs/{job_id}/receptor_input_converted.pdbqt"
                 s3.put_object(Bucket=S3_BUCKET, Key=new_rec_key, Body=pdbqt_content.encode('utf-8'))
                 job_receptor_key = new_rec_key
             except Exception as rx:
                 print(f"Receptor Prep Error: {rx}")
                 raise HTTPException(status_code=400, detail=f"Receptor conversion failed: {str(rx)}")

        # 2. Ligand Preparation
        final_ligand_key = job_data['ligand_s3_key']
        if not final_ligand_key.lower().endswith('.pdbqt'):
            try:
                from services.smiles_converter import convert_to_pdbqt
                obj = s3.get_object(Bucket=S3_BUCKET, Key=final_ligand_key)
                content = obj['Body'].read().decode('utf-8')
                pdbqt_content, err = convert_to_pdbqt(content, job_data['ligand_filename'])
                if err: raise Exception(f"Conversion failed: {err}")
                
                new_key = f"jobs/{job_id}/ligand_input_converted.pdbqt"
                s3.put_object(Bucket=S3_BUCKET, Key=new_key, Body=pdbqt_content.encode('utf-8'))
                final_ligand_key = new_key
            except Exception as lx:
                 print(f"Ligand Prep Error: {lx}")
                 raise HTTPException(status_code=400, detail=f"Ligand conversion failed: {str(lx)}")
        
        # Submit to AWS
        batch_job_id = submit_batch_job(
            job_id,
            job_receptor_key,
            final_ligand_key,
            engine=engine
        )
        
        auth_client.table('jobs').update({
            'status': 'SUBMITTED', 
            'batch_job_id': batch_job_id
        }).eq('id', job_id).execute()
        
        return {"job_id": job_id, "status": "SUBMITTED"}
        
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Start failed: {str(e)}")

@router.get("", response_model=list) # /jobs
async def list_jobs(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        res = auth_client.table('jobs').select('*').eq('user_id', current_user.id).order('created_at', desc=True).execute()
        return res.data if res.data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job_res = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).single().execute()
        if not job_res.data: raise HTTPException(status_code=404, detail="Job not found")
        
        job = job_res.data
        
        # Status Sync Logic
        if job['status'] in ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING']:
            batch_status = get_batch_job_status(job['batch_job_id'])
            if batch_status['status'] != job['status']:
                update = {'status': batch_status['status']}
                if batch_status['status'] == 'FAILED':
                    update['error_message'] = batch_status.get('status_reason')
                    # Refund logic could go here
                auth_client.table('jobs').update(update).eq('id', job_id).execute()
                job['status'] = batch_status['status']

        # Lazy Repair Score
        if job['status'] == 'SUCCEEDED' and (not job.get('binding_affinity') or float(job.get('binding_affinity') or 0) == 0.0):
             try:
                 s3 = boto3.client('s3', region_name=AWS_REGION)
                 obj = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{job_id}/results.json")
                 res_data = json.loads(obj['Body'].read().decode('utf-8'))
                 if 'best_affinity' in res_data:
                     job['binding_affinity'] = res_data['best_affinity']
                     auth_client.table('jobs').update({'binding_affinity': res_data['best_affinity']}).eq('id', job_id).execute()
             except: pass

        download_urls = {}
        if job['status'] == 'SUCCEEDED':
            receptor_fname = os.path.basename(job.get('receptor_s3_key', 'receptor.pdb'))
            download_urls = {
                'output': generate_presigned_download_url(job_id, 'output.pdbqt'),
                'receptor': generate_presigned_download_url(job_id, receptor_fname),
                'log': generate_presigned_download_url(job_id, 'log.txt'),
                'config': generate_presigned_download_url(job_id, 'config.txt'),
                'results_json': generate_presigned_download_url(job_id, 'results.json'),
                'output_vina': generate_presigned_download_url(job_id, 'output_vina.pdbqt'),
                'output_gnina': generate_presigned_download_url(job_id, 'output_gnina.pdbqt')
            }

        return {
            "job_id": job_id, 
            "status": job['status'], 
            "download_urls": download_urls,
            "binding_affinity": job.get('binding_affinity'),
            "ligand_filename": job.get('ligand_filename'),
            "receptor_filename": job.get('receptor_filename'),
            "error_message": job.get('error_message')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/files/{file_type}")
async def get_job_file_url(
    job_id: str, 
    file_type: str, 
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job = auth_client.table('jobs').select('user_id').eq('id', job_id).single().execute()
        if not job.data: raise HTTPException(status_code=404, detail="Job not found")
        if job.data['user_id'] != current_user.id: raise HTTPException(status_code=403, detail="Not authorized")

        filename_map = {
            'log': 'log.txt', 'config': 'config.txt', 'output': 'output.pdbqt',
            'ligand': 'ligand_input.pdbqt', 'receptor': 'receptor_input.pdbqt',
            'results': 'results.json'
        }
        filename = filename_map.get(file_type)
        if not filename: raise HTTPException(status_code=400, detail="Invalid file type")
        return {"url": generate_presigned_download_url(job_id, filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/detect-cavities")
async def detect_cavities(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).single().execute()
        if not job.data: raise HTTPException(status_code=404, detail="Job not found")
        
        s3 = boto3.client('s3', region_name=AWS_REGION)
        resp = s3.get_object(Bucket=S3_BUCKET, Key=job.data['receptor_s3_key'])
        pdb_content = resp['Body'].read().decode('utf-8')
        
        detector = CavityDetector()
        cavities = detector.detect_cavities(pdb_content)
        return {"job_id": job_id, "cavities": cavities or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}/admet")
async def get_job_admet(job_id: str, current_user: dict = Depends(get_current_user), credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job = auth_client.table('jobs').select('*').eq('id', job_id).single().execute()
        if not job.data: raise HTTPException(404, "Job not found")
        
        s3 = boto3.client('s3', region_name=AWS_REGION)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=job.data['ligand_s3_key'])
        pdbqt_content = obj['Body'].read().decode('utf-8')
        
        pdb_content, err = pdbqt_to_pdb(pdbqt_content)
        mol = Chem.MolFromPDBBlock(pdb_content) if not err else None
        if not mol: raise HTTPException(400, "Structure parsing failed")
        
        smiles = Chem.MolToSmiles(mol)
        return DrugPropertiesCalculator().calculate_all(smiles)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/{job_id}/analysis")
async def get_docking_analysis(job_id: str, current_user: dict = Depends(get_current_user), credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job = auth_client.table('jobs').select('*').eq('id', job_id).single().execute()
        if not job.data: raise HTTPException(404, "Job not found")
        
        if job.data.get('docking_results'): return {"job_id": job_id, "analysis": job.data['docking_results'], "from_cache": True}
        
        s3 = boto3.client('s3', region_name=AWS_REGION)
        log = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{job_id}/log.txt")['Body'].read().decode('utf-8')
        analysis = parse_vina_log(log)
        
        auth_client.table('jobs').update({'docking_results': analysis}).eq('id', job_id).execute()
        return {"job_id": job_id, "analysis": analysis, "from_cache": False}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/{job_id}/interactions")
async def get_interaction_analysis(job_id: str, current_user: dict = Depends(get_current_user), credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        job = auth_client.table('jobs').select('*').eq('id', job_id).single().execute()
        if not job.data: raise HTTPException(404, "Job not found")

        if job.data.get('interaction_results'): return {"job_id": job_id, "interactions": job.data['interaction_results'], "from_cache": True}
        
        s3 = boto3.client('s3', region_name=AWS_REGION)
        rec = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{job_id}/receptor.pdb")['Body'].read().decode('utf-8') # Optimistic key
        out = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{job_id}/output.pdbqt")['Body'].read().decode('utf-8')
        
        interactions = InteractionAnalyzer().analyze_interactions(rec, out)
        auth_client.table('jobs').update({'interaction_results': interactions}).eq('id', job_id).execute()
        return {"job_id": job_id, "interactions": interactions}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/{job_id}/explain")
async def explain_results(job_id: str, request: dict, current_user: dict = Depends(get_current_user), credentials: HTTPAuthorizationCredentials = Security(security)):
    auth_client = get_authenticated_client(credentials.credentials)
    job = auth_client.table('jobs').select('*').eq('id', job_id).single().execute()
    if not job.data: raise HTTPException(404, "Not found")
    
    explainer = AIExplainer()
    async def generate():
        async for chunk in explainer.explain_results(job.data, job.data.get('docking_results'), job.data.get('interaction_results'), request.get('question')):
            yield chunk
    return StreamingResponse(generate(), media_type="text/event-stream")

async def _generate_job_export(job_id, current_user, credentials, format_type):
    auth_client = get_authenticated_client(credentials.credentials)
    job = auth_client.table('jobs').select('*').eq('id', job_id).single().execute()
    if not job.data: raise HTTPException(404, "Not found")
    
    job_data = job.data
    # Helper to clean up calling ExportService with non-None data
    analysis = job_data.get('docking_results') or {}
    interactions = job_data.get('interaction_results') or {}
    
    if format_type == 'pdf': return await ExportService.export_job_pdf(job_data, analysis, interactions)
    elif format_type == 'csv': return ExportService.export_jobs_csv([job_data])
    elif format_type == 'json': return ExportService.export_jobs_json([job_data])
    elif format_type == 'pymol':
        # Need Signed URLs for PyMOL to load remote files
        receptor_url = generate_presigned_download_url(job_id, job_data.get('receptor_filename', 'receptor.pdb'))
        # Try finding standard output keys
        ligand_url = generate_presigned_download_url(job_id, 'output.pdbqt')
        return ExportService.export_job_pymol(job_id, receptor_url, ligand_url)

@router.get("/{job_id}/export/pdf")
async def export_pdf(job_id: str, u=Depends(get_current_user), c=Security(security)): return await _generate_job_export(job_id, u, c, 'pdf')

@router.get("/{job_id}/export/csv")
async def export_csv(job_id: str, u=Depends(get_current_user), c=Security(security)): return await _generate_job_export(job_id, u, c, 'csv')

@router.get("/{job_id}/export/json")
async def export_json(job_id: str, u=Depends(get_current_user), c=Security(security)): return await _generate_job_export(job_id, u, c, 'json')

@router.get("/{job_id}/export/pymol")
async def export_pymol(job_id: str, u=Depends(get_current_user), c=Security(security)): return await _generate_job_export(job_id, u, c, 'pymol')
