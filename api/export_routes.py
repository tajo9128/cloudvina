from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from .auth import get_current_user, get_authenticated_client
from .services.export import ExportService

router = APIRouter(tags=["Export"])
security = HTTPBearer()

@router.get("/jobs/{job_id}/export/{format}")
async def export_single_job(
    job_id: str,
    format: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Export a single job to CSV, JSON, or PDF
    """
    try:
        from .services.vina_parser import parse_vina_log
        from .services.interaction_analyzer import InteractionAnalyzer
        import boto3
        import os
        
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Fetch job details
        response = auth_client.table('jobs').select('*').eq('id', job_id).eq('user_id', current_user.id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = response.data[0]
        
        # Get Analysis & Interaction Data
        analysis = job.get('docking_results')
        interactions = job.get('interaction_results')
        
        # Auto-trigger analysis if missing and job succeeded (Logic copied from main.py)
        if job['status'] == 'SUCCEEDED' and (not analysis or not interactions):
            s3 = boto3.client('s3')
            bucket = os.getenv('S3_BUCKET', 'BioDockify-jobs-use1-1763775915')
            
            try:
                # Parse docking results if missing
                if not analysis:
                    try:
                        log_obj = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}/log.txt")
                        log_content = log_obj['Body'].read().decode('utf-8')
                        analysis = parse_vina_log(log_content)
                        
                        # Save to DB
                        auth_client.table('jobs').update({
                            'best_affinity': analysis.get('best_affinity'),
                            'num_poses': analysis.get('num_poses'),
                            'energy_range_min': analysis.get('energy_range_min'),
                            'energy_range_max': analysis.get('energy_range_max'),
                            'docking_results': analysis
                        }).eq('id', job_id).execute()
                    except Exception as e:
                        print(f"Failed to parse log for PDF export: {e}")

                # Analyze interactions if missing
                if not interactions:
                    try:
                        # Try standard paths
                        receptor_key = job.get('receptor_s3_key') or f"jobs/{job_id}/receptor.pdb"
                        output_key = f"jobs/{job_id}/output.pdbqt"
                        
                        receptor_obj = s3.get_object(Bucket=bucket, Key=receptor_key)
                        receptor_content = receptor_obj['Body'].read().decode('utf-8')
                        
                        output_obj = s3.get_object(Bucket=bucket, Key=output_key)
                        output_content = output_obj['Body'].read().decode('utf-8')
                        
                        analyzer = InteractionAnalyzer()
                        interactions = analyzer.analyze_interactions(receptor_content, output_content)
                        
                        # Save to DB
                        auth_client.table('jobs').update({
                            'interaction_results': interactions
                        }).eq('id', job_id).execute()
                    except Exception as e:
                        print(f"Failed to analyze interactions for PDF export: {e}")
                        
            except Exception as e:
                print(f"Warning: Could not auto-generate analysis: {e}")
        
        if format == 'csv':
            return ExportService.export_jobs_csv([job])
        elif format == 'json':
            return ExportService.export_jobs_json([job])
        elif format == 'pdf':
            # PASS THE DATA!
            return ExportService.export_job_pdf(job, analysis, interactions)
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use csv, json, or pdf")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{format}")
async def export_all_jobs(
    format: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Export all user jobs to CSV or JSON
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        # Fetch all jobs
        response = auth_client.table('jobs').select('*').eq('user_id', current_user.id).order('created_at', desc=True).execute()
        jobs = response.data
        
        if format == 'csv':
            return ExportService.export_jobs_csv(jobs)
        elif format == 'json':
            return ExportService.export_jobs_json(jobs)
        elif format == 'pdf':
            raise HTTPException(status_code=400, detail="PDF export is only available for single jobs")
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use csv or json")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
