from fastapi import APIRouter, Depends, HTTPException, status, Security, UploadFile, File
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

        # Check limits (MAX 100, Multiple of 10)
        num_files = len(request.ligand_filenames)
        if num_files > 100:
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
        engine = request.engine

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
                    job['ligand_s3_key'],
                    engine=engine
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


# ==================== CSV SMILES BATCH ENDPOINT ====================

class CSVBatchSubmitRequest(BaseModel):
    """Request model for CSV batch with grid params"""
    grid_center_x: float = 0.0
    grid_center_y: float = 0.0
    grid_center_z: float = 0.0
    grid_size_x: float = 20.0
    grid_size_y: float = 20.0
    grid_size_z: float = 20.0
    engine: str = "vina"


@router.post("/submit-csv", status_code=status.HTTP_201_CREATED)
async def submit_csv_batch(
    receptor_file: UploadFile = File(...),
    csv_file: UploadFile = File(...),
    grid_center_x: float = 0.0,
    grid_center_y: float = 0.0,
    grid_center_z: float = 0.0,
    grid_size_x: float = 20.0,
    grid_size_y: float = 20.0,
    grid_size_z: float = 20.0,
    engine: str = "vina",
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Submit a batch docking job from CSV containing SMILES.
    
    CSV must have a 'smiles' column. Optional 'name' column for compound names.
    Each SMILES is converted to PDBQT and docked.
    
    Maximum: 100 compounds per batch.
    """
    import pandas as pd
    import io
    from services.smiles_converter import smiles_to_pdbqt
    from services.config_generator import generate_vina_config
    from aws_services import submit_batch_job as submit_to_aws
    
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        batch_id = str(uuid.uuid4())
        
        # 1. Read and validate CSV
        csv_content = await csv_file.read()
        try:
            df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Check for smiles column (case-insensitive)
        smiles_col = None
        for col in df.columns:
            if col.lower() == 'smiles':
                smiles_col = col
                break
        
        if not smiles_col:
            raise HTTPException(status_code=400, detail="CSV must contain a 'smiles' column")
        
        smiles_list = df[smiles_col].dropna().tolist()
        
        if len(smiles_list) == 0:
            raise HTTPException(status_code=400, detail="No valid SMILES found in CSV")
        
        if len(smiles_list) > 50:
            raise HTTPException(status_code=400, detail=f"Maximum 50 compounds per batch. Found: {len(smiles_list)}")
        
        # Get optional name column
        name_col = None
        for col in df.columns:
            if col.lower() == 'name':
                name_col = col
                break
        
        # 2. Upload Receptor to S3
        receptor_content = await receptor_file.read()
        receptor_ext = os.path.splitext(receptor_file.filename)[1].lower() or '.pdb'
        receptor_key = f"batches/{batch_id}/receptor{receptor_ext}"
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=receptor_key,
            Body=receptor_content
        )
        
        # Auto-calculate receptor center if grid is at origin (0,0,0)
        if grid_center_x == 0 and grid_center_y == 0 and grid_center_z == 0:
            try:
                # Parse PDB/PDBQT to find geometric center
                coords = []
                for line in receptor_content.decode('utf-8').split('\n'):
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            coords.append((x, y, z))
                        except (ValueError, IndexError):
                            continue
                
                if coords:
                    grid_center_x = sum(c[0] for c in coords) / len(coords)
                    grid_center_y = sum(c[1] for c in coords) / len(coords)
                    grid_center_z = sum(c[2] for c in coords) / len(coords)
                    print(f"Auto-calculated receptor center: ({grid_center_x:.2f}, {grid_center_y:.2f}, {grid_center_z:.2f})")
            except Exception as e:
                print(f"Failed to auto-calculate center: {e}")
        
        # 3. Convert each SMILES to PDBQT and create jobs
        grid_params = {
            'grid_center_x': grid_center_x,
            'grid_center_y': grid_center_y,
            'grid_center_z': grid_center_z,
            'grid_size_x': grid_size_x,
            'grid_size_y': grid_size_y,
            'grid_size_z': grid_size_z
        }
        
        jobs_created = []
        conversion_errors = []
        
        for idx, smiles in enumerate(smiles_list):
            # Get name for this compound
            if name_col and idx < len(df) and pd.notna(df.iloc[idx].get(name_col)):
                compound_name = str(df.iloc[idx][name_col])
            else:
                compound_name = f"compound_{idx + 1}"
            
            # Convert SMILES to PDBQT
            pdbqt_content, error = smiles_to_pdbqt(smiles, compound_name)
            
            if error:
                conversion_errors.append({"index": idx, "smiles": smiles[:30], "error": error})
                continue
            
            # Upload ligand PDBQT to S3
            job_id = str(uuid.uuid4())
            ligand_key = f"jobs/{job_id}/ligand.pdbqt"
            
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=ligand_key,
                Body=pdbqt_content.encode('utf-8')
            )
            
            # Create job record
            job_data = {
                'id': job_id,
                'user_id': current_user.id,
                'status': 'PENDING',
                'batch_id': batch_id,
                'receptor_s3_key': receptor_key,
                'ligand_s3_key': ligand_key,
                'receptor_filename': receptor_file.filename,
                'ligand_filename': f"{compound_name}.pdbqt"
                # Note: smiles stored in ligand_filename prefix for reference
            }
            
            auth_client.table('jobs').insert(job_data).execute()
            
            # Generate config and submit to AWS
            try:
                generate_vina_config(job_id, grid_params=grid_params)
                aws_job_id = submit_to_aws(job_id, receptor_key, ligand_key, engine=engine)
                
                auth_client.table('jobs').update({
                    'status': 'SUBMITTED',
                    'batch_job_id': aws_job_id
                }).eq('id', job_id).execute()
                
                jobs_created.append({
                    "job_id": job_id,
                    "compound_name": compound_name,
                    "smiles": smiles[:30] + "..." if len(smiles) > 30 else smiles
                })
            except Exception as e:
                print(f"Failed to submit job {job_id}: {e}")
                auth_client.table('jobs').update({
                    'status': 'FAILED',
                    'error_message': str(e)[:500]
                }).eq('id', job_id).execute()
        
        return {
            "batch_id": batch_id,
            "jobs_created": len(jobs_created),
            "conversion_errors": len(conversion_errors),
            "jobs": jobs_created[:10],  # Return first 10 for preview
            "errors": conversion_errors[:5] if conversion_errors else None,
            "message": f"Batch submitted: {len(jobs_created)} jobs started, {len(conversion_errors)} conversion errors"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CSV batch submission failed: {str(e)}")

