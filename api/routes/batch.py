from fastapi import APIRouter, Depends, HTTPException, status, Security, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from auth import get_current_user, get_authenticated_client
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
import boto3
from rdkit import Chem
from meeko import MoleculePreparation
import io

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
    engine: Optional[str] = "vina"

def generate_batch_urls(batch_id: str, receptor_filename: str, ligand_filenames: List[str]):
    try:
        # IMPORTANT: Use jobs/{batch_id}/ path (not batches/) to match container expectations
        # Container script downloads from jobs/{JOB_ID}/ path
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
            ligand_ext = os.path.splitext(filename)[1].lower() or '.pdbqt'
            # Use jobs/{batch_id}/ligands/ path for batch jobs
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
    """
    Submit a batch of ligands for docking against one receptor.
    """
    print(f"DEBUG: Endpoint called: submit_batch (files)")
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
    print(f"DEBUG: Endpoint called: start_batch for {batch_id}")
    try:
        from services.config_generator import generate_vina_config
        from services.smiles_converter import pdb_to_pdbqt
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
                # CRITICAL FIX: Copy files from batch folder to job folder
                # Database has: jobs/{batch_id}/receptor and jobs/{batch_id}/ligands/file.pdbqt
                # Container expects: jobs/{job_id}/receptor_input and jobs/{job_id}/ligand_input
                
                job_id = job['id']
                print(f"DEBUG: Processing job {job_id}")
                print(f"DEBUG: Original receptor key: {job['receptor_s3_key']}")
                print(f"DEBUG: Original ligand key: {job['ligand_s3_key']}")
                
                # Define job-specific S3 keys (destination always PDBQT)
                job_receptor_key = f"jobs/{job_id}/receptor_input.pdbqt"
                job_ligand_key = f"jobs/{job_id}/ligand_input.pdbqt"
                
                # --- RECEPTOR PREPARATION ---
                # Check if we need to convert or copy
                if job['receptor_filename'].lower().endswith('.pdb'):
                     # Convert PDB -> PDBQT
                     print(f"DEBUG: Converting receptor {job['receptor_filename']} to PDBQT")
                     try:
                         # Download PDB content
                         pdb_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=job['receptor_s3_key'])
                         pdb_content = pdb_obj['Body'].read().decode('utf-8')
                         
                         # Convert
                         pdbqt_content, err = pdb_to_pdbqt(pdb_content, add_hydrogens=True)
                         if err or not pdbqt_content:
                             raise Exception(f"Receptor conversion failed: {err}")
                             
                         # Upload PDBQT
                         s3_client.put_object(
                             Bucket=S3_BUCKET,
                             Key=job_receptor_key,
                             Body=pdbqt_content.encode('utf-8')
                         )
                         print(f"DEBUG: Receptor conversion & upload successful")
                     except Exception as conv_err:
                         print(f"ERROR: Receptor preparation failed: {conv_err}")
                         raise
                
                elif 'batch' in job['receptor_s3_key'] or batch_id in job['receptor_s3_key']:
                    # Source is likely PDBQT, just copy
                    try:
                        print(f"DEBUG: Copying receptor from {job['receptor_s3_key']} to {job_receptor_key}")
                        s3_client.copy_object(
                            Bucket=S3_BUCKET,
                            CopySource={'Bucket': S3_BUCKET, 'Key': job['receptor_s3_key']},
                            Key=job_receptor_key
                        )
                        print(f"DEBUG: Receptor copy successful")
                    except Exception as copy_err:
                        print(f"ERROR: Failed to copy receptor: {copy_err}")
                        raise
                else:
                    # Already in correct location (CSV batch case, already converted)
                    print(f"DEBUG: Receptor already in correct location")
                    job_receptor_key = job['receptor_s3_key']


                # --- LIGAND PREPARATION ---
                if job['ligand_filename'].lower().endswith('.pdb'):
                     # Convert PDB -> PDBQT
                     print(f"DEBUG: Converting ligand {job['ligand_filename']} to PDBQT")
                     try:
                         # Download PDB content
                         pdb_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=job['ligand_s3_key'])
                         pdb_content = pdb_obj['Body'].read().decode('utf-8')
                         
                         # Convert
                         pdbqt_content, err = pdb_to_pdbqt(pdb_content, add_hydrogens=True)
                         if err or not pdbqt_content:
                             raise Exception(f"Ligand conversion failed: {err}")
                             
                         # Upload PDBQT
                         s3_client.put_object(
                             Bucket=S3_BUCKET,
                             Key=job_ligand_key,
                             Body=pdbqt_content.encode('utf-8')
                         )
                         print(f"DEBUG: Ligand conversion & upload successful")
                     except Exception as conv_err:
                         print(f"ERROR: Ligand preparation failed: {conv_err}")
                         raise

                elif 'batch' in job['ligand_s3_key'] or batch_id in job['ligand_s3_key']:
                    try:
                        print(f"DEBUG: Copying ligand from {job['ligand_s3_key']} to {job_ligand_key}")
                        s3_client.copy_object(
                            Bucket=S3_BUCKET,
                            CopySource={'Bucket': S3_BUCKET, 'Key': job['ligand_s3_key']},
                            Key=job_ligand_key
                        )
                        print(f"DEBUG: Ligand copy successful")
                    except Exception as copy_err:
                        print(f"ERROR: Failed to copy ligand: {copy_err}")
                        raise
                else:
                    # Already in correct location (CSV batch case)
                    print(f"DEBUG: Ligand already in correct location")
                    job_ligand_key = job['ligand_s3_key']
                
                # 1. Generate Config (uses job_id, will upload to jobs/{job_id}/config.txt)
                print(f"DEBUG: Generating config for job {job_id}")
                generate_vina_config(job_id, grid_params=grid_params)

                # 2. Submit to AWS with corrected S3 keys
                print(f"DEBUG: Submitting to AWS Batch")
                aws_job_id = submit_to_aws(
                    job_id,
                    job_receptor_key,
                    job_ligand_key,
                    engine=engine
                )
                print(f"DEBUG: AWS job ID: {aws_job_id}")

                # 3. Update Status
                auth_client.table('jobs').update({
                    'status': 'SUBMITTED',
                    'batch_job_id': aws_job_id
                }).eq('id', job_id).execute()

                started_count += 1
            except Exception as e:
                print(f"ERROR: Failed to start job {job['id']}: {str(e)}")
                import traceback
                traceback.print_exc()
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
    from services.smiles_converter import smiles_to_pdbqt, pdb_to_pdbqt
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
        
        
        # 2. Upload Receptor to S3 (Convert PDB -> PDBQT if needed)
        receptor_content = await receptor_file.read()
        if not receptor_content or len(receptor_content) == 0:
            raise HTTPException(status_code=400, detail="Receptor file is empty")
        
        receptor_ext = os.path.splitext(receptor_file.filename)[1].lower() or '.pdb'
        receptor_key = f"jobs/{batch_id}/receptor_input.pdbqt" # Always store as PDBQT, use jobs/ path
        
        try:
            # Check if conversion is needed (PDB -> PDBQT)
            final_content = receptor_content
            
            # If extension is .pdb or content looks like PDB (lines starting with ATOM but no ROOT/BRANCH), convert
            is_pdb_ext = receptor_ext.lower() == '.pdb'
            
            if is_pdb_ext:
                print(f"Converting receptor {receptor_file.filename} from PDB to PDBQT...")
                try:
                    pdb_string = receptor_content.decode('utf-8')
                    mol = Chem.MolFromPDBBlock(pdb_string)
                    if mol:
                        mol = Chem.AddHs(mol, addCoords=True) # Add explicit Hs with coordinates
                        preparator = MoleculePreparation()
                        preparator.prepare(mol)
                        pdbqt_string = preparator.write_pdbqt_string()
                        final_content = pdbqt_string.encode('utf-8')
                        print("Receptor conversion successful")
                    else:
                        print("Warning: RDKit PDB parsing failed (returned None), using original content")
                except Exception as conv_err:
                     print(f"Warning: Receptor conversion failed: {conv_err}. Using original content.")
            
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=receptor_key,
                Body=final_content
            )
            print(f"Successfully uploaded receptor to {receptor_key} ({len(final_content)} bytes)")
        except Exception as e:
            print(f"Failed to upload/convert receptor: {str(e)}")
            # Fail hard if we can't upload the receptor
            raise HTTPException(status_code=500, detail=f"Failed to process receptor: {str(e)}")
        
        # Auto-calculate receptor center if grid is at origin (0,0,0)
        # Use existing content for parsing
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
                # LOGIC: If one fails conversion, we record it but CONTINUE to the next one.
                # This ensures optimal yield (e.g. 9/10 run successfully).
                conversion_errors.append({"index": idx, "smiles": smiles[:30], "error": error})
                continue
            
            # Upload ligand PDBQT to S3
            job_id = str(uuid.uuid4())
            ligand_key = f"jobs/{job_id}/ligand_input.pdbqt"
            
            # IMPORTANT: Copy receptor to this job's folder
            # Each job needs its own receptor copy since container expects jobs/{JOB_ID}/receptor_input.pdbqt
            receptor_for_job = f"jobs/{job_id}/receptor_input.pdbqt"
            
            # Copy receptor from batch location to job location
            s3_client.copy_object(
                Bucket=S3_BUCKET,
                CopySource={'Bucket': S3_BUCKET, 'Key': receptor_key},
                Key=receptor_for_job
            )
            
            # Upload ligand
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
                'receptor_s3_key': receptor_for_job,  # Use job-specific receptor copy
                'ligand_s3_key': ligand_key,
                'receptor_filename': receptor_file.filename,
                'ligand_filename': f"{compound_name}.pdbqt"
            }
            
            auth_client.table('jobs').insert(job_data).execute()
            
            # Generate config and submit to AWS
            try:
                generate_vina_config(job_id, grid_params=grid_params)
                aws_job_id = submit_to_aws(job_id, receptor_for_job, ligand_key, engine=engine)
                
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
                # LOGIC: If submission fails, we mark this single job as FAILED but CONTINUE the batch.
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


@router.get("/{batch_id}", response_model=dict)
async def get_batch_details(
    batch_id: str,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get details for a specific batch, including all its jobs.
    """
    try:
        auth_client = get_authenticated_client(credentials.credentials)
        
        # 1. Fetch all jobs for this batch
        jobs_res = auth_client.table('jobs').select('*').eq('batch_id', batch_id).eq('user_id', current_user.id).execute()
        jobs = jobs_res.data
        
        if not jobs:
             raise HTTPException(status_code=404, detail="Batch not found")

        # 2. Calculate Stats
        total_jobs = len(jobs)
        completed_jobs = sum(1 for j in jobs if j['status'] == 'SUCCEEDED')
        failed_jobs = sum(1 for j in jobs if j['status'] == 'FAILED')
        pending_jobs = total_jobs - completed_jobs - failed_jobs
        
        # Best affinity logic
        best_affinity = None
        for job in jobs:
            if job.get('binding_affinity') is not None:
                try:
                    val = float(job['binding_affinity'])
                    if best_affinity is None or val < best_affinity:
                        best_affinity = val
                except:
                    pass

        return {
            "batch_id": batch_id,
            "created_at": jobs[0]['created_at'] if jobs else None,
            "stats": {
                "total": total_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "pending": pending_jobs,
                "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
                "best_affinity": best_affinity
            },
            "jobs": jobs
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch batch details: {str(e)}")
