
import boto3
import os
import json
import logging
import traceback
import tempfile
import time
from datetime import datetime
from supabase import Client
from api.services.config_generator import generate_vina_config
from api.aws_services import submit_batch_job as submit_to_aws
from api.services.smiles_converter import convert_to_pdbqt, convert_receptor_to_pdbqt
from api.services.fda_service import fda_service
from api.utils.db import safe_update
from api.services.layer1_generator import Layer1Generator
import uuid

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("queue_processor")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")

class QueueProcessor:
    def __init__(self, supabase_client: Client):
        self.db = supabase_client
        self.s3 = boto3.client('s3', region_name=AWS_REGION)
        self.config_cache = {} # Cache batch configs to avoid repetitive S3 reads

    async def process_queue(self):
        """
        Main Loop: Consumes jobs from the 'QUEUED' state.
        Guarantees At-Least-Once processing.
        """
        try:
            # 1. Fetch NEXT available job
            # We use a simple select. In a multi-worker setup, we'd need row locking (SKIP LOCKED).
            # For this single-instance deployment, simple select is fine.
            response = self.db.table("jobs") \
                .select("*") \
                .eq("status", "QUEUED") \
                .order("created_at", desc=False) \
                .limit(1) \
                .execute()
            
            jobs = response.data or []
            if not jobs:
                return # Queue empty

            job = jobs[0]
            await self._process_single_job(job)

        except Exception as e:
            logger.error(f"Queue Scan Error: {e}")
            traceback.print_exc()

    async def _process_single_job(self, job):
        job_id = job['id']
        batch_id = job['batch_id']
        user_id = job['user_id']
        
        logger.info(f"🚀 [Queue] Picking up Job {job_id} (Batch {batch_id})")

        try:
            # A. Mark as PROCESSING immediately
            safe_update(self.db, "jobs", {"id": job_id}, {
                "status": "PROCESSING",
                "notes": "Queue: Starting Preparation...",
                "started_at": datetime.utcnow().isoformat()
            })

            # B. Fetch Configuration (Grid Params, Engine)
            # We stored this in S3 at "jobs/{batch_id}/batch_config.json" during start_batch
            config = self._get_batch_config(batch_id)
            grid_params = config.get('grid_params')
            engine = config.get('engine', 'consensus')

            # C. Execute Preparation Pipeline
            # Layer 1 Integration: We now generate an ensemble of receptors
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Download Raw Receptor
                safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Layer 1 (Ensemble Generation)...'})
                
                raw_rec_path = os.path.join(temp_dir, job['receptor_filename'])
                self.s3.download_file(S3_BUCKET, job['receptor_s3_key'], raw_rec_path)
                
                # 2. Layer 1: Ensemble vs Standard
                ensemble_mode = config.get('ensemble_mode', False)
                if ensemble_mode:
                    safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Layer 1 (Ensemble Mode)...'})
                    layer1 = Layer1Generator(raw_rec_path, job_id)
                    ensemble_pdbs = layer1.generate() # [Crystal, AF, NMA]
                else:
                    safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Standard Mode (Crystal Only)...'})
                    ensemble_pdbs = [raw_rec_path] # Just Crystal
                
                # 3. Prepare Ligand (Common for all)
                safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Preparing Ligand...'})
                
                # We reuse the logic to convert/upload ligand for the MAIN job first
                main_lig_key = f"jobs/{job_id}/ligand.pdbqt"
                
                # Helper for conversion (using thread/process logic inline or simplified)
                # We'll use a simplified synchronous call here for safety/clarity, relying on the fact 
                # that we are already in an async loop worker.
                # Actually, strictly better to use the imported functions.
                
                lig_local_path = os.path.join(temp_dir, job['ligand_filename'])
                self.s3.download_file(S3_BUCKET, job['ligand_s3_key'], lig_local_path)
                
                lig_pdbqt_content = None
                if job['ligand_filename'].lower().endswith('.pdbqt'):
                    with open(lig_local_path, 'r') as f: lig_pdbqt_content = f.read()
                else:
                    with open(lig_local_path, 'r') as f: content = f.read()
                    lig_pdbqt_content, err = convert_to_pdbqt(content, job['ligand_filename'])
                    if err: raise Exception(f"Ligand conversion failed: {err}")
                
                # Upload Main Ligand
                # We need to save it locally for reuse
                lig_pdbqt_path = os.path.join(temp_dir, "ligand.pdbqt")
                with open(lig_pdbqt_path, 'w') as f: f.write(lig_pdbqt_content)
                self.s3.upload_file(lig_pdbqt_path, S3_BUCKET, main_lig_key)

                # 4. Process Ensemble Loop
                for idx, rec_path in enumerate(ensemble_pdbs):
                    # Determine Job ID (Original vs Clone)
                    if idx == 0:
                        current_job_id = job_id
                        is_clone = False
                        current_rec_key = f"jobs/{current_job_id}/receptor.pdbqt"
                        current_lig_key = main_lig_key
                    else:
                        current_job_id = str(uuid.uuid4())
                        is_clone = True
                        current_rec_key = f"jobs/{current_job_id}/receptor.pdbqt"
                        current_lig_key = f"jobs/{current_job_id}/ligand.pdbqt"
                        
                        # CLONE JOB IN DB
                        # We must act fast to return status
                        try:
                            # Clone relevant fields
                            clone_data = {
                                'id': current_job_id,
                                'user_id': user_id,
                                'batch_id': batch_id,
                                'status': 'PROCESSING',
                                'receptor_filename': os.path.basename(rec_path),
                                'ligand_filename': job['ligand_filename'],
                                'receptor_s3_key': current_rec_key,
                                'ligand_s3_key': current_lig_key,
                                'notes': f"Ensemble Variant #{idx} (Layer 1)",
                                'created_at': datetime.utcnow().isoformat(),
                                'started_at': datetime.utcnow().isoformat()
                            }
                            self.db.table('jobs').insert(clone_data).execute()
                            logger.info(f"   + Created Clone Job {current_job_id} (Variant {idx})")
                            
                            # Upload Ligand for Clone (Copy)
                            self.s3.upload_file(lig_pdbqt_path, S3_BUCKET, current_lig_key)
                            
                        except Exception as clone_err:
                            logger.error(f"Failed to clone job {idx}: {clone_err}")
                            continue

                    # 5. Convert Receptor -> PDBQT
                    # rec_path is local PDB (from Layer 1)
                    with open(rec_path, 'r') as f: rec_content = f.read()
                    
                    # Convert
                    rec_pdbqt, r_err = convert_receptor_to_pdbqt(rec_content, os.path.basename(rec_path))
                    if r_err or not rec_pdbqt:
                        if is_clone:
                            # Fail the clone but continue
                            safe_update(self.db, "jobs", {"id": current_job_id}, {"status": "FAILED", "error_message": "Receptor conversion failed"})
                            continue
                        else:
                            raise Exception(f"Main receptor conversion failed: {r_err}")
                    
                    # Upload Receptor
                    local_rec_pdbqt = rec_path + ".pdbqt"
                    with open(local_rec_pdbqt, 'w') as f: f.write(rec_pdbqt)
                    self.s3.upload_file(local_rec_pdbqt, S3_BUCKET, current_rec_key)
                    
                    # 6. Generate Config
                    generate_vina_config(current_job_id, grid_params=grid_params, receptor_content=rec_pdbqt)
                    
                    # 7. Submit to AWS
                    aws_id = submit_to_aws(current_job_id, current_rec_key, current_lig_key, engine=engine)
                    
                    # 8. Update Status
                    safe_update(self.db, "jobs", {"id": current_job_id}, {
                        'status': 'SUBMITTED', 
                        'batch_job_id': aws_id, 
                        'notes': f'Batch Submitted (Variant {idx})'
                    })

            logger.info(f"✅ [Queue] Job {job_id} (and ensembles) submitted successfully.")

        except Exception as e:
            traceback.print_exc()
            logger.error(f"❌ [Queue] Job {job_id} failed: {e}")
            safe_update(self.db, "jobs", {"id": job_id}, {
                "status": "FAILED",
                "error_message": str(e),
                "notes": "Queue Processing Failed"
            })

    def _get_batch_config(self, batch_id):
        """Fetches batch config from S3, with caching"""
        if batch_id in self.config_cache:
            return self.config_cache[batch_id]
            
        key = f"jobs/{batch_id}/batch_config.json"
        try:
            obj = self.s3.get_object(Bucket=S3_BUCKET, Key=key)
            config = json.loads(obj['Body'].read().decode('utf-8'))
            self.config_cache[batch_id] = config
            return config
        except Exception as e:
            # Fallback for old batches without config file?
            # User shouldn't be running old batches in this new system immediately, 
            # but defaults are safe.
            logger.warning(f"Could not load config for batch {batch_id}: {e}")
            return {'grid_params': {}, 'engine': 'consensus'}
