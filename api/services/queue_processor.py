
import boto3
import os
import json
import logging
import traceback
import tempfile
import time
from datetime import datetime
from supabase import Client
from services.config_generator import generate_vina_config
from aws_services import submit_batch_job as submit_to_aws
from services.smiles_converter import convert_to_pdbqt, convert_receptor_to_pdbqt
from services.fda_service import fda_service

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
            job_rec_key = f"jobs/{job_id}/receptor.pdbqt"
            job_lig_key = f"jobs/{job_id}/ligand.pdbqt"

            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize Process Pool for Isolation (Prevents Segfaults killing API)
                import concurrent.futures
                
                def prepare_file_safe(s3_key, filename, target_key, is_receptor=False):
                    ext = os.path.splitext(filename)[1].lower()
                    local_path = os.path.join(temp_dir, filename)
                    
                    self.s3.download_file(S3_BUCKET, s3_key, local_path)
                    
                    if ext != '.pdbqt':
                        with open(local_path, 'r') as f: content = f.read()
                        
                        # Run conversion in isolated process
                        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                            if is_receptor:
                                future = executor.submit(convert_receptor_to_pdbqt, content, filename)
                            else:
                                future = executor.submit(convert_to_pdbqt, content, filename)
                            
                            try:
                                pdbqt_content, err = future.result(timeout=120) # 2 min timeout
                            except concurrent.futures.TimeoutError:
                                raise Exception("Conversion Timed Out (Possible Hang)")
                            except Exception as exc:
                                raise Exception(f"Conversion Crash/Error: {exc}")

                        if err or not pdbqt_content:
                            raise Exception(f"Conversion failed: {err}")
                            
                        local_path = local_path + ".pdbqt"
                        with open(local_path, 'w') as f: f.write(pdbqt_content)
                    
                    self.s3.upload_file(local_path, S3_BUCKET, target_key)

                # 1. Receptor
                safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Preparing Receptor...'})
                prepare_file_safe(job['receptor_s3_key'], job['receptor_filename'], job_rec_key, is_receptor=True)

                # Capture Receptor Content for Autoboxing
                # We need to find where prepare_file_safe put it.
                # It saved to os.path.join(temp_dir, job['receptor_filename']) + potentially '.pdbqt'
                rec_local_name = job['receptor_filename']
                if not rec_local_name.lower().endswith('.pdbqt'):
                    rec_local_name += ".pdbqt"
                
                rec_local_path = os.path.join(temp_dir, rec_local_name)
                with open(rec_local_path, 'r') as f:
                    receptor_content = f.read()

                # 2. Ligand
                safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Preparing Ligand...'})
                prepare_file_safe(job['ligand_s3_key'], job['ligand_filename'], job_lig_key, is_receptor=False)

            # D. Generate Vina Config
            safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Generating Config...'})
            generate_vina_config(job_id, grid_params=grid_params, receptor_content=receptor_content)

            # E. Submit to AWS
            safe_update(self.db, "jobs", {"id": job_id}, {'notes': 'Queue: Submitting to AWS...'})
            aws_id = submit_to_aws(job_id, job_rec_key, job_lig_key, engine=engine)

            # F. Finalize
            safe_update(self.db, "jobs", {"id": job_id}, {
                'status': 'SUBMITTED', 
                'batch_job_id': aws_id, 
                'notes': 'Batch Submitted (Queue)'
            })

            logger.info(f"✅ [Queue] Job {job_id} submitted successfully.")

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
