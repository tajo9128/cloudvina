#!/usr/bin/env python3
"""
CloudVina - Docking Runner Script
Orchestrates the complete docking workflow:
1. Download input files from S3
2. Convert ligand (supports .pdbqt, .sdf, .mol2)
3. Prepare receptor
4. Run Docking (Vina or rDock via ODDT)
5. Upload results back to S3
"""

import os
import sys
import subprocess
import json
import csv
from pathlib import Path

# Add /app to python path to import services
sys.path.append('/app')

try:
    import boto3
    from botocore.exceptions import ClientError
    # Import Unified Docking Engine
    from services.docking_engine import DockingEngine
except ImportError as e:
    print(f"ERROR: Import failed: {e}")
    # Fallback to local import if testing outside docker
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
        from services.docking_engine import DockingEngine
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("CRITICAL: Services or Boto3 not found.")
        sys.exit(1)


class DockingRunner:
    def __init__(self):
        print("--- CLOUDVINA RUNNER v3.0 (ODDT/rDock/Vina) ---")
        # Get job parameters
        self.job_id = os.environ.get('JOB_ID')
        self.s3_bucket = os.environ.get('S3_BUCKET', 'cloudvina-jobs')
        self.receptor_key = os.environ.get('RECEPTOR_S3_KEY')
        self.ligand_key = os.environ.get('LIGAND_S3_KEY')
        self.engine_type = os.environ.get('DOCKING_ENGINE', 'consensus').lower()  # Default: ALL 3 engines
        
        # Validate required parameters
        if not all([self.job_id, self.receptor_key, self.ligand_key]):
            print("ERROR: Missing required environment variables")
            sys.exit(1)
        
        self.s3 = boto3.client('s3')
        self.work_dir = Path('/app/work')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Run] Job ID: {self.job_id}")
        print(f"[Run] Engine: {self.engine_type}")

    def download_from_s3(self, key: str, local_path: Path) -> bool:
        try:
            print(f"Downloading {key}...")
            self.s3.download_file(self.s3_bucket, key, str(local_path))
            return True
        except ClientError as e:
            print(f"Failed to download {key}: {e}")
            return False

    def upload_to_s3(self, local_path: Path, key: str) -> bool:
        try:
            print(f"Uploading {local_path.name}...")
            if not local_path.exists():
                print(f"File {local_path} missing!")
                return False
            self.s3.upload_file(str(local_path), self.s3_bucket, key)
            return True
        except Exception as e:
            print(f"Failed to upload {key}: {e}")
            return False

    def run(self):
        print("\n" + "="*60)
        print(f"CloudVina Pipeline {self.engine_type.upper()}")
        print("="*60 + "\n")
        
        # file extensions
        rec_ext = Path(self.receptor_key).suffix
        lig_ext = Path(self.ligand_key).suffix
        
        receptor_input = self.work_dir / f'receptor{rec_ext}'
        ligand_input = self.work_dir / f'ligand{lig_ext}'
        
        # Download inputs
        if not self.download_from_s3(self.receptor_key, receptor_input): sys.exit(1)
        if not self.download_from_s3(self.ligand_key, ligand_input): sys.exit(1)
        
        # Download config (optional)
        config_file = self.work_dir / 'config.txt'
        config_key = f'jobs/{self.job_id}/config.txt'
        config_params = {}
        
        try:
            self.s3.download_file(self.s3_bucket, config_key, str(config_file))
            # Parse simple Vina config for params if needed
            # For now, we assume defaults if parsing complex
            print("Config downloaded.")
            # Basic parsing to dict
            with open(config_file) as f:
                for line in f:
                    if '=' in line:
                        k, v = line.split('=', 1)
                        config_params[k.strip()] = v.strip()
        except ClientError:
            print("No config file, using defaults.")

        # Set output path
        # Vina -> pdbqt, rDock -> sdf usually
        output_filename = 'output.pdbqt' if self.engine_type == 'vina' else 'output.sdf'
        output_path = self.work_dir / output_filename
        
        try:
            # Initialize Engine
            engine = DockingEngine(self.engine_type)
            
            # RUN DOCKING
            # Engine handles file loading (via ODDT) and docking
            result = engine.run_docking(
                receptor_path=str(receptor_input),
                ligand_path=str(ligand_input),
                output_path=str(output_path),
                config=config_params
            )
            
            print(f"Docking Success! Best Affinity: {result['best_affinity']}")
             
            # Upload Result
            output_prefix = f'jobs/{self.job_id}'
            
            if result.get('consensus'):
                # Consensus Mode: Upload multiple files
                print("Uploading Consensus Results...")
                
                # 1. Upload individual engine outputs
                for eng, res in result.get('engines', {}).items():
                    eng_out = Path(res.get('output_file', ''))
                    if eng_out.exists():
                        # naming: output_vina.pdbqt, output_rdock.sdf etc.
                        key = f'{output_prefix}/{eng_out.name}'
                        self.upload_to_s3(eng_out, key)
                
                # 2. Upload aggregated results.json
                results_json_path = self.work_dir / 'results.json'
                with open(results_json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                self.upload_to_s3(results_json_path, f'{output_prefix}/results.json')
                
                # 2.5 Generate and Upload results.csv (Consensus Summary)
                try:
                    results_csv_path = self.work_dir / 'results.csv'
                    with open(results_csv_path, 'w', newline='') as csvfile:
                        fieldnames = ['Engine', 'Best Affinity', 'CNN Score', 'Output File', 'Note']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        # Write rows for each engine
                        for eng, res in result.get('engines', {}).items():
                            writer.writerow({
                                'Engine': eng,
                                'Best Affinity': res.get('best_affinity', 'N/A'),
                                'CNN Score': res.get('cnn_score', 'N/A'),
                                'Output File': Path(res.get('output_file', '')).name,
                                'Note': 'Simulated Unit' if eng == 'rdock' else 'kcal/mol'
                            })
                            
                        # Write Average row
                        writer.writerow({
                            'Engine': 'AVERAGE (Vina+Gnina)',
                            'Best Affinity': result.get('average_affinity', 'N/A'),
                            'CNN Score': '-',
                            'Output File': '-',
                            'Note': 'Consensus Metric'
                        })
                        
                    self.upload_to_s3(results_csv_path, f'{output_prefix}/results.csv')
                    print("Uploaded results.csv")
                except Exception as csv_err:
                    print(f"Failed to generate CSV: {csv_err}")

                # 3. Ensure primary "output.pdbqt" exists (Use Vina's) for basic compatibility
                # If Vina output exists
                vina_out = result.get('engines', {}).get('vina', {}).get('output_file')
                if vina_out and os.path.exists(vina_out):
                    self.upload_to_s3(Path(vina_out), f'{output_prefix}/output.pdbqt')
                
                upload_key = f'{output_prefix}/results.json' # Point log to JSON
                
            else:
                # Standard Single Engine Mode
                upload_key = f'{output_prefix}/{output_filename}'
                if not self.upload_to_s3(output_path, upload_key):
                    sys.exit(1)
                
            # Create a simple log file for compatibility
            log_path = self.work_dir / 'log.txt'
            with open(log_path, 'w') as f:
                f.write(f"Engine: {self.engine_type}\n")
                if result.get('consensus'):
                    f.write(f"Consensus Scores:\n")
                    for eng, res in result.get('engines', {}).items():
                        score = res.get('best_affinity', 'N/A')
                        f.write(f"  {eng}: {score}\n")
                    f.write(f"Average Vina/Gnina: {result.get('average_affinity', 'N/A')}\n")
                else:
                    f.write(f"Best Score: {result.get('best_affinity')}\n")
                f.write(f"Output: {upload_key}\n")
                
            self.upload_to_s3(log_path, f'{output_prefix}/log.txt')
            
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print("\nPipeline Completed Successfully.")

if __name__ == '__main__':
    runner = DockingRunner()
    runner.run()
