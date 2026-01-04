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
            print("Config downloaded.")
            # Basic parsing to dict
            with open(config_file) as f:
                for line in f:
                    if '=' in line:
                        k, v = line.split('=', 1)
                        config_params[k.strip()] = v.strip()
        except ClientError:
            print("No config file, using defaults.")
        
        # Hardcode grid size to 20x20x20 for optimal results
        # FIXED: Respect Config Generator Auto-Boxing (User Feedback)
        if 'size_x' not in config_params: config_params['size_x'] = '20'
        if 'size_y' not in config_params: config_params['size_y'] = '20'
        if 'size_z' not in config_params: config_params['size_z'] = '20'
        
        # For consensus mode, create comprehensive config showing both engines
        if self.engine_type == 'consensus':
            consensus_config_path = self.work_dir / 'consensus_config.txt'
            with open(consensus_config_path, 'w') as f:
                f.write("CONSENSUS DOCKING CONFIGURATION\n")
                f.write("="*60 + "\n\n")
                f.write("Grid Box Parameters (Shared by both engines):\n")
                f.write(f"  center_x = {config_params.get('center_x', '0')}\n")
                f.write(f"  center_y = {config_params.get('center_y', '0')}\n")
                f.write(f"  center_z = {config_params.get('center_z', '0')}\n")
                f.write(f"  size_x = 20  # Fixed\n")
                f.write(f"  size_y = 20  # Fixed\n")
                f.write(f"  size_z = 20  # Fixed\n")
                f.write(f"  exhaustiveness = {config_params.get('exhaustiveness', '8')}\n")
                f.write(f"  cpu = 1\n\n")
                f.write("AutoDock Vina:\n")
                f.write("  - Scoring: Classical force field\n")
                f.write("  - num_modes = 9\n\n")
                f.write("Gnina (AI):\n")
                f.write("  - Scoring: CNN + Classical\n")
                f.write("  - CNN Model: Default (dense)\n")
                f.write("  - num_modes = 9\n")
            # Upload the enhanced config
            self.upload_to_s3(consensus_config_path, f'jobs/{self.job_id}/config.txt')

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
                
            # Create comprehensive execution log
            log_path = self.work_dir / 'log.txt'
            with open(log_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("BIODOCKIFY EXECUTION LOG\n")
                f.write("="*80 + "\n\n")
                
                if result.get('consensus'):
                    f.write(f"Docking Mode: CONSENSUS (Vina + Gnina)\n")
                    f.write(f"Job ID: {self.job_id}\n\n")
                    
                    # Vina Section
                    if 'vina' in result.get('engines', {}):
                        f.write("\n" + "="*80 + "\n")
                        f.write("AUTODOCK VINA EXECUTION\n")
                        f.write("="*80 + "\n")
                        vina_res = result['engines']['vina']
                        if 'command' in vina_res:
                            f.write(f"\nCommand: {vina_res['command']}\n")
                        f.write(f"\nBest Affinity: {vina_res.get('best_affinity', 'N/A')} kcal/mol\n")
                        if vina_res.get('stdout'):
                            f.write(f"\nFull Vina Output:\n{'-'*80}\n")
                            f.write(vina_res['stdout'])
                            f.write("\n" + "-"*80 + "\n")
                   
                    # Gnina Section
                    if 'gnina' in result.get('engines', {}):
                        f.write("\n" + "="*80 + "\n")
                        f.write("GNINA (AI-POWERED CNN) EXECUTION\n")
                        f.write("="*80 + "\n")
                        gnina_res = result['engines']['gnina']
                        if 'command' in gnina_res:
                            f.write(f"\nCommand: {gnina_res['command']}\n")
                        f.write(f"\nBest Affinity: {gnina_res.get('best_affinity', 'N/A')} kcal/mol\n")
                        if gnina_res.get('cnn_score'):
                            f.write(f"CNN Score: {gnina_res['cnn_score']}\n")
                        if gnina_res.get('stdout'):
                            f.write(f"\nFull Gnina Output:\n{'-'*80}\n")
                            f.write(gnina_res['stdout'])
                            f.write("\n" + "-"*80 + "\n")
                    
                    # Summary
                    f.write("\n" + "="*80 + "\n")
                    f.write("CONSENSUS SUMMARY\n")
                    f.write("="*80 + "\n")
                    f.write(f"Average Affinity: {result.get('average_affinity', 'N/A')} kcal/mol\n")
                    f.write(f"Best Affinity: {result.get('best_affinity', 'N/A')} kcal/mol\n")
                    f.write(f"Output Files: output_vina.pdbqt, output_gnina.pdbqt\n")
                else:
                    f.write(f"Engine: {self.engine_type}\n")
                    f.write(f"Best Score: {result.get('best_affinity')}\n")
                    f.write(f"Output: {upload_key}\n")
                
            self.upload_to_s3(log_path, f'{output_prefix}/log.txt')
            
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            import traceback
            error_trace = traceback.format_exc()
            traceback.print_exc()
            
            # Upload error details to S3 for debugging
            try:
                error_log_path = self.work_dir / 'ERROR_LOG.txt'
                with open(error_log_path, 'w') as f:
                    f.write(f"ERROR: {str(e)}\n\n")
                    f.write(f"Job ID: {self.job_id}\n")
                    f.write(f"Engine: {self.engine_type}\n")
                    f.write(f"Receptor: {self.receptor_key}\n")
                    f.write(f"Ligand: {self.ligand_key}\n\n")
                    f.write("TRACEBACK:\n")
                    f.write(error_trace)
                
                # Upload using specific key format
                self.upload_to_s3(error_log_path, f'jobs/{self.job_id}/ERROR_LOG.txt')
                print("Error details uploaded to S3")
            except Exception as upload_err:
                print(f"Failed to upload error log: {upload_err}")
            
            sys.exit(1)

        print("\nPipeline Completed Successfully.")

if __name__ == '__main__':
    runner = DockingRunner()
    runner.run()
