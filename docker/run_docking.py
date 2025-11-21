#!/usr/bin/env python3
"""
CloudVina - Docking Runner Script
Orchestrates the complete docking workflow:
1. Download input files from S3
2. Convert ligand to PDBQT format
3. Run AutoDock Vina
4. Upload results back to S3
"""

import os
import sys
import subprocess
import json
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install via 'pip install boto3'")
    sys.exit(1)


class DockingRunner:
    def __init__(self):
        # Get job parameters from environment variables
        self.job_id = os.environ.get('JOB_ID')
        self.s3_bucket = os.environ.get('S3_BUCKET', 'cloudvina-jobs')
        self.receptor_key = os.environ.get('RECEPTOR_S3_KEY')
        self.ligand_key = os.environ.get('LIGAND_S3_KEY')
        
        # Validate required parameters
        if not all([self.job_id, self.receptor_key, self.ligand_key]):
            print("ERROR: Missing required environment variables")
            print(f"  JOB_ID: {self.job_id}")
            print(f"  RECEPTOR_S3_KEY: {self.receptor_key}")
            print(f"  LIGAND_S3_KEY: {self.ligand_key}")
            sys.exit(1)
        
        # Initialize S3 client
        self.s3 = boto3.client('s3')
        
        # Working directory
        self.work_dir = Path('/app/work')
        self.work_dir.mkdir(exist_ok=True)
        
        print(f"[CloudVina] Job ID: {self.job_id}")
        print(f"[CloudVina] S3 Bucket: {self.s3_bucket}")

    def download_from_s3(self, key: str, local_path: Path) -> bool:
        """Download file from S3"""
        try:
            print(f"[1/5] Downloading {key} from S3...")
            self.s3.download_file(self.s3_bucket, key, str(local_path))
            print(f"      ✓ Saved to {local_path}")
            return True
        except ClientError as e:
            print(f"      ✗ Failed to download {key}: {e}")
            return False

    def upload_to_s3(self, local_path: Path, key: str) -> bool:
        """Upload file to S3"""
        try:
            print(f"[5/5] Uploading {local_path.name} to S3...")
            self.s3.upload_file(str(local_path), self.s3_bucket, key)
            print(f"      ✓ Uploaded to s3://{self.s3_bucket}/{key}")
            return True
        except ClientError as e:
            print(f"      ✗ Failed to upload {local_path}: {e}")
            return False

    def convert_ligand_to_pdbqt(self, input_file: Path, output_file: Path) -> bool:
        """Convert ligand from SDF/MOL2/PDB to PDBQT using OpenBabel"""
        try:
            print(f"[2/5] Converting ligand to PDBQT format...")
            cmd = ['obabel', str(input_file), '-O', str(output_file), '-h']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"      ✗ Conversion failed: {result.stderr}")
                return False
            
            print(f"      ✓ Converted to {output_file}")
            return True
        except Exception as e:
            print(f"      ✗ Error during conversion: {e}")
            return False

    def prepare_receptor(self, receptor_file: Path, output_file: Path) -> bool:
        """Prepare receptor (for now, just copy if already PDBQT)"""
        # In production, you'd use AutoDockTools to properly prepare the receptor
        # For MVP, we assume the user uploads a pre-prepared PDBQT
        try:
            print(f"[3/5] Preparing receptor...")
            
            if receptor_file.suffix == '.pdbqt':
                # Already in correct format
                subprocess.run(['cp', str(receptor_file), str(output_file)])
                print(f"      ✓ Receptor ready (PDBQT format)")
                return True
            else:
                # Convert PDB to PDBQT (basic conversion)
                cmd = ['obabel', str(receptor_file), '-O', str(output_file), '-h']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"      ✗ Conversion failed: {result.stderr}")
                    return False
                
                print(f"      ✓ Converted receptor to PDBQT")
                return True
        except Exception as e:
            print(f"      ✗ Error preparing receptor: {e}")
            return False

    def run_vina(self, receptor: Path, ligand: Path, output: Path, log: Path) -> bool:
        """Run AutoDock Vina docking"""
        try:
            print(f"[4/5] Running AutoDock Vina...")
            
            # Basic docking parameters (center of mass, 20x20x20 box)
            # In production, these would come from user input
            cmd = [
                'vina',
                '--receptor', str(receptor),
                '--ligand', str(ligand),
                '--out', str(output),
                '--log', str(log),
                '--center_x', '0',
                '--center_y', '0', 
                '--center_z', '0',
                '--size_x', '20',
                '--size_y', '20',
                '--size_z', '20',
                '--cpu', '1',  # AWS Free Tier t2.micro has 1 vCPU
                '--exhaustiveness', '8'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"      ✗ Vina failed: {result.stderr}")
                return False
            
            print(f"      ✓ Docking complete!")
            print(f"      ✓ Results saved to {output}")
            
            # Parse log for binding affinity
            if log.exists():
                with open(log) as f:
                    log_content = f.read()
                    if 'REMARK VINA RESULT:' in log_content:
                        lines = [l for l in log_content.split('\n') if 'REMARK VINA RESULT:' in l]
                        if lines:
                            print(f"      📊 Best affinity: {lines[0].split()[3]} kcal/mol")
            
            return True
        except Exception as e:
            print(f"      ✗ Error running Vina: {e}")
            return False

    def run(self):
        """Execute complete docking workflow"""
        print("\n" + "="*60)
        print("CloudVina - Molecular Docking Pipeline")
        print("="*60 + "\n")
        
        # Define file paths
        receptor_input = self.work_dir / 'receptor_input.pdb'
        ligand_input = self.work_dir / 'ligand_input.sdf'
        receptor_pdbqt = self.work_dir / 'receptor.pdbqt'
        ligand_pdbqt = self.work_dir / 'ligand.pdbqt'
        output_pdbqt = self.work_dir / 'output.pdbqt'
        log_file = self.work_dir / 'log.txt'
        
        # Step 1: Download input files
        if not self.download_from_s3(self.receptor_key, receptor_input):
            sys.exit(1)
        if not self.download_from_s3(self.ligand_key, ligand_input):
            sys.exit(1)
        
        # Step 2: Convert ligand to PDBQT
        if not self.convert_ligand_to_pdbqt(ligand_input, ligand_pdbqt):
            sys.exit(1)
        
        # Step 3: Prepare receptor
        if not self.prepare_receptor(receptor_input, receptor_pdbqt):
            sys.exit(1)
        
        # Step 4: Run docking
        if not self.run_vina(receptor_pdbqt, ligand_pdbqt, output_pdbqt, log_file):
            sys.exit(1)
        
        # Step 5: Upload results
        output_prefix = f'jobs/{self.job_id}'
        if not self.upload_to_s3(output_pdbqt, f'{output_prefix}/output.pdbqt'):
            sys.exit(1)
        if not self.upload_to_s3(log_file, f'{output_prefix}/log.txt'):
            sys.exit(1)
        
        # Create success marker
        success_file = self.work_dir / 'SUCCESS'
        success_file.write_text(f"Job {self.job_id} completed successfully\n")
        self.upload_to_s3(success_file, f'{output_prefix}/SUCCESS')
        
        print("\n" + "="*60)
        print("✅ DOCKING COMPLETE!")
        print("="*60)
        print(f"Results location: s3://{self.s3_bucket}/{output_prefix}/")
        print()


if __name__ == '__main__':
    runner = DockingRunner()
    runner.run()
