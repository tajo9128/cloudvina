#!/usr/bin/env python3
"""
CloudVina - Docking Runner Script
Orchestrates the complete docking workflow:
1. Download input files from S3
2. Convert ligand to PDBQT format (supports .pdbqt, .sdf, .mol2)
3. Prepare receptor (convert to PDBQT if needed)
4. Run AutoDock Vina
5. Upload results back to S3
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
        print("--- CLOUDVINA RUNNER v2.0 (SDF SUPPORT) ---")
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
            if not local_path.exists():
                print(f"      ⚠️ File {local_path} does not exist, skipping upload")
                return False
                
            self.s3.upload_file(str(local_path), self.s3_bucket, key)
            print(f"      ✓ Uploaded to s3://{self.s3_bucket}/{key}")
            return True
        except Exception as e:
            print(f"      ✗ Failed to upload {key}: {e}")
            return False

    def convert_ligand_to_pdbqt(self, input_file: Path, output_file: Path) -> bool:
        """Convert ligand (SDF, MOL2, PDB) to PDBQT using OpenBabel"""
        print(f"[2/5] Preparing ligand...")
        
        if not input_file.exists():
            print(f"      ✗ Input file {input_file} not found")
            return False
            
        # Check if already PDBQT
        if input_file.suffix.lower() == '.pdbqt':
            print(f"      ℹ️  Input is already PDBQT")
            subprocess.run(['cp', str(input_file), str(output_file)])
            return True
            
        try:
            # Construct obabel command
            # -i <format> input_file -o pdbqt -O output_file --gen3d
            input_format = input_file.suffix.lower().lstrip('.')
            if input_format == 'sdf': input_format = 'sdf'
            elif input_format == 'mol2': input_format = 'mol2'
            elif input_format == 'pdb': input_format = 'pdb'
            
            cmd = ['obabel', f'-i{input_format}', str(input_file), '-o', 'pdbqt', '-O', str(output_file), '--gen3d', '-h']
            
            print(f"      Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"      ✗ OpenBabel conversion failed: {result.stderr}")
                return False
                
            if not output_file.exists() or output_file.stat().st_size == 0:
                print(f"      ✗ Output PDBQT file is empty or missing")
                return False
                
            print(f"      ✓ Converted {input_format.upper()} to PDBQT")
            return True
            
        except Exception as e:
            print(f"      ✗ Error converting ligand: {e}")
            return False

    def prepare_receptor(self, input_file: Path, output_file: Path) -> bool:
        """Prepare receptor (convert to PDBQT)"""
        print(f"[3/5] Preparing receptor...")
        
        if not input_file.exists():
            print(f"      ✗ Input file {input_file} not found")
            return False
            
        if input_file.suffix.lower() == '.pdbqt':
            print(f"      ℹ️  Input is already PDBQT")
            subprocess.run(['cp', str(input_file), str(output_file)])
            return True
            
        try:
            # Convert PDB to PDBQT using obabel
            # -xr: output as rigid molecule (no branches)
            cmd = ['obabel', str(input_file), '-O', str(output_file), '-xr', '-h']
            
            print(f"      Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"      ✗ OpenBabel conversion failed: {result.stderr}")
                return False
                
            print(f"      ✓ Converted receptor to PDBQT")
            return True
            
        except Exception as e:
            print(f"      ✗ Error preparing receptor: {e}")
            return False

    def run_vina(self, receptor: Path, ligand: Path, output: Path, log: Path, config: Path) -> bool:
        """Run AutoDock Vina docking using config file"""
        try:
            print(f"[4/5] Running AutoDock Vina...")
            
            # Verify inputs
            if not receptor.exists() or not ligand.exists():
                print("      ✗ Missing receptor or ligand PDBQT files")
                return False
            
            if not config.exists():
                print(f"      ⚠️ Config file not found at {config}, using default parameters")
                # Fallback to hardcoded defaults if config is missing
                cmd = [
                    'vina',
                    '--receptor', str(receptor),
                    '--ligand', str(ligand),
                    '--out', str(output),
                    '--center_x', '0',
                    '--center_y', '0', 
                    '--center_z', '0',
                    '--size_x', '20',
                    '--size_y', '20',
                    '--size_z', '20',
                    '--cpu', '1',
                    '--exhaustiveness', '8'
                ]
            else:
                # Use config file (RECOMMENDED)
                print(f"      ✓ Using config file: {config}")
                cmd = [
                    'vina',
                    '--config', str(config),
                    '--receptor', str(receptor),
                    '--ligand', str(ligand),
                    '--out', str(output)
                ]
            
            # Capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Write log
            with open(log, 'w') as f:
                f.write(result.stdout)
            
            if result.returncode != 0:
                print(f"      ✗ Vina failed: {result.stderr}")
                return False
            
            print(f"      ✓ Docking complete!")
            
            # Parse affinity
            if log.exists():
                with open(log) as f:
                    content = f.read()
                    for line in content.splitlines():
                        if line.strip().startswith('1'):
                            parts = line.split()
                            if len(parts) >= 2:
                                print(f"      📊 Best affinity: {parts[1]} kcal/mol")
                                break
            
            return True
        except Exception as e:
            print(f"      ✗ Error running Vina: {e}")
            return False

    def run(self):
        """Execute complete docking workflow"""
        print("\n" + "="*60)
        print("CloudVina - Molecular Docking Pipeline")
        print("="*60 + "\n")
        
        # Determine input file extensions based on S3 keys
        receptor_ext = Path(self.receptor_key).suffix
        ligand_ext = Path(self.ligand_key).suffix
        
        # Define file paths
        receptor_input = self.work_dir / f'receptor_input{receptor_ext}'
        ligand_input = self.work_dir / f'ligand_input{ligand_ext}'
        receptor_pdbqt = self.work_dir / 'receptor.pdbqt'
        ligand_pdbqt = self.work_dir / 'ligand.pdbqt'
        output_pdbqt = self.work_dir / 'output.pdbqt'
        log_file = self.work_dir / 'log.txt'
        config_file = self.work_dir / 'config.txt'
        
        # Step 1: Download input files
        if not self.download_from_s3(self.receptor_key, receptor_input):
            sys.exit(1)
        if not self.download_from_s3(self.ligand_key, ligand_input):
            sys.exit(1)
        
        # Step 1.5: Download config file (if exists)
        config_key = f'jobs/{self.job_id}/config.txt'
        print(f"[1.5/5] Downloading config file...")
        if not self.download_from_s3(config_key, config_file):
            print(f"      ⚠️ Config file not found, will use defaults")
        
        # Step 2: Convert ligand to PDBQT
        if not self.convert_ligand_to_pdbqt(ligand_input, ligand_pdbqt):
            sys.exit(1)
        
        # Step 3: Prepare receptor
        if not self.prepare_receptor(receptor_input, receptor_pdbqt):
            sys.exit(1)
        
        # Step 4: Run docking
        if not self.run_vina(receptor_pdbqt, ligand_pdbqt, output_pdbqt, log_file, config_file):
            sys.exit(1)
        
        # Step 5: Upload results
        output_prefix = f'jobs/{self.job_id}'
        if not self.upload_to_s3(output_pdbqt, f'{output_prefix}/output.pdbqt'):
            sys.exit(1)
        if not self.upload_to_s3(log_file, f'{output_prefix}/log.txt'):
            sys.exit(1)
        
        print("\n" + "="*60)
        print("✅ DOCKING COMPLETE!")
        print("="*60)
        print(f"Results location: s3://{self.s3_bucket}/{output_prefix}/")
        print()


if __name__ == '__main__':
    runner = DockingRunner()
    runner.run()
