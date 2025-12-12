#!/usr/bin/env python3
"""
Local Test Script for CloudVina Docker Container
This version bypasses S3 and uses local files for testing
"""

import subprocess
import sys
from pathlib import Path


def run_local_test():
    """Run a local docking test without AWS"""
    
    print("\n" + "="*60)
    print("CloudVina - Local Test Mode")
    print("="*60 + "\n")
    
    # Check if test files exist
    work_dir = Path('/app/work')
    receptor_file = work_dir / 'test_receptor.pdb'
    ligand_file = work_dir / 'test_ligand.sdf'
    
    if not receptor_file.exists():
        print("❌ ERROR: test_receptor.pdb not found in /app/work")
        print("   Please mount your test_data directory:")
        print("   docker run -v $(pwd)/test_data:/app/work ...")
        sys.exit(1)
    
    if not ligand_file.exists():
        print("❌ ERROR: test_ligand.sdf not found in /app/work")
        sys.exit(1)
    
    print("[1/4] Converting ligand to PDBQT...")
    cmd = ['obabel', str(ligand_file), '-O', str(work_dir / 'ligand.pdbqt'), '-h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Conversion failed: {result.stderr}")
        sys.exit(1)
    print("   ✓ Ligand converted")
    
    print("[2/4] Preparing receptor...")
    cmd = ['obabel', str(receptor_file), '-O', str(work_dir / 'receptor.pdbqt'), '-h']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Conversion failed: {result.stderr}")
        sys.exit(1)
    print("   ✓ Receptor prepared")
    
    print("[3/4] Running AutoDock Vina...")
    print("   (This may take 1-5 minutes...)")
    
    cmd = [
        'vina',
        '--receptor', str(work_dir / 'receptor.pdbqt'),
        '--ligand', str(work_dir / 'ligand.pdbqt'),
        '--out', str(work_dir / 'output.pdbqt'),
        '--log', str(work_dir / 'log.txt'),
        '--center_x', '1.5',
        '--center_y', '0',
        '--center_z', '0',
        '--size_x', '20',
        '--size_y', '20',
        '--size_z', '20',
        '--cpu', '1',
        '--exhaustiveness', '4'  # Faster for testing
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Vina failed: {result.stderr}")
        sys.exit(1)
    
    print("   ✓ Docking complete!")
    
    print("[4/4] Checking results...")
    
    output_file = work_dir / 'output.pdbqt'
    log_file = work_dir / 'log.txt'
    
    if output_file.exists():
        print(f"   ✓ Output saved: {output_file}")
    
    if log_file.exists():
        print(f"   ✓ Log saved: {log_file}")
        
        # Parse log for binding affinity
        with open(log_file) as f:
            log_content = f.read()
            if 'REMARK VINA RESULT:' in log_content:
                lines = [l for l in log_content.split('\n') if 'REMARK VINA RESULT:' in l]
                if lines:
                    print(f"\n   📊 Best binding affinity: {lines[0].split()[3]} kcal/mol")
    
    print("\n" + "="*60)
    print("✅ LOCAL TEST COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    for f in work_dir.glob('*.pdbqt'):
        print(f"  - {f.name}")
    print(f"  - log.txt")
    print("\nDocker container is working correctly! ✓")
    print()


if __name__ == '__main__':
    run_local_test()
