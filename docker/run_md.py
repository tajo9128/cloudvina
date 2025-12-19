import os
import sys
import argparse
import boto3
import json
import joblib
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rms, rmsf

# OpenMM Imports
import openmm as mm
import openmm.app as app
from openmm import unit

# --- CONFIGURATION ---
S3_BUCKET = os.environ.get("BUCKET_NAME", "biodockify-md-stability-engine")
DB_TABLE_NAME = os.environ.get("DB_TABLE", "md_stability_jobs")
MODEL_FILE = "alzheimers_ensemble_model.pkl"

def setup_s3():
    return boto3.client('s3')

def download_file(s3_key, local_path):
    s3 = setup_s3()
    print(f"⬇️ Downloading {s3_key} from {S3_BUCKET}...")
    s3.download_file(S3_BUCKET, s3_key, local_path)

def upload_file(local_path, s3_key):
    s3 = setup_s3()
    print(f"⬆️ Uploading {local_path} to {S3_BUCKET}/{s3_key}...")
    s3.upload_file(local_path, S3_BUCKET, s3_key)

def run_simulation(pdb_file, steps=5000):
    """Runs a short OpenMM simulation (Energy Min + 10ps Production)"""
    print(f"🧪 Starting Simulation on {pdb_file}...")
    
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    # Create System (Implicit Solvent for Speed on Batch)
    system = forcefield.createSystem(pdb.topology, 
                                   nonbondedMethod=app.NoCutoff, 
                                   constraints=app.HBonds)
    
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    # Try GPU, fallback to CPU
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        print("   🚀 Using CUDA Platform")
    except:
        platform = mm.Platform.getPlatformByName('CPU')
        print("   ⚠️ Using CPU Platform (Slower)")
        
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    
    # Minimize
    print("   📉 Minimizing Energy...")
    simulation.minimizeEnergy()
    
    # Production
    print(f"   🏃 Running {steps} steps...")
    
    # Save Trajectory
    dcd_reporter = app.DCDReporter('trajectory.dcd', 100)
    simulation.reporters.append(dcd_reporter)
    simulation.step(steps)
    print("   ✅ Simulation Complete.")
    
    # Save Final PDB for Reference
    with open('final_frame.pdb', 'w') as f:
        app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)
        
    return 'trajectory.dcd', 'final_frame.pdb'

def analyze_trajectory(topology, trajectory):
    """Calculates RMSD and RMSF using MDAnalysis"""
    print("📊 Start Analysis...")
    u = mda.Universe(topology, trajectory)
    
    # 1. RMSD (Backbone)
    R = rms.RMSD(u, select="backbone")
    R.run()
    mean_rmsd = np.mean(R.rmsd[:, 2]) # Column 2 is the rmsd value
    print(f"   Measurements: Mean RMSD = {mean_rmsd:.2f} Å")
    
    # 2. RMSF (C-alpha)
    calphas = u.select_atoms("name CA")
    rmsfer = rmsf.RMSF(calphas)
    rmsfer.run()
    mean_rmsf = np.mean(rmsfer.rmsf)
    print(f"   Measurements: Mean RMSF = {mean_rmsf:.2f} Å")
    
    return mean_rmsd, mean_rmsf

def score_stability(rmsd, rmsf):
    """Uses the AI Ensemble to predict Stability Score"""
    print("🧠 Consulting AI Stability Council...")
    
    # Load Model
    if not os.path.exists(MODEL_FILE):
        print("   ❌ Model not found locally. Ensure it is COPY'd in Dockerfile.")
        return 0.0
        
    model = joblib.load(MODEL_FILE)
    
    # Predict
    # Input format: [[rmsd, rmsf]]
    prediction = model.predict([[rmsd, rmsf]])[0]
    score = round(prediction, 2)
    print(f"   🔮 Predicted Stability Score: {score}/100")
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--pdb_key", required=True)
    args = parser.parse_args()
    
    job_id = args.job_id
    
    try:
        # 1. Download Input
        local_pdb = "input.pdb"
        download_file(args.pdb_key, local_pdb)
        
        # 2. Run Simulation
        traj_file, final_pdb = run_simulation(local_pdb)
        
        # 3. Analyze
        rmsd, rmsf = analyze_trajectory(local_pdb, traj_file)
        
        # 4. Score
        score = score_stability(rmsd, rmsf)
        
        # 5. Upload Results & Update DB (or S3 Metadata)
        output_key = f"jobs/{job_id}/trajectory.dcd"
        upload_file(traj_file, output_key)
        
        # Save JSON Result
        result = {
            "job_id": job_id,
            "mean_rmsd": float(rmsd),
            "mean_rmsf": float(rmsf),
            "stability_score": float(score),
            "status": "SUCCESS"
        }
        
        with open("result.json", "w") as f:
            json.dump(result, f)
            
        print(f"✅ Job Done. Result: {json.dumps(result)}")
        
        # Optional: Direct DB Update if this script has DB access (or let Backend poll S3)
        # For Strict Isolation, we might just write to S3 and let a Lambda trigger update DB.
        upload_file("result.json", f"jobs/{job_id}/md_result.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
