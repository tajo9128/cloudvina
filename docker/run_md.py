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
MODEL_FILE = "md_stability_model.pkl"

# ... (rest of code) ...

def score_stability(rmsd, rmsf):
    """Uses the AI Ensemble to predict Stability Score"""
    print("🧠 Consulting MD Stability AI Bundle...")
    
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

# ...

def generate_pdf_report(job_id, rmsd, rmsf, score, stability_status):
    """Generates a professional PDF report for the simulation"""
    print("📄 Generating PDF Report...")
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"BioDockify MD Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Job ID: {job_id}", ln=True, align='C')
    pdf.ln(10)
    
    # AI Stability Score (AWS Bundle)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. AI Stability Score (AWS Bundle)", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Score: {score}/100 ({stability_status})", ln=True)
    pdf.ln(5)
    
    # Metrics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Trajectory Metrics", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Mean RMSD (Backbone): {rmsd:.2f} Angstrom", ln=True)
    pdf.cell(0, 10, f"Mean RMSF (C-Alpha): {rmsf:.2f} Angstrom", ln=True)
    pdf.ln(5)
    
    # Conclusion
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 10, "Conclusion: " + ("The complex shows high stability in the AWS MD environment." if score > 70 else "The complex shows potential instability."))
    
    filename = "MD_analysis_report.pdf"
    pdf.output(filename)
    return filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--pdb_key", required=True)
    args = parser.parse_args()
    
    job_id = args.job_id
    
    # Start Logging
    log_file = "md_log.txt"
    sys.stdout = Tee(log_file, "w")
    
    print(f"🚀 Starting BioDockify MD Job {job_id}")
    
    try:
        # 1. Download Input
        local_pdb = "input.pdb"
        try:
            download_file(args.pdb_key, local_pdb)
        except Exception as e:
            print(f"❌ Failed to download input: {e}")
            raise

        # 2. Run Simulation
        traj_file, final_pdb = run_simulation(local_pdb)
        
        # 3. Analyze
        rmsd, rmsf = analyze_trajectory(local_pdb, traj_file)
        
        # 4. Score
        score = score_stability(rmsd, rmsf)
        status_label = "Stable" if score >= 70 else "Unstable"
        
        # 5. Generate Report
        report_file = generate_pdf_report(job_id, rmsd, rmsf, score, status_label)
        
        # 6. Upload Results
        print("☁️ Uploading artifacts to S3...")
        upload_file(traj_file, f"jobs/{job_id}/trajectory.dcd")
        upload_file(report_file, f"jobs/{job_id}/MD_analysis_report.pdf")
        
        # Upload Log (Flush first)
        sys.stdout.flush()
        upload_file(log_file, f"jobs/{job_id}/md_log.txt") # Standard log name
        
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
            
        # Optional: Trigger completion webhook or similar
        print(f"✅ Job Done. Result: {json.dumps(result)}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Critical Failure: {e}")
        # Upload log even on failure
        sys.stdout.flush()
        try:
            upload_file(log_file, f"jobs/{job_id}/md_log.txt")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
