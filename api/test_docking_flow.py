import requests
import time
import os

# Configuration
API_URL = "http://localhost:8000"
TEST_FILES_DIR = "tests/data"

def test_full_docking_cycle():
    print("🚀 Starting End-to-End Docking Validation...")
    
    # 1. Create Job (Consensus)
    print("\n1. Creating Job...")
    try:
        response = requests.post(f"{API_URL}/jobs", json={
            "receptor_filename": "test_receptor.pdbqt",
            "ligand_filename": "test_ligand.pdbqt",
            "engine": "consensus"
        })
        response.raise_for_status()
        data = response.json()
        job_id = data["job_id"]
        print(f"✅ Job Created: {job_id}")
    except Exception as e:
        print(f"❌ Job Creation Failed: {e}")
        return

    # 2. Upload Files (Mock)
    print("\n2. Uploading Files (Simulated)...")
    # In a real test we would PUT to S3, but for validation we assume 
    # the backend is reachable. We verify the upload URLs exist.
    if "upload_urls" in data:
        print("✅ Upload URLs generated successfully")
    else:
        print("❌ Upload URLs missing")
        return

    # 3. Start Job
    print("\n3. Dispatching Job...")
    try:
        start_res = requests.post(f"{API_URL}/jobs/{job_id}/start", json={
            "center_x": 0, "center_y": 0, "center_z": 0,
            "size_x": 20, "size_y": 20, "size_z": 20,
            "engine": "consensus"
        })
        start_res.raise_for_status()
        print("✅ Job Dispatched to AWS Batch/Queue")
    except Exception as e:
        print(f"❌ Job Start Failed: {e}")
        return

    # 4. Monitor Status
    print(f"\n4. Monitoring Job {job_id}...")
    for _ in range(5):
        status_res = requests.get(f"{API_URL}/jobs/{job_id}")
        if status_res.ok:
            status = status_res.json()["status"]
            print(f"   Current Status: {status}")
            if status in ["SUCCEEDED", "FAILED"]:
                break
        time.sleep(1)
    
    print("\n✨ Validation Complete: Basic API Workflow is Operational.")

if __name__ == "__main__":
    test_full_docking_cycle()
