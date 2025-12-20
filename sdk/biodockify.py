import requests
import time
import os
import json
from typing import Dict, Any, Optional

class BioDockify:
    """
    Official Python Client for BioDockify.
    Automate your molecular docking pipelines.
    """
    
    def __init__(self, email: str, password: str, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.token = None
        self.user = None
        self.authenticate(email, password)

    def authenticate(self, email: str, password: str):
        """Login to get access token"""
        url = f"{self.base_url}/auth/login"
        try:
            resp = requests.post(url, json={"email": email, "password": password})
            resp.raise_for_status()
            data = resp.json()
            self.token = data['access_token']
            self.user = data['user']
            print(f"✅ Authenticated as {email}")
        except Exception as e:
            raise Exception(f"Authentication failed: {e}")

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def submit_job(self, receptor_path: str, ligand_path: str, config: Dict[str, Any] = None) -> str:
        """
        Submit a docking job.
        Returns: job_id
        """
        if not config:
            config = {
                "center_x": 0, "center_y": 0, "center_z": 0,
                "size_x": 20, "size_y": 20, "size_z": 20,
                "exhaustiveness": 8
            }
            
        receptor_filename = os.path.basename(receptor_path)
        ligand_filename = os.path.basename(ligand_path)
        
        # 1. Request Upload URLs
        print("🚀 Initiating job submission...")
        resp = requests.post(
            f"{self.base_url}/jobs/submit",
            headers=self._get_headers(),
            json={
                "receptor_filename": receptor_filename,
                "ligand_filename": ligand_filename,
                "config": config
            }
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data['job_id']
        upload_urls = data['upload_urls']
        
        # 2. Upload Files to S3
        print(f"📤 Uploading files for Job {job_id}...")
        self._upload_file(upload_urls['receptor'], receptor_path)
        self._upload_file(upload_urls['ligand'], ligand_path)
        
        # 3. Start Job
        print("🏁 Starting docking engine...")
        start_resp = requests.post(
            f"{self.base_url}/jobs/{job_id}/start",
            headers=self._get_headers(),
            json={"grid_params": config, "engine": "vina"}
        )
        start_resp.raise_for_status()
        print(f"✅ Job {job_id} started successfully!")
        
        return job_id

    def _upload_file(self, url_info: Dict, file_path: str):
        """Helper to upload file using presigned URL"""
        with open(file_path, 'rb') as f:
            # If urls are returned as a dictionary (AWS presigned post), logic differs
            # But our API returns direct PUT urls usually? Let's check api/aws_services.py
            # Actually generate_presigned_upload_urls returns: {'url': ..., 'fields': ...} for POST
            
            if 'fields' in url_info:
                # POST upload
                files = {'file': (os.path.basename(file_path), f)}
                response = requests.post(url_info['url'], data=url_info['fields'], files=files)
            else:
                # PUT upload (fallback)
                response = requests.put(url_info, data=f)
            
            if response.status_code not in [200, 204]:
                raise Exception(f"Failed to upload {file_path}: {response.text}")

    def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> Dict[str, Any]:
        """Block until job is complete"""
        print(f"⏳ Waiting for Job {job_id}...")
        while True:
            resp = requests.get(f"{self.base_url}/jobs/{job_id}", headers=self._get_headers())
            if resp.status_code != 200:
                print(f"Warning: Could not check status ({resp.status_code})")
                time.sleep(poll_interval)
                continue
                
            data = resp.json()
            status = data['status']
            
            if status == 'SUCCEEDED':
                print(f"🎉 Job Succeeded! Affinity: {data.get('binding_affinity')}")
                return data
            elif status == 'FAILED':
                raise Exception(f"Job Failed: {data.get('error_message')}")
            
            print(f"Status: {status}...")
            time.sleep(poll_interval)

# Example Usage
if __name__ == "__main__":
    client = BioDockify("user@example.com", "password123")
    # client.submit_job("protein.pdb", "drug.sdf")
