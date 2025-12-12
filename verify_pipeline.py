import requests
import time
import sys
import json
import random
import string

# Configuration
API_URL = "http://localhost:8000"  # or production URL if needed
TEST_EMAIL = f"test_e2e_{''.join(random.choices(string.ascii_lowercase, k=5))}@example.com"
TEST_PASSWORD = "testpassword123"

def print_step(step):
    print(f"\n[STEP] {step}...")

def run_e2e_test():
    session = requests.Session()

    # 1. Signup/Login
    print_step(f"Creating User {TEST_EMAIL}")
    try:
        # Try signup
        auth_url = f"{API_URL}/auth/signup"
        payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
        resp = session.post(auth_url, json=payload)
        
        if resp.status_code == 200:
            print("Signup successful")
        else:
            # If fail, maybe User exists, try Login (though random email should avoid this)
            print(f"Signup failed ({resp.status_code}), trying login...")
            login_url = f"{API_URL}/auth/token"
            data = {"username": TEST_EMAIL, "password": TEST_PASSWORD} # OAuth2 form data
            resp = session.post(login_url, data=data)
            if resp.status_code != 200:
                print(f"Login failed: {resp.text}")
                return False
            
        # Get Token
        token_data = resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            # For supabase auth, it might be different. 
            # Assuming our API returns token or we simulate Supabase client behavior
            # If backend uses direct Supabase Auth, we might need a workaround for creating users via admin key
            # But let's check if the API has a login proxy. 
            # If not, we might need to rely on existing user.
            pass
            
        print(f"Authenticated. Token: {access_token[:10]}...")
        headers = {"Authorization": f"Bearer {access_token}"}
        
    except Exception as e:
        print(f"Auth specific error: {e}")
        # Proceeding without auth (will fail if guarded) to test public endpoints or assuming dev mode
        headers = {}

    # 2. Upload Dummy Files
    print_step("Uploading Files")
    try:
        files = {
            'receptor': ('protein.pdb', 'DUMMY PROTEIN CONTENT', 'text/plain'),
            'ligands': ('ligands.sdf', 'DUMMY LIGAND CONTENT', 'text/plain')
        }
        # Assuming /jobs endpoint accepts multipart/form-data
        upload_resp = requests.post(f"{API_URL}/jobs", files=files, headers=headers)
        
        if upload_resp.status_code != 200:
            print(f"Job creation failed: {upload_resp.text}")
            return False
            
        job_data = upload_resp.json()
        job_id = job_data.get("id")
        print(f"Job Created: {job_id}")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

    # 3. Poll Status
    print_step("Polling Job Status (7 Phases)")
    start_time = time.time()
    while time.time() - start_time < 60: # 1 minute timeout
        status_resp = requests.get(f"{API_URL}/jobs/{job_id}", headers=headers)
        if status_resp.status_code != 200:
             print(f"Poll failed: {status_resp.text}")
             break
             
        status_data = status_resp.json()
        status = status_data.get("status")
        progress = status_data.get("progress", 0)
        
        # Check specific phases if available
        # Assuming response has 'phases_status' or similar
        phases = status_data.get("phases", {})
        
        print(f"Status: {status} | Progress: {progress}% | Phases Completed: {len([k for k,v in phases.items() if v=='completed'])}")
        
        if status == 'completed':
            print("\nSUCCESS: Job Completed!")
            return True
        if status == 'failed':
            print("\nFAILURE: Job Failed!")
            return False
            
        time.sleep(2)
        
    print("\nTIMEOUT: Job took too long")
    return False

if __name__ == "__main__":
    if run_e2e_test():
        sys.exit(0)
    else:
        sys.exit(1)
