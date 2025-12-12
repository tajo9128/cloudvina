import sys
import os
import json
import time
from unittest.mock import MagicMock, patch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock heavy dependencies before import
sys.modules['supabase'] = MagicMock()

# Set dummy env vars for Auth
os.environ['SUPABASE_URL'] = 'https://dummy.supabase.co'
os.environ['SUPABASE_KEY'] = 'dummy-key'

# Mock Routers to avoid relative import issues in test environment
mock_router = MagicMock()

# 1. API Prefix Mocks
sys.modules['api.routes.md'] = MagicMock(router=mock_router)
sys.modules['api.routes.pymol'] = MagicMock(router=mock_router)
sys.modules['api.routes.users'] = MagicMock(router=mock_router)
sys.modules['api.routes.admin'] = MagicMock(router=mock_router)
sys.modules['api.routes.ranking'] = MagicMock(router=mock_router)
sys.modules['api.routes.user_system'] = MagicMock(router=mock_router)
sys.modules['api.routes.reporting'] = MagicMock(router=mock_router)
sys.modules['api.services.target_prediction'] = MagicMock(router=mock_router)

# 2. Direct Import Mocks (for when main.py does "from routes...")
sys.modules['routes.md'] = MagicMock(router=mock_router)
sys.modules['routes.pymol'] = MagicMock(router=mock_router)
sys.modules['routes.users'] = MagicMock(router=mock_router)
sys.modules['routes.admin'] = MagicMock(router=mock_router)
sys.modules['routes.ranking'] = MagicMock(router=mock_router)
sys.modules['routes.user_system'] = MagicMock(router=mock_router)
sys.modules['routes.reporting'] = MagicMock(router=mock_router)
sys.modules['services.target_prediction'] = MagicMock(router=mock_router)

# Import App
from api.main import app, get_current_user

# Test Client
from fastapi.testclient import TestClient
client = TestClient(app)

# Override Auth
def mock_get_current_user():
    return MagicMock(id="test-user-123", email="test@biodockify.com")

app.dependency_overrides[get_current_user] = mock_get_current_user

def validate_batch(n=10):
    print(f"\nStarting Batch Validation for N={n} Consensus Jobs...")
    
    # Mock AWS and DB
    with patch('api.main.submit_batch_job') as mock_submit, \
         patch('api.main.generate_vina_config') as mock_config, \
         patch('api.main.get_authenticated_client') as mock_db:
            
        # Setup Mocks
        mock_submit.return_value = "batch-job-id-123"
        mock_config.return_value = "s3://config.txt"
        
        # Mock DB flow: create -> return id
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        # Mock 'insert' for create job
        def side_effect_insert(data):
            return MagicMock(data=[{'id': f'job-{int(time.time()*1000)}', 'created_at': '2025-01-01'}])
        
        mock_db_instance.table.return_value.insert.return_value.execute.side_effect = side_effect_insert
        
        # Mock 'select' for get job
        mock_db_instance.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {'id': 'job-123', 'status': 'SUBMITTED', 'receptor_s3_key': 'r.pdb', 'ligand_s3_key': 'l.pdbqt'}
        ]

        success_count = 0
        
        start_time = time.time()
        
        for i in range(n):
            # 1. Create Job
            res_create = client.post("/jobs/create", json={
                "receptor_filename": "receptor.pdb",
                "ligand_filename": "ligand.pdbqt",
                "description": f"Batch Test {i+1}"
            })
            
            if res_create.status_code != 200:
                print(f"X Job {i+1} Creation Failed: {res_create.text}")
                continue
                
            job_id = res_create.json()['job_id']
            
            # 2. Start Job (Consensus)
            res_start = client.post(f"/jobs/{job_id}/start", json={
                "grid_params": {"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 20, "size_y": 20, "size_z": 20},
                "engine": "consensus"
            })
            
            if res_start.status_code == 200:
                success_count += 1
                # print(f" OK Job {i+1} ({job_id}) Started (Consensus)")
            else:
                print(f"X Job {i+1} Start Failed: {res_start.text}")

        duration = time.time() - start_time
        print(f"Completed {n} jobs in {duration:.2f}s")
        print(f"Success Rate: {success_count}/{n}")
        
        if success_count == n:
            print("BATCH VALIDATION PASSED")
        else:
            print("BATCH VALIDATION FAILED")

if __name__ == "__main__":
    print("Running Scalability Test: 10 to 100 Jobs")
    for load in range(10, 101, 10):
        print(f"\n--- Batch Size: {load} ---")
        validate_batch(load)
        time.sleep(1) # Cool down
from unittest.mock import MagicMock, patch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock heavy dependencies before import
sys.modules['supabase'] = MagicMock()

# Set dummy env vars for Auth
os.environ['SUPABASE_URL'] = 'https://dummy.supabase.co'
os.environ['SUPABASE_KEY'] = 'dummy-key'

# Mock Routers to avoid relative import issues in test environment
mock_router = MagicMock()

# 1. API Prefix Mocks
sys.modules['api.routes.md'] = MagicMock(router=mock_router)
sys.modules['api.routes.pymol'] = MagicMock(router=mock_router)
sys.modules['api.routes.users'] = MagicMock(router=mock_router)
sys.modules['api.routes.admin'] = MagicMock(router=mock_router)
sys.modules['api.routes.ranking'] = MagicMock(router=mock_router)
sys.modules['api.routes.user_system'] = MagicMock(router=mock_router)
sys.modules['api.routes.reporting'] = MagicMock(router=mock_router)
sys.modules['api.services.target_prediction'] = MagicMock(router=mock_router)

# 2. Direct Import Mocks (for when main.py does "from routes...")
sys.modules['routes.md'] = MagicMock(router=mock_router)
sys.modules['routes.pymol'] = MagicMock(router=mock_router)
sys.modules['routes.users'] = MagicMock(router=mock_router)
sys.modules['routes.admin'] = MagicMock(router=mock_router)
sys.modules['routes.ranking'] = MagicMock(router=mock_router)
sys.modules['routes.user_system'] = MagicMock(router=mock_router)
sys.modules['routes.reporting'] = MagicMock(router=mock_router)
sys.modules['services.target_prediction'] = MagicMock(router=mock_router)

# 3. Service Mocks
mock_config_gen = MagicMock()
sys.modules['services.config_generator'] = mock_config_gen
sys.modules['api.services.config_generator'] = mock_config_gen

# Import App
from api.main import app, get_current_user

# Test Client
from fastapi.testclient import TestClient
client = TestClient(app)

# Override Auth
def mock_get_current_user():
    return MagicMock(id="test-user-123", email="test@biodockify.com")

app.dependency_overrides[get_current_user] = mock_get_current_user

def validate_batch(n=10):
    print(f"\n> Starting Batch Validation for N={n} Consensus Jobs...")
    
    # Mock AWS and DB
    with patch('api.main.submit_batch_job') as mock_submit, \
         patch.object(mock_config_gen, 'generate_vina_config') as mock_config_func, \
         patch('api.main.get_authenticated_client') as mock_db:
         
        # Setup Mocks
        mock_submit.return_value = "batch-job-id-123"
        mock_config_func.return_value = "s3://config.txt"
        
        # Mock DB flow: create -> return id
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        # Mock 'insert' for create job
        def side_effect_insert(data):
            return MagicMock(data=[{'id': f'job-{int(time.time()*1000)}', 'created_at': '2025-01-01'}])
        
        mock_db_instance.table.return_value.insert.return_value.execute.side_effect = side_effect_insert
        
        # Mock 'select' for get job
        mock_db_instance.table.return_value.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
            {'id': 'job-123', 'status': 'SUBMITTED', 'receptor_s3_key': 'r.pdb', 'ligand_s3_key': 'l.pdbqt'}
        ]

        success_count = 0
        
        start_time = time.time()
        
        for i in range(n):
            # 1. Create Job
            res_create = client.post("/jobs/create", json={
                "receptor_filename": "receptor.pdb",
                "ligand_filename": "ligand.pdbqt",
                "description": f"Batch Test {i+1}"
            })
            
            if res_create.status_code != 200:
                print(f"❌ Job {i+1} Creation Failed: {res_create.text}")
                continue
                
            job_id = res_create.json()['job_id']
            
            # 2. Start Job (Consensus)
            res_start = client.post(f"/jobs/{job_id}/start", json={
                "grid_params": {"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 20, "size_y": 20, "size_z": 20},
                "engine": "consensus"
            })
            
            if res_start.status_code == 200:
                success_count += 1
                # print(f"✅ Job {i+1} ({job_id}) Started (Consensus)")
            else:
                print(f"❌ Job {i+1} Start Failed: {res_start.text}")

        duration = time.time() - start_time
        if success_count == n:
            print("BATCH VALIDATION PASSED")
        else:
            print("BATCH VALIDATION FAILED")

if __name__ == "__main__":
    print("Running Scalability Test: 10 to 100 Jobs")
    for load in range(10, 101, 10):
        print(f"\n--- Batch Size: {load} ---")
        validate_batch(load)
        time.sleep(1) # Cool down
