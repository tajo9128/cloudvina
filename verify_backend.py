import requests
import sys

try:
    response = requests.get('http://localhost:8000/health')
    if response.status_code == 200:
        print("Backend is Healthy: OK")
        sys.exit(0)
    else:
        print(f"Backend Returned {response.status_code}: {response.text}")
        sys.exit(1)
except Exception as e:
    print(f"Backend Connection Failed: {e}")
    sys.exit(1)
