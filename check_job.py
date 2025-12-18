import requests
import sys

job_id = "4311cc68-a8b1-4cf2-be29-fc01bf13c9e5"
url = f"https://cloudvina-api.onrender.com/jobs/{job_id}"

# We don't have a token, but let's see what happens.
# Some endpoints might give partial info or 401.
try:
    print(f"Querying {url}...")
    res = requests.get(url)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")
except Exception as e:
    print(f"Error: {e}")
