import os
from supabase import create_client, Client

# Credentials (Recovered)
SUPABASE_URL = "https://ohzfktmtwmubyhvspexv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9oemZrdG10d211YnlodnNwZXh2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM4MTk2MjgsImV4cCI6MjA3OTM5NTYyOH0.v8qHeRx5jkL8iaNEbEP_NMIvvUk4oidwwW6PkXo_DVY"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def list_recent_jobs():
    try:
        print("Fetching recent jobs from Supabase...")
        # Fetch latest 10 jobs
        response = supabase.table('jobs').select('id, status, batch_job_id, created_at').order('created_at', desc=True).limit(10).execute()
        
        if not response.data:
            print("No jobs found.")
            return

        print(f"{'Job ID':<40} {'Status':<15} {'Created At':<25} {'AWS Batch ID'}")
        print("-" * 100)
        for job in response.data:
            print(f"{job['id']:<40} {job['status']:<15} {job['created_at']:<25} {job.get('batch_job_id', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_recent_jobs()
