
import os
import sys
import asyncio
from dotenv import load_dotenv

# Add api directory to path to import backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

# Load env from .env file in root or api
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'api', '.env'))

from supabase import create_client

async def promote_user(email: str):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("Error: Missing Supabase credentials")
        return

    supabase = create_client(url, key)
    
    # 1. Find User
    print(f"Searching for user: {email}...")
    # We can query 'profiles' directly for email if it's there, or use auth admin if needed
    # Assuming 'profiles' has email sync'd
    res = supabase.table("profiles").select("*").eq("email", email).execute()
    
    if not res.data:
        print("User not found in 'profiles' table.")
        return

    user_id = res.data[0]['id']
    print(f"Found User ID: {user_id}")
    
    # 2. Update is_admin
    print("Promoting to Admin...")
    update_res = supabase.table("profiles").update({"is_admin": True, "role": "admin"}).eq("id", user_id).execute()
    
    if update_res.data:
        print(f"SUCCESS: {email} is now an Admin.")
    else:
        print("Failed to update profile.")

if __name__ == "__main__":
    email = "biodockify@hotmail.com"
    asyncio.run(promote_user(email))
