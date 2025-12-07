"""
Authentication utilities using Supabase
"""
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
import os

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Service Role Client (for Admin actions)
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def get_service_client() -> Client:
    """
    Get a Supabase client with Service Role privileges.
    REQUIRED for admin actions like create_user/delete_user.
    """
    if not SUPABASE_SERVICE_ROLE_KEY:
        # Fallback to standard key if service key not set, but warn/fail if permissions insufficient
        print("WARNING: SUPABASE_SERVICE_ROLE_KEY not set. Admin actions may fail.")
        return supabase
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

security = HTTPBearer()

def get_authenticated_client(token: str) -> Client:
    """
    Get a Supabase client authenticated with the given token.
    This ensures RLS policies are respected.
    """
    # Create a new client instance to avoid shared state issues
    # We use the same URL and Key, but inject the user's token
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Set the auth token for PostgREST requests
    client.postgrest.auth(token)
    
    return client

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    Dependency to get current authenticated user from JWT token
    """
    token = credentials.credentials
    
    try:
        # Verify token with Supabase
        user = supabase.auth.get_user(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return user.user
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}"
        )
