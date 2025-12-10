from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any
from app.auth import get_current_user, get_supabase_client

router = APIRouter()

# Schema for Token Exchange (if needed)
class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(payload: LoginRequest):
    """
    Login user via Supabase Auth and return session/token.
    """
    supabase = get_supabase_client()
    try:
        res = supabase.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/register")
def register(payload: LoginRequest):
    """
    Sign up new user via Supabase Auth.
    """
    supabase = get_supabase_client()
    try:
        res = supabase.auth.sign_up({
            "email": payload.email,
            "password": payload.password
        })
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me")
def read_users_me(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user profile.
    """
    return user
