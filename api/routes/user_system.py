from fastapi import APIRouter, Depends, HTTPException, Body
from auth import get_current_user, supabase
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/user", tags=["User System"])

# --- Models ---

class UserProfileUpdate(BaseModel):
    phone: Optional[str] = None
    designation: Optional[str] = None
    organization: Optional[str] = None
    social_links: Optional[Dict[str, str]] = None

class SupportTicketCreate(BaseModel):
    subject: str
    category: str
    message: str

class EmailUpdate(BaseModel):
    email: EmailStr

# --- Routes ---

@router.get("/profile")
def get_user_profile(user: dict = Depends(get_current_user)):
    user_id = user['id']
    
    # Fetch Profile
    profile_res = supabase.table('profiles').select('*').eq('id', user_id).single().execute()
    profile = profile_res.data if profile_res.data else {}
    
    # Fetch Credits/Plan
    credits_res = supabase.table('user_credits').select('*').eq('user_id', user_id).single().execute()
    credits_data = credits_res.data if credits_res.data else {}
    
    # Fetch Email from Auth User object (passed from dependency)
    email = user.get('email')
    
    return {
        "profile": {**profile, "email": email},
        "billing": credits_data
    }

@router.put("/profile")
def update_user_profile(
    update_data: UserProfileUpdate, 
    user: dict = Depends(get_current_user)
):
    user_id = user['id']
    data = update_data.model_dump(exclude_unset=True)
    
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")
        
    res = supabase.table('profiles').update(data).eq('id', user_id).execute()
    if not res.data:
         # Create if not exists (edge case)
        data['id'] = user_id
        res = supabase.table('profiles').insert(data).execute()
        
    return {"message": "Profile updated", "data": res.data[0]}

@router.delete("/account")
def delete_user_account(user: dict = Depends(get_current_user)):
    user_id = user['id']
    
    # Supabase Auth Admin delete requires service role usually, 
    # but we can try calling rpc or just deleting from public tables 
    # and letting Auth remain (soft delete) or use admin API if available.
    # For now, we will clear profile data to "deactivate" access.
    # Ideally, this should call an Edge Function or Admin API.
    
    # Strategy: Delete public profile. Auth user deletion requires Service Key.
    # We will return instructions to contact support if automated deletion fails
    # OR we can assume `supabase.auth.admin` is available if we initialize it.
    
    # Deleting public data:
    supabase.table('profiles').delete().eq('id', user_id).execute()
    supabase.table('user_credits').delete().eq('user_id', user_id).execute()
    
    return {"message": "Account data deleted. Please contact support to permanently remove login credentials if needed."}


# --- Billing Routes ---

@router.get("/billing/history")
def get_billing_history(user: dict = Depends(get_current_user)):
    user_id = user['id']
    # Use jobs as billing history proxy
    res = supabase.table('jobs')\
        .select('id, created_at, job_type, status, protein_name, ligand_name')\
        .eq('user_id', user_id)\
        .order('created_at', desc=True)\
        .limit(50)\
        .execute()
        
    history = []
    for job in res.data:
        cost = 1 # Standard cost
        if job.get('job_type') == 'batch':
            cost = 10 # Example placeholder
            
        history.append({
            "date": job['created_at'],
            "activity": f"{job['job_type'].title()} - {job.get('protein_name', 'Unknown')}",
            "cost": -cost,
            "status": job['status']
        })
        
    return history


# --- Support Routes ---

@router.get("/support/tickets")
def get_support_tickets(user: dict = Depends(get_current_user)):
    user_id = user['id']
    res = supabase.table('support_tickets')\
        .select('*')\
        .eq('user_id', user_id)\
        .order('created_at', desc=True)\
        .execute()
    return res.data

@router.post("/support/tickets")
def create_support_ticket(
    ticket: SupportTicketCreate, 
    user: dict = Depends(get_current_user)
):
    user_id = user['id']
    data = {
        "user_id": user_id,
        "subject": ticket.subject,
        "category": ticket.category,
        "message": ticket.message,
        "status": "open"
    }
    
    res = supabase.table('support_tickets').insert(data).execute()
    return {"message": "Ticket created", "ticket": res.data[0]}
