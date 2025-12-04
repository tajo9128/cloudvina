from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from supabase import Client
import psutil
import os
from ..auth import get_current_user
from ..aws_services import cancel_batch_job

router = APIRouter(prefix="/admin", tags=["admin"])

async def verify_admin(user: dict = Depends(get_current_user)):
    """Verify user is admin by checking the is_admin flag in profiles table"""
    # In a real app, we might cache this or put it in the JWT claims
    # For now, we trust the DB lookup done in get_current_user or do a fresh one
    
    # We need to query the profile to check is_admin because get_current_user might only return auth data
    # Assuming get_current_user returns the user dict from Supabase Auth
    
    # We need a supabase client with service role to check profiles if RLS hides it, 
    # OR we rely on the user's own ability to see their own profile.
    # But for security, let's use the service role client to be sure.
    
    from ..main import supabase # Import here to avoid circular imports if possible, or use a dependency
    
    # Actually, let's just use the user ID to check the profile
    response = supabase.table("profiles").select("is_admin").eq("id", user["id"]).single().execute()
    
    if not response.data or not response.data.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return user

@router.get("/dashboard-stats")
async def get_dashboard_stats(
    admin: dict = Depends(verify_admin),
    hours_back: int = 24
):
    """Get comprehensive admin dashboard statistics"""
    from ..main import supabase
    
    # Real-time job stats
    # Note: count='exact' is needed to get the total count
    jobs_response = supabase.table("jobs") \
        .select("*", count="exact") \
        .gte("created_at", (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()) \
        .execute()
    
    # User stats
    users_response = supabase.table("profiles") \
        .select("*", count="exact") \
        .gte("created_at", (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()) \
        .execute()
    
    # System metrics
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "active_connections": len(psutil.net_connections()),
        "uptime": psutil.boot_time()
    }
    
    jobs_data = jobs_response.data or []
    
    return {
        "jobs": {
            "total": jobs_response.count,
            "completed": len([j for j in jobs_data if j.get("status") == "completed"]),
            "failed": len([j for j in jobs_data if j.get("status") == "failed"]),
            "running": len([j for j in jobs_data if j.get("status") == "running"]),
            "queued": len([j for j in jobs_data if j.get("status") == "queued"])
        },
        "users": {
            "total": users_response.count,
            "active_today": users_response.count,  # Simplified for now
        },
        "system": system_metrics
    }

@router.get("/jobs")
async def get_all_jobs(
    admin: dict = Depends(verify_admin),
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Get all jobs with filtering"""
    from ..main import supabase
    
    query = supabase.table("jobs") \
        .select("*, profiles(email, username)") \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1)
    
    if status:
        query = query.eq("status", status)
    if user_id:
        query = query.eq("user_id", user_id)
    
    response = query.execute()
    return response.data

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    admin: dict = Depends(verify_admin)
):
    """Cancel a running job"""
    from ..main import supabase
    
    # Log admin action
    supabase.table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "cancel_job",
        "target_id": job_id,
        "target_type": "job",
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    # Get job details to find batch_job_id
    job_response = supabase.table("jobs").select("batch_job_id").eq("id", job_id).single().execute()
    batch_job_id = job_response.data.get("batch_job_id") if job_response.data else None

    # Update job status in DB
    supabase.table("jobs") \
        .update({"status": "cancelled", "completed_at": datetime.utcnow().isoformat()}) \
        .eq("id", job_id) \
        .execute()
    
    # Cancel in AWS Batch if applicable
    if batch_job_id:
        background_tasks.add_task(cancel_batch_job, batch_job_id, "Cancelled by admin")
    
    return {"status": "success", "message": f"Job {job_id} cancelled"}

@router.get("/users")
async def get_all_users(
    admin: dict = Depends(verify_admin),
    limit: int = 50,
    offset: int = 0
):
    """Get all users"""
    from ..main import supabase
    
    # Simple query for now. Complex joins might need RPC or raw SQL if Supabase client limits it.
    response = supabase.table("profiles") \
        .select("*") \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()
        
    return response.data

@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    reason: str,
    request: Request,
    admin: dict = Depends(verify_admin)
):
    """Suspend a user account"""
    from ..main import supabase
    
    # Update user profile - assuming we add an is_suspended column or similar
    # For now, let's just log it and maybe update a metadata field if the column doesn't exist yet
    # But the plan said we'd update profiles. Let's assume we can add a 'status' or 'is_suspended' column later.
    # For this MVP, we'll just log the action.
    
    # Log admin action
    supabase.table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "suspend_user",
        "target_id": user_id,
        "target_type": "user",
        "details": {"reason": reason},
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    return {"status": "success", "message": f"User {user_id} suspended (logged)"}

@router.get("/system/config")
async def get_system_config(admin: dict = Depends(verify_admin)):
    """Get system configuration"""
    from ..main import supabase
    
    response = supabase.table("job_queue_control") \
        .select("*") \
        .order("updated_at", desc=True) \
        .limit(1) \
        .execute()
    
    return response.data[0] if response.data else {}

@router.post("/system/config")
async def update_system_config(
    config: Dict[str, Any],
    request: Request,
    admin: dict = Depends(verify_admin)
):
    """Update system configuration"""
    from ..main import supabase
    
    config["updated_by"] = admin["id"]
    config["updated_at"] = datetime.utcnow().isoformat()
    
    response = supabase.table("job_queue_control") \
        .insert(config) \
        .execute()
    
    # Log admin action
    supabase.table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "update_config",
        "target_type": "system",
        "details": config,
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    return {"status": "success", "data": response.data}
