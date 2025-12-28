from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, Body
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from supabase import Client
import psutil
import os

# Use absolute imports
from auth import supabase, get_current_user, get_service_client
from aws_services import cancel_batch_job
from services.fda_service import fda_service
from services.rbac_service import rbac_service

router = APIRouter(prefix="/admin", tags=["admin"])

async def verify_admin(user: dict = Depends(get_current_user)):
    """Verify user is admin by checking the is_admin flag in profiles table"""
    
    # Use the user ID to check the profile
    # Use Service Client to see ALL data (bypass RLS)
    service_client = get_service_client()
    response = service_client.table("profiles").select("is_admin").eq("id", user.id).single().execute()
    
    if not response.data or not response.data.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return user

@router.get("/dashboard-stats")
async def get_dashboard_stats(
    admin: dict = Depends(verify_admin),
    hours_back: int = 168  # Changed from 24 to 168 (7 days)
):
    """Get comprehensive admin dashboard statistics and recent activity"""
    
    # Use Service Client to see ALL data (bypass RLS)
    service_client = get_service_client()

    # 1. Summary Stats (Counts) - ALL TIME
    # Real-time job stats
    jobs_response = service_client.table("jobs") \
        .select("*", count="exact") \
        .execute()
    
    # User stats
    users_response = service_client.table("profiles") \
        .select("*", count="exact") \
        .execute()
    
    # 2. Detailed Lists for "All-in-One" View
    # Recent Jobs (Top 5)
    recent_jobs = service_client.table("jobs") \
        .select("*, profiles(email, username)") \
        .order("created_at", desc=True) \
        .limit(5) \
        .execute()
        
    # Recent Users (Top 5)
    recent_users = service_client.table("profiles") \
        .select("*") \
        .order("created_at", desc=True) \
        .limit(5) \
        .execute()
        
    # Recent Activity (Top 10)
    # We try-catch this in case admin_actions doesn't exist yet for some deployments
    try:
        activity_log = service_client.table("admin_actions") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        activity_data = activity_log.data
    except:
        activity_data = []

    # 3. System Metrics
    try:
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "active_connections": len(psutil.net_connections()),
            "uptime": psutil.boot_time()
        }
    except Exception:
        # Fallback if psutil fails on certain platforms
        system_metrics = {
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_percent": 0,
            "active_connections": 0,
            "uptime": 0
        }
    
    jobs_data = jobs_response.data or []
    
    return {
        "stats": {
            "jobs": {
                "total": jobs_response.count,
                "completed": len([j for j in jobs_data if j.get("status") == "completed"]),
                "failed": len([j for j in jobs_data if j.get("status") == "failed"]),
                "running": len([j for j in jobs_data if j.get("status") == "running"]),
                "queued": len([j for j in jobs_data if j.get("status") == "queued"])
            },
            "users": {
                "total": users_response.count,
                "active_all_time": users_response.count,
            },
            "system": system_metrics
        },
        "recent_jobs": recent_jobs.data or [],
        "recent_users": recent_users.data or [],
        "activity_log": activity_data or []
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
    
    # Use Service Client
    service_client = get_service_client()
    
    query = service_client.table("jobs") \
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
    
    # Use Service Client
    service_client = get_service_client()

    # Log admin action
    service_client.table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "cancel_job",
        "target_id": job_id,
        "target_type": "job",
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    # Get job details to find batch_job_id
    job_response = service_client.table("jobs").select("batch_job_id").eq("id", job_id).single().execute()
    batch_job_id = job_response.data.get("batch_job_id") if job_response.data else None

    # Update job status in DB
    service_client.table("jobs") \
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
    
    # Use Service Client
    service_client = get_service_client()
    
    response = service_client.table("profiles") \
        .select("*") \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()
        
    return response.data

@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    reason: str = Body(..., embed=True),
    request: Request = None,
    admin: dict = Depends(verify_admin)
):
    """Suspend a user account"""
    service_client = get_service_client()
    
    # Apply suspension
    service_client.table("profiles").update({
        "role": "suspended",
        "notes": f"Suspended by admin: {reason}"
    }).eq("id", user_id).execute()

    # Log admin action
    supabase.table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "suspend_user",
        "target_id": user_id,
        "target_type": "user",
        "details": {"reason": reason},
        "ip_address": request.client.host if request else "unknown",
        "user_agent": request.headers.get("user-agent") if request else "unknown"
    }).execute()
    
    return {"status": "success", "message": f"User {user_id} suspended"}

@router.get("/jobs/{job_id}/details")
async def get_job_details(
    job_id: str,
    admin: dict = Depends(verify_admin)
):
    """Get deep audit details for a job (Config, Logs, Metadata)"""
    try:
        from services.export import ExportService
        
        # Use Service Client to see ALL data (bypass RLS)
        service_client = get_service_client()
        
        # 1. Fetch DB Record
        job_res = service_client.table("jobs").select("*").eq("id", job_id).single().execute()
        if not job_res.data: raise HTTPException(404, "Job not found")
        job = job_res.data

        details = {
            "db_record": job,
            "s3_config": None,
            "s3_logs": None
        }

        # 2. Try Fetch Config from S3 if batch
        if job.get('batch_id'):
            try:
                # Assuming standard batch config path
                # Note: We need boto3 here
                import boto3
                s3 = boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-1"))
                bucket = os.getenv("S3_BUCKET")
                
                # Try fetch batch_config.json
                key = f"jobs/{job['batch_id']}/batch_config.json"
                obj = s3.get_object(Bucket=bucket, Key=key)
                details['s3_config'] = json.loads(obj['Body'].read().decode('utf-8'))
            except:
                pass # Config might not exist for all jobs

        return details
    except Exception as e:
        raise HTTPException(500, f"Audit failed: {str(e)}")

@router.get("/system/config")
async def get_system_config(admin: dict = Depends(verify_admin)):
    """Get system configuration"""
    
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
    
    config["updated_by"] = admin["id"]
    config["updated_at"] = datetime.utcnow().isoformat()
    
    response = supabase.table("job_queue_control") \
        .insert(config) \
        .execute()
    
    # Log admin action
    get_service_client().table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "update_config",
        "target_type": "system",
        "details": config,
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    return {"status": "success", "data": response.data}

# --- New Settings Endpoints ---
from utils.config_manager import load_config, update_section

@router.get("/settings")
async def get_admin_settings(admin: dict = Depends(verify_admin)):
    """Get global admin settings (Phases & Pricing)"""
    return load_config()

@router.post("/settings/{section}")
async def update_admin_settings(
    section: str,
    settings: Dict[str, Any],
    request: Request,
    admin: dict = Depends(verify_admin)
):
    """Update specific settings section (phases or pricing)"""
    if section not in ["phases", "pricing"]:
        raise HTTPException(status_code=400, detail="Invalid settings section")
        
    updated_config = update_section(section, settings)
    
    # Log admin action
    get_service_client().table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": f"update_{section}",
        "target_type": "settings",
        "details": settings,
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    return {"status": "success", "data": updated_config}

from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    password: str
    credits: int = 10
    is_admin: bool = False
    plan: str = "free"

class UserUpdate(BaseModel):
    credits: Optional[int] = None
    is_admin: Optional[bool] = None
    role: Optional[str] = None
    plan: Optional[str] = None

@router.post("/users")
async def create_user(
    user_data: UserCreate,
    request: Request,
    admin: dict = Depends(verify_admin)
):
    """Create a new user"""
    try:
        # Use Service Client for Admin Auth operations
        service_client = get_service_client()
        
        # Create user in Supabase Auth
        try:
            auth_response = service_client.auth.admin.create_user({
                "email": user_data.email,
                "password": user_data.password,
                "email_confirm": True
            })
            new_user = auth_response.user
        except Exception as auth_error:
            # Better error message for missing service key scenario
            if "service_role" in str(auth_error) or "401" in str(auth_error):
                raise HTTPException(status_code=500, detail="Admin configuration error: Missing Service Key")
            raise auth_error
        
        # Update profile with extra fields
        if new_user:
            service_client.table("profiles").update({
                "credits": user_data.credits,
                "is_admin": user_data.is_admin,
                "plan": user_data.plan
            }).eq("id", new_user.id).execute()
        
        # Log action (can use standard client)
        get_service_client().table("admin_actions").insert({
            "admin_id": admin["id"],
            "action_type": "create_user",
            "target_id": new_user.id,
            "target_type": "user",
            "details": {"email": user_data.email},
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent")
        }).execute()
        
        return {"status": "success", "user": new_user}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    request: Request,
    admin: dict = Depends(verify_admin)
):
    """Update user credits and roles"""
    
    update_dict = {}
    if user_data.credits is not None:
        update_dict["credits"] = user_data.credits
    if user_data.is_admin is not None:
        update_dict["is_admin"] = user_data.is_admin
    if user_data.role is not None:
        update_dict["role"] = user_data.role
    if user_data.plan is not None:
        update_dict["plan"] = user_data.plan
        
    if not update_dict:
        return {"status": "ignored", "message": "No fields to update"}
        
    # Use service client to update profile roles securely
    service_client = get_service_client()
    response = service_client.table("profiles").update(update_dict).eq("id", user_id).execute()
    
    # Log action
    get_service_client().table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "update_user",
        "target_id": user_id,
        "target_type": "user",
        "details": update_dict,
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }).execute()
    
    return {"status": "success", "data": response.data}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    admin: dict = Depends(verify_admin)
):
    """Delete a user"""
    
    try:
        # Use Service Client
        service_client = get_service_client()
        
        # Delete from Auth (Users table)
        # This cascades to profiles usually
        service_client.auth.admin.delete_user(user_id)
        
        # Log action
        get_service_client().table("admin_actions").insert({
            "admin_id": admin["id"],
            "action_type": "delete_user",
            "target_id": user_id,
            "target_type": "user",
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent")
        }).execute()
        
        return {"status": "success", "message": f"User {user_id} deleted"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fda/logs")
async def get_fda_logs(
    limit: int = 100,
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    admin: dict = Depends(verify_admin)
):
    """Get FDA Audit Logs (21 CFR Part 11)"""
    # Use Service Client to read all logs (Bypassing RLS since Admin is verified)
    svc_client = get_service_client()
    logs = await fda_service.get_audit_trail(svc_client, limit, user_id, resource_id)
    return logs

# --- RBAC Routes ---

@router.get("/rbac/roles")
async def get_all_roles(admin: dict = Depends(verify_admin)):
    """Get list of available roles"""
    return await rbac_service.get_all_roles(supabase)

@router.get("/rbac/users/{target_user_id}/roles")
async def get_user_roles(target_user_id: str, admin: dict = Depends(verify_admin)):
    """Get roles for a specific user"""
    return await rbac_service.get_user_roles(supabase, target_user_id)

@router.post("/rbac/assign")
async def assign_role(
    role_code: str = Body(...),
    user_id: str = Body(...),
    admin: dict = Depends(verify_admin)
):
    """Assign a role to a user"""
    service_client = get_service_client()
    success = await rbac_service.assign_role(service_client, user_id, role_code, admin['id'])
    
    # Audit
    if success:
        get_service_client().table("admin_actions").insert({
            "admin_id": admin["id"],
            "action_type": "assign_role",
            "target_id": user_id,
            "details": {"role": role_code}
        }).execute()
        
    return {"success": success}

@router.post("/rbac/remove")
async def remove_role(
    role_code: str = Body(...),
    user_id: str = Body(...),
    admin: dict = Depends(verify_admin)
):
    """Remove a role from a user"""
    service_client = get_service_client()
    success = await rbac_service.remove_role(service_client, user_id, role_code)
         
    # Audit
    if success:
        get_service_client().table("admin_actions").insert({
            "admin_id": admin["id"],
            "action_type": "remove_role",
            "target_id": user_id,
            "details": {"role": role_code}
        }).execute()

    return {"success": success}

# --- Sentinel Auto-Healer ---

@router.post("/sentinel/scan")
async def trigger_sentinel(
    background_tasks: BackgroundTasks,
    admin: dict = Depends(verify_admin)
):
    """
    Trigger the Sentinel Self-Healing System manually.
    Scans for Stuck Processing, Spot Failures, and Zombie jobs.
    """
    from services.sentinel import BioDockifySentinel
    
    # Use service client for omniscient access
    svc_client = get_service_client()
    sentinel = BioDockifySentinel(svc_client)
    
    # Run scan
    report = await sentinel.scan_and_heal()
    
    # Log the scan
    # Log the scan
    get_service_client().table("admin_actions").insert({
        "admin_id": admin["id"],
        "action_type": "sentinel_manual_trigger",
        "details": report
    }).execute()
    
    return {"status": "success", "report": report}
