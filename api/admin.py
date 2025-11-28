"""
Admin Panel API Endpoints
Provides Django-like admin functionality for managing users, pricing, and system settings.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from auth import supabase, get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])

# ============================================================================
# Pydantic Models
# ============================================================================

class UserProfile(BaseModel):
    id: str
    email: str
    role: str
    is_verified: bool
    created_at: str

class UpdateUserRequest(BaseModel):
    is_verified: Optional[bool] = None
    role: Optional[str] = None

class UpdateCreditsRequest(BaseModel):
    credits: int

class PricingPlan(BaseModel):
    id: Optional[str] = None
    name: str
    price: float
    credits: int
    features: List[str]
    is_active: bool = True

# ============================================================================
# Admin Middleware
# ============================================================================

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    """Verify that the current user has admin role"""
    # Check if user has admin role in profiles table
    response = supabase.table('profiles').select('role').eq('id', current_user.id).execute()
    
    if not response.data or response.data[0]['role'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user

# ============================================================================
# User Management Endpoints
# ============================================================================

@router.get("/users", response_model=List[UserProfile])
async def list_users(admin_user: dict = Depends(get_admin_user)):
    """Get all users with their profiles"""
    try:
        response = supabase.table('profiles').select('*').execute()
        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users: {str(e)}"
        )

@router.post("/users/{user_id}/verify")
async def verify_user(
    user_id: str,
    verified: bool,
    admin_user: dict = Depends(get_admin_user)
):
    """Verify or unverify a user"""
    try:
        response = supabase.table('profiles').update({
            'is_verified': verified
        }).eq('id', user_id).execute()
        
        return {
            "message": f"User {'verified' if verified else 'unverified'} successfully",
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )

@router.post("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str,
    admin_user: dict = Depends(get_admin_user)
):
    """Update user role (user/admin)"""
    if role not in ['user', 'admin']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'user' or 'admin'"
        )
    
    try:
        response = supabase.table('profiles').update({
            'role': role
        }).eq('id', user_id).execute()
        
        return {
            "message": f"User role updated to {role}",
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update role: {str(e)}"
        )

@router.post("/users/{user_id}/credits")
async def update_user_credits(
    user_id: str,
    request: UpdateCreditsRequest,
    admin_user: dict = Depends(get_admin_user)
):
    """Add or remove credits for a user"""
    try:
        # Check if user_credits record exists
        existing = supabase.table('user_credits').select('*').eq('user_id', user_id).execute()
        
        if existing.data:
            # Update existing record
            response = supabase.table('user_credits').update({
                'credits': request.credits
            }).eq('user_id', user_id).execute()
        else:
            # Create new record
            response = supabase.table('user_credits').insert({
                'user_id': user_id,
                'credits': request.credits
            }).execute()
        
        return {
            "message": "Credits updated successfully",
            "user_id": user_id,
            "credits": request.credits
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update credits: {str(e)}"
        )

# ============================================================================
# Pricing Management Endpoints
# ============================================================================

@router.get("/pricing", response_model=List[PricingPlan])
async def list_pricing_plans(admin_user: dict = Depends(get_admin_user)):
    """Get all pricing plans"""
    try:
        response = supabase.table('pricing_plans').select('*').execute()
        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch pricing plans: {str(e)}"
        )

@router.post("/pricing")
async def create_pricing_plan(
    plan: PricingPlan,
    admin_user: dict = Depends(get_admin_user)
):
    """Create a new pricing plan"""
    try:
        response = supabase.table('pricing_plans').insert({
            'name': plan.name,
            'price': plan.price,
            'credits': plan.credits,
            'features': plan.features,
            'is_active': plan.is_active
        }).execute()
        
        return {
            "message": "Pricing plan created successfully",
            "plan": response.data[0]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pricing plan: {str(e)}"
        )

@router.put("/pricing/{plan_id}")
async def update_pricing_plan(
    plan_id: str,
    plan: PricingPlan,
    admin_user: dict = Depends(get_admin_user)
):
    """Update an existing pricing plan"""
    try:
        response = supabase.table('pricing_plans').update({
            'name': plan.name,
            'price': plan.price,
            'credits': plan.credits,
            'features': plan.features,
            'is_active': plan.is_active
        }).eq('id', plan_id).execute()
        
        return {
            "message": "Pricing plan updated successfully",
            "plan": response.data[0]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pricing plan: {str(e)}"
        )

@router.delete("/pricing/{plan_id}")
async def delete_pricing_plan(
    plan_id: str,
    admin_user: dict = Depends(get_admin_user)
):
    """Delete a pricing plan"""
    try:
        response = supabase.table('pricing_plans').delete().eq('id', plan_id).execute()
        
        return {
            "message": "Pricing plan deleted successfully",
            "plan_id": plan_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete pricing plan: {str(e)}"
        )

# ============================================================================
# System Stats Endpoints
# ============================================================================

@router.get("/stats")
async def get_system_stats(admin_user: dict = Depends(get_admin_user)):
    """Get system statistics"""
    try:
        # Count total users
        users_response = supabase.table('profiles').select('id', count='exact').execute()
        total_users = users_response.count
        
        # Count total jobs
        jobs_response = supabase.table('jobs').select('id', count='exact').execute()
        total_jobs = jobs_response.count
        
        # Count jobs by status
        succeeded = supabase.table('jobs').select('id', count='exact').eq('status', 'SUCCEEDED').execute()
        failed = supabase.table('jobs').select('id', count='exact').eq('status', 'FAILED').execute()
        running = supabase.table('jobs').select('id', count='exact').in_('status', ['SUBMITTED', 'RUNNABLE', 'STARTING', 'RUNNING']).execute()
        
        return {
            "total_users": total_users,
            "total_jobs": total_jobs,
            "jobs_succeeded": succeeded.count,
            "jobs_failed": failed.count,
            "jobs_running": running.count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stats: {str(e)}"
        )

# ============================================================================
# Analytics Endpoints (NEW for Phase 6)
# ============================================================================

from services.analytics import AnalyticsService
from models import ActivityLog, async_session_maker
from sqlalchemy import select

@router.get("/analytics")
async def get_analytics(admin_user: dict = Depends(get_admin_user)):
    """Get comprehensive analytics dashboard data"""
    try:
        service = AnalyticsService()
        stats = await service.get_dashboard_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch analytics: {str(e)}"
        )

@router.get("/analytics/timeline")
async def get_job_timeline(
    days: int = 7,
    admin_user: dict = Depends(get_admin_user)
):
    """Get job submission timeline for charts"""
    try:
        service = AnalyticsService()
        timeline = await service.get_job_timeline(days)
        return timeline
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch timeline: {str(e)}"
        )

@router.get("/activity-logs")
async def get_activity_logs(
    skip: int = 0,
    limit: int = 100,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    admin_user: dict = Depends(get_admin_user)
):
    """Get recent activity logs with optional filters"""
    try:
        async with async_session_maker() as session:
            query = select(ActivityLog).order_by(ActivityLog.created_at.desc())
            
            if action:
                query = query.where(ActivityLog.action.contains(action))
            
            if resource_type:
                query = query.where(ActivityLog.resource_type == resource_type)
            
            query = query.offset(skip).limit(limit)
            
            result = await session.execute(query)
            logs = result.scalars().all()
            
            return [
                {
                    "id": str(log.id),
                    "user_id": str(log.user_id) if log.user_id else None,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "details": log.details,
                    "ip_address": str(log.ip_address) if log.ip_address else None,
                    "created_at": log.created_at.isoformat()
                }
                for log in logs
            ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch activity logs: {str(e)}"
        )


@router.post("/fix-s3-cors")
async def fix_s3_cors(admin_user: dict = Depends(get_admin_user)):
    """Configure S3 CORS to allow uploads from any origin"""
    import boto3
    import os
    
    try:
        AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
        S3_BUCKET = os.getenv("S3_BUCKET", "BioDockify-jobs-use1-1763775915")
        
        s3 = boto3.client('s3', region_name=AWS_REGION)
        
        cors_configuration = {
            'CORSRules': [{
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'PUT', 'POST', 'HEAD'],
                'AllowedOrigins': ['*'],
                'ExposeHeaders': ['ETag'],
                'MaxAgeSeconds': 3000
            }]
        }
        
        s3.put_bucket_cors(Bucket=S3_BUCKET, CORSConfiguration=cors_configuration)
        
        return {"message": f"Successfully configured CORS for bucket: {S3_BUCKET}"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure CORS: {str(e)}"
        )
