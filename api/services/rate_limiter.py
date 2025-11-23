from fastapi import HTTPException, status
from supabase import Client
from datetime import date

class RateLimiter:
    """Rate limiting for free tier users"""
    
    FREE_TIER_DAILY_LIMIT = 3
    
    @staticmethod
    async def check_can_submit(supabase: Client, user_id: str) -> dict:
        """
        Check if user can submit a job
        Returns: dict with 'allowed' boolean and 'message' string
        """
        try:
            # Check email verification
            user_response = supabase.auth.admin.get_user_by_id(user_id)
            if not user_response.user.email_confirmed_at:
                return {
                    "allowed": False,
                    "message": "Please verify your email address before submitting jobs. Check your inbox for verification link.",
                    "reason": "email_not_verified"
                }
            
            # Check phone verification
            profile_response = supabase.table('user_profiles').select('phone_verified').eq('id', user_id).single().execute()
            if not profile_response.data or not profile_response.data.get('phone_verified'):
                return {
                    "allowed": False,
                    "message": "Please verify your phone number before submitting jobs. Contact support for phone verification.",
                    "reason": "phone_not_verified"
                }
            
            # Check user plan
            credits_response = supabase.table('user_credits').select('plan, credits').eq('user_id', user_id).single().execute()
            
            if not credits_response.data:
                return {
                    "allowed": False,
                    "message": "User account not properly initialized. Please contact support.",
                    "reason": "no_credits_record"
                }
            
            user_plan = credits_response.data.get('plan', 'free')
            user_credits = credits_response.data.get('credits', 0)
            
            # Check if user has credits
            if user_credits <= 0:
                return {
                    "allowed": False,
                    "message": "You have run out of credits. Please upgrade your plan to continue.",
                    "reason": "no_credits"
                }
            
            # Paid users: only check credits, no daily limit
            if user_plan and user_plan != 'free':
                return {
                    "allowed": True,
                    "message": "Job submission allowed",
                    "credits_remaining": user_credits
                }
            
            # Free users: check daily limit
            today = date.today().isoformat()
            usage_response = supabase.table('daily_job_usage')\
                .select('job_count')\
                .eq('user_id', user_id)\
                .eq('job_date', today)\
                .execute()
            
            jobs_today = 0
            if usage_response.data and len(usage_response.data) > 0:
                jobs_today = usage_response.data[0].get('job_count', 0)
            
            if jobs_today >= RateLimiter.FREE_TIER_DAILY_LIMIT:
                return {
                    "allowed": False,
                    "message": f"Daily limit reached. Free users can submit {RateLimiter.FREE_TIER_DAILY_LIMIT} jobs per day. Please try again in 24 hours or upgrade to a paid plan for unlimited daily jobs.",
                    "reason": "daily_limit_reached",
                    "jobs_today": jobs_today,
                    "limit": RateLimiter.FREE_TIER_DAILY_LIMIT
                }
            
            return {
                "allowed": True,
                "message": "Job submission allowed",
                "jobs_today": jobs_today,
                "limit": RateLimiter.FREE_TIER_DAILY_LIMIT,
                "remaining_today": RateLimiter.FREE_TIER_DAILY_LIMIT - jobs_today,
                "credits_remaining": user_credits
            }
            
        except Exception as e:
            # In case of error, deny access for safety
            return {
                "allowed": False,
                "message": f"Error checking submission eligibility: {str(e)}",
                "reason": "error"
            }
    
    @staticmethod
    async def increment_usage(supabase: Client, user_id: str):
        """Increment daily job usage for user"""
        try:
            today = date.today().isoformat()
            
            # Try to get existing record
            existing = supabase.table('daily_job_usage')\
                .select('job_count')\
                .eq('user_id', user_id)\
                .eq('job_date', today)\
                .execute()
            
            if existing.data and len(existing.data) > 0:
                # Update existing
                current_count = existing.data[0].get('job_count', 0)
                supabase.table('daily_job_usage')\
                    .update({'job_count': current_count + 1})\
                    .eq('user_id', user_id)\
                    .eq('job_date', today)\
                    .execute()
            else:
                # Insert new
                supabase.table('daily_job_usage').insert({
                    'user_id': user_id,
                    'job_date': today,
                    'job_count': 1
                }).execute()
                
        except Exception as e:
            # Log error but don't fail the job submission
            print(f"Error incrementing usage: {str(e)}")
