from fastapi import HTTPException, status
from supabase import Client
from datetime import date, datetime

class RateLimiter:
    """
    Rate limiting for free tier users
    - First 30 days: 3 jobs/day
    - After 30 days: 1 job/day
    """
    
    @staticmethod
    def get_daily_limit(account_created_at: datetime, plan: str) -> int:
        """
        Get daily job limit based on account age and plan
        """
        if plan and plan != 'free':
            return 99999  # Paid users: no limit
        
        # Calculate account age in days
        age_days = (datetime.now() - account_created_at).days
        
        # First 30 days: 3 jobs/day
        if age_days < 30:
            return 3
        else:
            # After 30 days: 1 job/day
            return 1
    
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
            
            # Check phone verification - DISABLED for now as per user request
            # profile_response = supabase.table('user_profiles').select('phone_verified').eq('id', user_id).single().execute()
            # if not profile_response.data or not profile_response.data.get('phone_verified'):
            #     return {
            #         "allowed": False,
            #         "message": "Please verify your phone number before submitting jobs. Contact support for phone verification.",
            #         "reason": "phone_not_verified"
            #     }
            
            # Get user credits and account info
            credits_response = supabase.table('user_credits').select('plan, credits, account_created_at').eq('user_id', user_id).single().execute()
            
            if not credits_response.data:
                return {
                    "allowed": False,
                    "message": "User account not properly initialized. Please contact support.",
                    "reason": "no_credits_record"
                }
            
            user_plan = credits_response.data.get('plan', 'free')
            user_credits = credits_response.data.get('credits', 0)
            account_created_str = credits_response.data.get('account_created_at')
            
            # Parse account creation date
            account_created_at = datetime.fromisoformat(account_created_str.replace('Z', '+00:00')) if account_created_str else datetime.now()
            
            # Check if user has credits
            if user_credits <= 0:
                return {
                    "allowed": False,
                    "message": "You have run out of credits. Please upgrade your plan to continue.",
                    "reason": "no_credits"
                }
            
            # Get dynamic daily limit
            daily_limit = RateLimiter.get_daily_limit(account_created_at, user_plan)
            
            # Paid users: only check credits, no daily limit
            if user_plan and user_plan != 'free':
                return {
                    "allowed": True,
                    "message": "Job submission allowed",
                    "credits_remaining": user_credits,
                    "daily_limit": None
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
            
            if jobs_today >= daily_limit:
                account_age = (datetime.now() - account_created_at).days
                message = f"Daily limit reached. You can submit {daily_limit} job{'s' if daily_limit > 1 else ''} per day "
                if account_age < 30:
                    message += f"(first month bonus). Try again in 24 hours or upgrade for unlimited access."
                else:
                    message += f"(1 per day after first month). Upgrade for unlimited access."
                
                return {
                    "allowed": False,
                    "message": message,
                    "reason": "daily_limit_reached",
                    "jobs_today": jobs_today,
                    "limit": daily_limit
                }
            
            return {
                "allowed": True,
                "message": "Job submission allowed",
                "jobs_today": jobs_today,
                "limit": daily_limit,
                "remaining_today": daily_limit - jobs_today,
                "credits_remaining": user_credits
            }
            
        except Exception as e:
            # In case of error, deny access for safety
            print(f"CRITICAL ERROR in RateLimiter: {str(e)}")  # Added for debugging
            import traceback
            traceback.print_exc()  # Print full stack trace
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
