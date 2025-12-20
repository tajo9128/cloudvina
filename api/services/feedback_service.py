import logging
from typing import Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

class FeedbackService:
    """
    Service to handle user feedback on job results.
    Primary component of the 'Hybrid Engine' Phase 4.
    """
    
    def log_feedback(self, client, user_id: str, job_id: str, rating: int, comment: Optional[str] = None) -> Dict:
        """
        Log user feedback (Like/Dislike) for a specific job.
        
        Args:
            client: Supabase client (authenticated)
            user_id: ID of the user submitting feedback
            job_id: ID of the job being rated
            rating: 1 (Like) or -1 (Dislike)
            comment: Optional text comment
            
        Returns:
            Created feedback record
        """
        if rating not in [1, -1]:
            raise ValueError("Rating must be 1 (Like) or -1 (Dislike)")
            
        # Check if feedback already exists (Upsert)
        data = {
            "user_id": user_id,
            "job_id": job_id,
            "rating": rating,
            "comment": comment
        }
        
        try:
            # We use upsert to allow changing mind
            res = client.table("job_feedback").upsert(data, on_conflict="job_id, user_id").execute()
            return res.data[0] if res.data else {}
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")
            raise e

    def get_job_feedback(self, client, job_id: str, user_id: str) -> Optional[Dict]:
        """
        Get existing feedback for a job by a specific user.
        """
        try:
            res = client.table("job_feedback").select("*").eq("job_id", job_id).eq("user_id", user_id).single().execute()
            return res.data
        except Exception:
            return None # No feedback found

    def get_user_stats(self, client, user_id: str) -> Dict:
        """
        Get feedback statistics for a user (how many jobs reviewed).
        """
        try:
            res_likes = client.table("job_feedback").select("id", count="exact").eq("user_id", user_id).eq("rating", 1).execute()
            res_dislikes = client.table("job_feedback").select("id", count="exact").eq("user_id", user_id).eq("rating", -1).execute()
            
            return {
                "likes": res_likes.count,
                "dislikes": res_dislikes.count,
                "total_reviewed": res_likes.count + res_dislikes.count
            }
        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return {"likes": 0, "dislikes": 0, "total_reviewed": 0}

# Singleton instance
feedback_service = FeedbackService()
