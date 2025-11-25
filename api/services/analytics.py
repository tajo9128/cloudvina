"""
Analytics Service for CloudVina
Provides dashboard statistics and insights
"""
from sqlalchemy import func, select, and_, or_
from models import Job, User, ActivityLog, async_session_maker
from datetime import datetime, timedelta
from typing import Dict, List

class AnalyticsService:
    """Service for generating analytics and statistics"""
    
    async def get_dashboard_stats(self) -> Dict:
        """Get key metrics for admin dashboard"""
        async with async_session_maker() as session:
            today = datetime.utcnow().date()
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            # Total counts
            total_users = await session.scalar(select(func.count(User.id)))
            total_jobs = await session.scalar(select(func.count(Job.id)))
            
            # Jobs by status
            jobs_succeeded = await session.scalar(
                select(func.count(Job.id)).where(Job.status == 'SUCCEEDED')
            )
            jobs_failed = await session.scalar(
                select(func.count(Job.id)).where(Job.status == 'FAILED')
            )
            jobs_running = await session.scalar(
                select(func.count(Job.id)).where(
                    Job.status.in_(['RUNNING', 'STARTING', 'RUNNABLE', 'SUBMITTED'])
                )
            )
            
            # This week's activity
            jobs_this_week = await session.scalar(
                select(func.count(Job.id)).where(Job.created_at >= week_ago)
            )
            new_users_this_week = await session.scalar(
                select(func.count(User.id)).where(User.created_at >= week_ago)
            )
            
            # This month's activity
            jobs_this_month = await session.scalar(
                select(func.count(Job.id)).where(Job.created_at >= month_ago)
            )
            
            # Average affinity
            avg_affinity = await session.scalar(
                select(func.avg(Job.binding_affinity)).where(
                    Job.status == 'SUCCEEDED',
                    Job.binding_affinity.isnot(None)
                )
            )
            
            # Best (lowest) affinity
            best_affinity = await session.scalar(
                select(func.min(Job.binding_affinity)).where(
                    Job.status == 'SUCCEEDED',
                    Job.binding_affinity.isnot(None)
                )
            )
            
            # Calculate success rate
            success_rate = round(jobs_succeeded / total_jobs * 100, 1) if total_jobs > 0 else 0
            
            return {
                "total_users": total_users or 0,
                "total_jobs": total_jobs or 0,
                "jobs_succeeded": jobs_succeeded or 0,
                "jobs_failed": jobs_failed or 0,
                "jobs_running": jobs_running or 0,
                "success_rate": success_rate,
                "jobs_this_week": jobs_this_week or 0,
                "new_users_this_week": new_users_this_week or 0,
                "jobs_this_month": jobs_this_month or 0,
                "avg_binding_affinity": round(avg_affinity, 2) if avg_affinity else None,
                "best_binding_affinity": round(best_affinity, 2) if best_affinity else None
            }
    
    async def get_job_timeline(self, days: int = 7) -> List[Dict]:
        """Get jobs submitted per day for chart"""
        async with async_session_maker() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Query jobs grouped by date
            query = select(
                func.date(Job.created_at).label('date'),
                func.count(Job.id).label('count')
            ).where(
                Job.created_at >= cutoff
            ).group_by(
                func.date(Job.created_at)
            ).order_by(
                func.date(Job.created_at)
            )
            
            result = await session.execute(query)
            rows = result.all()
            
            return [
                {
                    "date": row.date.isoformat(),
                    "count": row.count
                }
                for row in rows
            ]
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a specific user"""
        async with async_session_maker() as session:
            total_jobs = await session.scalar(
                select(func.count(Job.id)).where(Job.user_id == user_id)
            )
            
            succeeded_jobs = await session.scalar(
                select(func.count(Job.id)).where(
                    Job.user_id == user_id,
                    Job.status == 'SUCCEEDED'
                )
            )
            
            avg_affinity = await session.scalar(
                select(func.avg(Job.binding_affinity)).where(
                    Job.user_id == user_id,
                    Job.status == 'SUCCEEDED',
                    Job.binding_affinity.isnot(None)
                )
            )
            
            return {
                "total_jobs": total_jobs or 0,
                "succeeded_jobs": succeeded_jobs or 0,
                "success_rate": round(succeeded_jobs / total_jobs * 100, 1) if total_jobs > 0 else 0,
                "avg_binding_affinity": round(avg_affinity, 2) if avg_affinity else None
            }
