from supabase import Client
import datetime
from typing import Optional, Dict, Any

class FDAService:
    """
    Service for handling FDA 21 CFR Part 11 Compliance.
    Focus: Immutable Audit Trails.
    """

    @staticmethod
    async def log_audit_event(
        supabase_client: Client,
        user_id: str,
        action: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Logs a critical system event to the immutable fda_audit_logs table.
        """
        try:
            payload = {
                "user_id": user_id,
                "action": action,
                "resource_id": resource_id,
                "details": details or {},
                "ip_address": ip_address,
                "user_agent": user_agent,
                # created_at is handled by DB default
            }

            # Execute Insert
            supabase_client.table("fda_audit_logs").insert(payload).execute()
            return True
        except Exception as e:
            # Critical Log Failure - In a strict environment, this might halt execution.
            print(f"[FDA CRITICAL] Audit Log Failed: {e}")
            return False

    @staticmethod
    async def get_audit_trail(
        supabase_client: Client,
        limit: int = 100,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> list:
        """
        Retrieves the audit trail.
        Requires a client with Admin/Select permissions (RLS enforced).
        """
        try:
            query = supabase_client.table("fda_audit_logs").select("*").order("created_at", desc=True).limit(limit)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            if resource_id:
                query = query.eq("resource_id", resource_id)
                
            response = query.execute()
            return response.data
        except Exception as e:
            print(f"[FDA ERROR] Fetch Audit Trail Failed: {e}")
            return []

fda_service = FDAService()
