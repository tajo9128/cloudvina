from supabase import Client
from typing import Optional, List

class RBACService:
    """
    Service for Role-Based Access Control.
    Manages granular permissions beyond 'is_admin'.
    """

    @staticmethod
    async def get_user_roles(supabase_client: Client, user_id: str) -> List[dict]:
        """Get all roles assigned to a user."""
        try:
            res = supabase_client.table("rbac_user_roles") \
                .select("role_id, rbac_roles(code, name)") \
                .eq("user_id", user_id) \
                .execute()
            
            roles = [item['rbac_roles'] for item in res.data if item.get('rbac_roles')]
            return roles
        except Exception as e:
            print(f"[RBAC ERROR] Get Roles: {e}")
            return []

    @staticmethod
    async def get_all_roles(supabase_client: Client) -> List[dict]:
        """Get list of all available roles."""
        res = supabase_client.table("rbac_roles").select("*").execute()
        return res.data

    @staticmethod
    async def assign_role(
        supabase_client: Client, 
        user_id: str, 
        role_code: str, 
        assigned_by: str
    ) -> bool:
        """Assign a role to a user."""
        try:
            # 1. Get Role ID
            role_res = supabase_client.table("rbac_roles").select("id").eq("code", role_code).single().execute()
            if not role_res.data:
                return False
            role_id = role_res.data['id']
            
            # 2. Assign
            supabase_client.table("rbac_user_roles").insert({
                "user_id": user_id,
                "role_id": role_id,
                "assigned_by": assigned_by
            }).execute()
            
            return True
        except Exception as e:
            # Likely duplicate key error (already has role)
            print(f"[RBAC INFO] Assign Role: {e}")
            return False

    @staticmethod
    async def remove_role(supabase_client: Client, user_id: str, role_code: str) -> bool:
        """Remove a role from a user."""
        try:
            # 1. Get Role ID
            role_res = supabase_client.table("rbac_roles").select("id").eq("code", role_code).single().execute()
            if not role_res.data:
                return False
            role_id = role_res.data['id']
            
            # 2. Delete
            supabase_client.table("rbac_user_roles") \
                .delete() \
                .eq("user_id", user_id) \
                .eq("role_id", role_id) \
                .execute()
            
            return True
        except Exception as e:
            print(f"[RBAC ERROR] Remove Role: {e}")
            return False

    @staticmethod
    async def has_permission(supabase_client: Client, user_id: str, permission_code: str) -> bool:
        """Check if user has a specific permission via any role."""
        try:
            # Query Logic:
            # user -> rbac_user_roles -> rbac_role_permissions -> rbac_permissions(code)
            # Complex joins are hard in PostgREST, better to fetch roles and check permissions locally or use RPC.
            # Simplified: Check if they have a role that maps to the permission.
            
            # TODO: Implementation requires a view or RPC for efficiency.
            # For now, we trust the Client to have verified 'is_admin' or we check explicit roles.
            return False 
        except Exception:
            return False

rbac_service = RBACService()
