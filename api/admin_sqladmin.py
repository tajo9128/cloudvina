"""
SQLAdmin configuration for CloudVina
Django-style auto-generated admin panel
"""
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import RedirectResponse
from api.models import User, Job, PricingPlan, ActivityLog, engine
import os

# Authentication Backend
class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        username, password = form.get("username"), form.get("password")
        
        # Simple auth - check against environment variables
        if username == os.getenv("ADMIN_USERNAME") and password == os.getenv("ADMIN_PASSWORD"):
            request.session.update({"authenticated": True})
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return request.session.get("authenticated", False)

# User Admin View
class UserAdmin(ModelView, model=User):
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-user"
    
    column_list = [User.email, User.role, User.is_verified, User.credits, User.created_at]
    column_searchable_list = [User.email]
    column_sortable_list = [User.email, User.created_at, User.credits]
    column_default_sort = [(User.created_at, True)]
    
    # Editable columns
    form_columns = [User.email, User.role, User.is_verified, User.credits]
    
    # Filters
    column_filters = [User.role, User.is_verified]
    
    # Labels
    column_labels = {
        User.email: "Email Address",
        User.role: "Role",
        User.is_verified: "Verified",
        User.credits: "Credits Remaining",
        User.created_at: "Joined Date"
    }

# Job Admin View
class JobAdmin(ModelView, model=Job):
    name = "Job"
    name_plural = "Docking Jobs"
    icon = "fa-solid fa-flask"
    
    column_list = [Job.job_id, Job.status, Job.binding_affinity, Job.created_at]
    column_searchable_list = [Job.job_id]
    column_sortable_list = [Job.created_at, Job.status, Job.binding_affinity]
    column_default_sort = [(Job.created_at, True)]
    
    # Read-only fields
    form_excluded_columns = [Job.id, Job.aws_job_id, Job.receptor_s3_key, Job.ligand_s3_key]
    
    # Filters
    column_filters = [Job.status]
    
    # Custom formatting
    column_formatters = {
        Job.binding_affinity: lambda m, a: f"{a:.2f} kcal/mol" if a else "N/A"
    }
    
    # Labels
    column_labels = {
        Job.job_id: "Job ID",
        Job.status: "Status",
        Job.binding_affinity: "Binding Affinity",
        Job.parameters: "Docking Parameters",
        Job.created_at: "Submitted"
    }

# Pricing Plan Admin View
class PricingPlanAdmin(ModelView, model=PricingPlan):
    name = "Pricing Plan"
    name_plural = "Pricing Plans"
    icon = "fa-solid fa-dollar-sign"
    
    column_list = [PricingPlan.name, PricingPlan.price, PricingPlan.credits, PricingPlan.is_active]
    column_searchable_list = [PricingPlan.name]
    column_sortable_list = [PricingPlan.price, PricingPlan.credits]
    
    # Editable
    form_columns = [PricingPlan.name, PricingPlan.price, PricingPlan.credits, PricingPlan.features, PricingPlan.is_active]
    
    # Filters
    column_filters = [PricingPlan.is_active]
    
    # Labels
    column_labels = {
        PricingPlan.name: "Plan Name",
        PricingPlan.price: "Price (USD)",
        PricingPlan.credits: "Credits",
        PricingPlan.features: "Features",
        PricingPlan.is_active: "Active"
    }

# Activity Log Admin View (Read-only)
class ActivityLogAdmin(ModelView, model=ActivityLog):
    name = "Activity Log"
    name_plural = "Activity Logs"
    icon = "fa-solid fa-list"
    
    # Read-only view
    can_create = False
    can_edit = False
    can_delete = False
    
    column_list = [ActivityLog.action, ActivityLog.resource_type, ActivityLog.user_id, ActivityLog.created_at]
    column_searchable_list = [ActivityLog.action, ActivityLog.resource_type]
    column_sortable_list = [ActivityLog.created_at, ActivityLog.action]
    column_default_sort = [(ActivityLog.created_at, True)]
    
    # Filters
    column_filters = [ActivityLog.action, ActivityLog.resource_type, ActivityLog.created_at]
    
    # Custom formatting
    column_formatters = {
        ActivityLog.user_id: lambda m, a: str(a)[:8] + "..." if a else "System"
    }
    
    # Labels
    column_labels = {
        ActivityLog.action: "Action",
        ActivityLog.resource_type: "Resource",
        ActivityLog.resource_id: "Resource ID",
        ActivityLog.user_id: "User",
        ActivityLog.ip_address: "IP Address",
        ActivityLog.created_at: "Timestamp"
    }

# Initialize Admin
def setup_admin(app):
    """Setup SQLAdmin with authentication"""
    authentication_backend = AdminAuth(secret_key=os.getenv("SECRET_KEY", "your-secret-key-change-in-production"))
    
    admin = Admin(
        app, 
        engine,
        title="CloudVina Admin",
        authentication_backend=authentication_backend,
        base_url="/sqladmin"
    )
    
    # Add views
    admin.add_view(UserAdmin)
    admin.add_view(JobAdmin)
    admin.add_view(PricingPlanAdmin)
    admin.add_view(ActivityLogAdmin)
    
    return admin
