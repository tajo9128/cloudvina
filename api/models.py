"""
SQLAlchemy models for BioDockify
Maps to existing Supabase database schema
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, ARRAY
from sqlalchemy.dialects.postgresql import JSONB, INET, UUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os
import uuid

Base = declarative_base()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    # Replace postgres:// or postgresql:// with postgresql+asyncpg:// for async support
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # Replace pooler port 6543 with direct connection port 5432 for asyncpg
    if ":6543/" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace(":6543/", ":5432/")
    
    print(f"Using DATABASE_URL: {DATABASE_URL[:50]}...")  # Debug: print first 50 chars

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Models
class User(Base):
    """User model from auth.users schema"""
    __tablename__ = 'users'
    __table_args__ = {'schema': 'auth'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    role = Column(String, default='user')
    is_verified = Column(Boolean, default=False)
    credits = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User {self.email}>"

class Job(Base):
    """Docking job model"""
    __tablename__ = 'jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('auth.users.id'))
    job_id = Column(String, unique=True, nullable=False)
    status = Column(String, default='SUBMITTED')
    binding_affinity = Column(Float, nullable=True)
    receptor_s3_key = Column(String)
    ligand_s3_key = Column(String)
    output_s3_key = Column(String, nullable=True)
    log_s3_key = Column(String, nullable=True)
    aws_job_id = Column(String, nullable=True)
    parameters = Column(JSONB, default={})  # NEW: Advanced docking parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<Job {self.job_id} - {self.status}>"

class PricingPlan(Base):
    """Pricing plan model"""
    __tablename__ = 'pricing_plans'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    credits = Column(Integer, nullable=False)
    features = Column(ARRAY(String))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PricingPlan {self.name} - ${self.price}>"

class ActivityLog(Base):
    """Activity log / audit trail model - NEW"""
    __tablename__ = 'activity_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('auth.users.id'), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    details = Column(JSONB)
    ip_address = Column(INET)
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<ActivityLog {self.action} at {self.created_at}>"

class FDAAuditLog(Base):
    """FDA 21 CFR Part 11 Audit Trail Table"""
    __tablename__ = 'fda_audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('auth.users.id'), nullable=True)
    action = Column(String, nullable=False)
    resource_id = Column(String)
    details = Column(JSONB, default={})
    ip_address = Column(String) # Changed from INET to String for compatibility if needed, or keep INET
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    signature = Column(String, nullable=True)
    
    user = relationship("User", foreign_keys=[user_id])

class RBACRole(Base):
    """RBAC Roles"""
    __tablename__ = 'rbac_roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class RBACUserRole(Base):
    """User-Role Mapping"""
    __tablename__ = 'rbac_user_roles'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('auth.users.id'), primary_key=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey('rbac_roles.id'), primary_key=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    assigned_by = Column(UUID(as_uuid=True))
    
    user = relationship("User", backref="rbac_roles")
    role = relationship("RBACRole")
