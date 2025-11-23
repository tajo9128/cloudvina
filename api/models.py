"""
SQLAlchemy models for CloudVina
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
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")

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
