from sqlalchemy import Column, String, Float, Integer, ForeignKey, JSON, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from .database import Base

class Project(Base):
    __tablename__ = "projects_ai"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True) # Linked to Supabase Auth ID
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    compounds = relationship("Compound", back_populates="project", cascade="all, delete-orphan")
    models = relationship("QSARModel", back_populates="project", cascade="all, delete-orphan")

class Compound(Base):
    __tablename__ = "compounds_ai"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects_ai.id"), nullable=False)
    smiles = Column(String, nullable=False)
    chem_name = Column(String, nullable=True)
    source = Column(String, default="upload")
    properties = Column(JSON, default={}) # Stores calculated props
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="compounds")

class QSARModel(Base):
    __tablename__ = "qsar_models_ai"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects_ai.id"), nullable=False)
    name = Column(String, nullable=False)
    model_type = Column(String, default="regression")
    target_column = Column(String)
    metrics = Column(JSON, default={}) # {r2, rmse, mae}
    model_path = Column(String, nullable=False) # HF Space Path
    status = Column(String, default="ready")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")

class Prediction(Base):
    __tablename__ = "predictions_ai"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("qsar_models_ai.id"), nullable=False)
    smiles = Column(String, nullable=False)
    result = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("QSARModel", back_populates="predictions")
