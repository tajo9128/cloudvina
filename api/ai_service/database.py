from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

# --- Configuration ---
# Zero-Cost: SQLite file. Easy to upgrade to Postgres later (Render/Railway/Supabase)
SQLALCHEMY_DATABASE_URL = "sqlite:///./compounds.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- The User's Exact Schema ---

class Compound(Base):
    """
    Table: Compounds
    - compound_id
    - smiles
    - compound_name
    - source
    - date_added
    """
    __tablename__ = "compounds"

    compound_id = Column(Integer, primary_key=True, index=True)
    smiles = Column(String, index=True, unique=True) # Enforce uniqueness on structure
    compound_name = Column(String, default="Unknown")
    source = Column(String, default="Manual Upload")
    date_added = Column(DateTime, default=datetime.utcnow)

    # Relationships
    experimental_results = relationship("ExperimentalResult", back_populates="compound")
    computational_predictions = relationship("ComputationalPrediction", back_populates="compound")


class ExperimentalResult(Base):
    """
    Table: Experimental_Results
    - compound_id
    - assay_type (IC50, toxicity, etc.)
    - value
    - unit
    - date_tested
    - notes
    """
    __tablename__ = "experimental_results"

    id = Column(Integer, primary_key=True, index=True)
    compound_id = Column(Integer, ForeignKey("compounds.compound_id"))
    assay_type = Column(String)
    value = Column(Float)
    unit = Column(String)
    date_tested = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    compound = relationship("Compound", back_populates="experimental_results")


class ComputationalPrediction(Base):
    """
    Table: Computational_Predictions
    - docking_score
    - md_stability_rmsd
    - bbp_prediction
    - toxicity_score
    """
    __tablename__ = "computational_predictions"

    id = Column(Integer, primary_key=True, index=True)
    compound_id = Column(Integer, ForeignKey("compounds.compound_id"))
    
    docking_score = Column(Float, nullable=True)
    md_stability_rmsd = Column(Float, nullable=True)
    mmpbsa_delta_g = Column(Float, nullable=True)
    bbp_prediction = Column(Float, nullable=True)
    toxicity_score = Column(Float, nullable=True)
    date_predicted = Column(DateTime, default=datetime.utcnow)

    compound = relationship("Compound", back_populates="computational_predictions")


class ModelPerformance(Base):
    """
    Table: Model_Performance
    - model_type (QSAR, GNN)
    - accuracy, precision, recall
    """
    __tablename__ = "model_performance"

    model_id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    training_date = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    r_squared = Column(Float, nullable=True)


# --- Init DB ---
def init_db():
    Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
