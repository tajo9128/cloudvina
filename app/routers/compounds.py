from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.auth import get_current_user
from app.database import get_db
from app.models import Compound, Project
import uuid

router = APIRouter()

# Schema
class CompoundCreate(BaseModel):
    project_id: str
    smiles: str
    chem_name: Optional[str] = None
    properties: Dict[str, Any] = {}
    source: str = "upload"

class CompoundResponse(BaseModel):
    id: str
    project_id: str
    smiles: str
    chem_name: Optional[str]
    properties: Dict[str, Any]
    created_at: str

    class Config:
        from_attributes = True

class CompoundBatchCreate(BaseModel):
    project_id: str
    compounds: List[CompoundCreate]

@router.post("/", response_model=List[CompoundResponse])
def upload_compounds(
    payload: CompoundBatchCreate,
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Batch upload compounds to a project (compounds_ai table).
    """
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == uuid.UUID(payload.project_id),
        Project.user_id == uuid.UUID(user['id'])
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    new_compounds = []
    for c in payload.compounds:
        # Ensure the payload project_id matches the one validated (redundant check but safe)
        if c.project_id != payload.project_id:
             continue 

        new_compounds.append(Compound(
            project_id=uuid.UUID(payload.project_id),
            smiles=c.smiles,
            chem_name=c.chem_name,
            properties=c.properties,
            source=c.source or "upload"
        ))
    
    if not new_compounds:
        return []

    try:
        db.add_all(new_compounds)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    
    # Refresh to get IDs
    for c in new_compounds:
        db.refresh(c)

    return [
        {
            "id": str(c.id),
            "project_id": str(c.project_id),
            "smiles": c.smiles,
            "chem_name": c.chem_name,
            "properties": c.properties,
            "created_at": str(c.created_at)
        }
        for c in new_compounds
    ]

@router.get("/{project_id}", response_model=List[CompoundResponse])
def list_compounds(
    project_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List compounds for a specific project.
    """
    # Verify project access
    project = db.query(Project).filter(
        Project.id == uuid.UUID(project_id),
        Project.user_id == uuid.UUID(user['id'])
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    compounds = db.query(Compound).filter(Compound.project_id == uuid.UUID(project_id)).all()
    
    return [
        {
            "id": str(c.id),
            "project_id": str(c.project_id),
            "smiles": c.smiles,
            "chem_name": c.chem_name,
            "properties": c.properties,
            "created_at": str(c.created_at)
        }
        for c in compounds
    ]
