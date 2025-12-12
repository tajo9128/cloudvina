from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.auth import get_current_user
from app.database import get_db
from app.models import Project
import uuid

router = APIRouter()

# Schema
class ProjectCreate(BaseModel):
    name: str
    description: str = None

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str = None
    user_id: str
    created_at: str

    class Config:
        from_attributes = True

@router.post("/", response_model=ProjectResponse)
def create_project(
    project: ProjectCreate, 
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new project in projects_ai table.
    """
    new_project = Project(
        user_id=user['id'],
        name=project.name,
        description=project.description
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    
    # Convert UUIDs to strings for Pydantic
    return {
        "id": str(new_project.id),
        "name": new_project.name,
        "description": new_project.description,
        "user_id": str(new_project.user_id),
        "created_at": str(new_project.created_at)
    }

@router.get("/", response_model=List[ProjectResponse])
def list_projects(
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List user projects from projects_ai table.
    """
    projects = db.query(Project).filter(Project.user_id == uuid.UUID(user['id'])).all()
    
    return [
        {
            "id": str(p.id),
            "name": p.name,
            "description": p.description,
            "user_id": str(p.user_id),
            "created_at": str(p.created_at)
        }
        for p in projects
    ]

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == uuid.UUID(project_id),
        Project.user_id == uuid.UUID(user['id'])
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "id": str(project.id),
        "name": project.name,
        "description": project.description,
        "user_id": str(project.user_id),
        "created_at": str(project.created_at)
    }

@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    project = db.query(Project).filter(
        Project.id == uuid.UUID(project_id),
        Project.user_id == uuid.UUID(user['id'])
    ).first()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(project)
    db.commit()
    return {"status": "deleted", "id": project_id}
