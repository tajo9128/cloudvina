from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from pydantic import BaseModel
from app.auth import get_current_user

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

@router.post("/", response_model=ProjectResponse)
def create_project(
    project: ProjectCreate, 
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new project (Placeholder - Needs DB interaction).
    """
    # TODO: Insert into Supabase 'projects' table
    return {
        "id": "proj_123_mock", 
        "name": project.name, 
        "description": project.description,
        "user_id": user['id']
    }

@router.get("/", response_model=List[ProjectResponse])
def list_projects(user: Dict[str, Any] = Depends(get_current_user)):
    """
    List user projects (Placeholder).
    """
    # TODO: Select from Supabase 'projects' table
    return []
