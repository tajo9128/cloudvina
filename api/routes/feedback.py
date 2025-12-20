from fastapi import APIRouter, Depends, HTTPException, Body, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth import get_current_user, get_authenticated_client
from services.feedback_service import feedback_service
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/feedback", tags=["Feedback"])
security = HTTPBearer()

class FeedbackRequest(BaseModel):
    job_id: str
    rating: int # 1 or -1
    comment: Optional[str] = None

@router.post("/", status_code=201)
async def submit_feedback(
    payload: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Submit feedback (Like/Dislike) for a job.
    Used for Active Learning optimization.
    """
    auth_client = get_authenticated_client(credentials.credentials)
    
    try:
        result = feedback_service.log_feedback(
            auth_client,
            user_id=current_user['id'],
            job_id=payload.job_id,
            rating=payload.rating,
            comment=payload.comment
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/stats")
async def get_feedback_stats(
    current_user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get user's feedback statistics.
    """
    auth_client = get_authenticated_client(credentials.credentials)
    return feedback_service.get_user_stats(auth_client, current_user['id'])
