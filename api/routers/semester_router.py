"""
Semester Router 

# Endpoints:

POST   /semesters/                    # Create semester
GET    /semesters/                    # List semesters
GET    /semesters/{semester_id}/stats # Get statistics

"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

from api.auth import get_current_user, UserInDB, PermissionChecker
from api.dependencies import SemesterServiceDep
from api.models import ErrorResponse

router = APIRouter()

# --- Pydantic Models ---
class SemesterCreateRequest(BaseModel):
    name: str
    start_date: datetime
    end_date: datetime
    mentor_quota: int
    mentee_quota: int

class SemesterResponse(BaseModel):
    semester_id: str
    name: str
    status: str
    created_at: str

# --- Endpoints ---
@router.post(
    "/",
    response_model=SemesterResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(PermissionChecker("can_create_semester"))]
)
async def create_semester(
    request: SemesterCreateRequest,
    semester_service: SemesterServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """Create new semester"""
    try:
        semester_id = semester_service.create_semester(
            organization_id=user.organization_id,
            name=request.name,
            start_date=request.start_date,
            end_date=request.end_date,
            mentor_quota=request.mentor_quota,
            mentee_quota=request.mentee_quota,
            created_by=user.email
        )
        
        semester = semester_service.get_semester(semester_id, user.organization_id)
        
        return SemesterResponse(
            semester_id=semester["_id"],
            name=semester["name"],
            status=semester["status"],
            created_at=semester["created_at"].isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/")
async def list_semesters(
    semester_service: SemesterServiceDep,
    user: UserInDB = Depends(get_current_user),
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50
):
    """List semesters for user's organization"""
    semesters = semester_service.list_semesters(
        organization_id=user.organization_id,
        status=status,
        skip=skip,
        limit=limit
    )
    
    return {"semesters": semesters, "count": len(semesters)}

@router.get("/{semester_id}/stats")
async def get_semester_stats(
    semester_id: str,
    semester_service: SemesterServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """Get semester statistics"""
    try:
        stats = semester_service.get_semester_stats(semester_id, user.organization_id)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))