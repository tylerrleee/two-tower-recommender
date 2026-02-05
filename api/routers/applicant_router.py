"""
Docstring for api.routers.applicant_router

Applicant Router - Applicant management and CSV upload endpoints

Endpoints:
POST   /applicants/upload            # CSV upload
GET    /applicants/                  # List applicants
GET    /applicants/{applicant_id}    # Get applicant
PATCH  /applicants/{applicant_id}    # Update applicant
DELETE /applicants/{applicant_id}    # Delete applicant
GET    /applicants/validate-csv      # Validate CSV (dry run)
"""

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Query
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
from io import StringIO
import logging

from api.auth import get_current_user, UserInDB, PermissionChecker
from api.dependencies import ApplicantServiceDep, SemesterServiceDep
from api.models import ErrorResponse

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CSVUploadResponse(BaseModel):
    """Response model for CSV upload"""
    total_uploaded: int
    mentors: int
    mentees: int
    duplicates_skipped: int
    errors: Optional[List[str]] = None
    semester_id: str
    uploaded_by: str
    uploaded_at: str


class ApplicantResponse(BaseModel):
    """Response model for applicant details"""
    applicant_id: str
    ufl_email: str
    full_name: str
    role: str
    status: str
    submitted_at: str
    survey_responses: dict


class ApplicantUpdateRequest(BaseModel):
    """Request model for updating applicant"""
    major: Optional[str] = None
    year: Optional[str] = None
    bio: Optional[str] = None
    interests: Optional[str] = None
    goals: Optional[str] = None
    # Add other survey fields as needed


class ApplicantListResponse(BaseModel):
    """Response model for list of applicants"""
    applicants: List[ApplicantResponse]
    count: int
    semester_id: str
    filter_role: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/upload",
    response_model=CSVUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(PermissionChecker("can_upload_applicants"))],
    summary="Upload applicants via CSV",
    description="Bulk upload applicants for a semester via CSV file"
)
async def upload_applicants_csv(
    semester_id: str = Query(..., description="Target semester ID"),
    file: UploadFile = File(..., description="CSV file with applicant data"),
    applicant_service: ApplicantServiceDep = Depends(),
    semester_service: SemesterServiceDep = Depends(),
    user: UserInDB = Depends(get_current_user)
):
    """
    Upload applicants via CSV
    
    Required CSV columns:
    - role: 0 or "mentor", 1 or "mentee"
    - name: Full name
    - ufl_email: Email address
    - major: Major field of study
    - year: Academic year (Freshman, Sophomore, Junior, Senior)
    
    Optional columns:
    - bio: Personal bio
    - interests: Interests
    - goals: Goals
    - (any other survey fields)
    
    Validation checks:
    1. Required columns present
    2. Email format validity
    3. No duplicate emails within CSV
    4. No duplicate emails with existing semester applicants
    5. Organization quota enforcement
    6. Minimum 2:1 mentee:mentor ratio
    
    Returns upload summary with counts and any errors.
    """
    # Verify semester exists and user has access
    try:
        semester = semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Check if semester is in correct status
    if semester["status"] not in ["draft", "active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upload applicants to semester in '{semester['status']}' status"
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV (.csv extension)"
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))
        
        logger.info(f"Received CSV with {len(df)} rows for semester {semester_id}")
        
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading CSV file: {str(e)}"
        )
    
    # Upload applicants
    try:
        result = applicant_service.upload_applicants(
            df=df,
            semester_id=semester_id,
            organization_id=user.organization_id,
            uploaded_by=user.email
        )
        
        from datetime import datetime, timezone
        
        return CSVUploadResponse(
            total_uploaded=result["total_uploaded"],
            mentors=result["mentors"],
            mentees=result["mentees"],
            duplicates_skipped=result["duplicates_skipped"],
            errors=result.get("errors"),
            semester_id=semester_id,
            uploaded_by=user.email,
            uploaded_at=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error uploading applicants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading applicants: {str(e)}"
        )


@router.get(
    "/",
    response_model=ApplicantListResponse,
    summary="List applicants for semester",
    description="Get list of applicants with optional role filter"
)
async def list_applicants(
    semester_id: str = Query(..., description="Semester ID"),
    role: Optional[str] = Query(None, description="Filter by role: mentor or mentee"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(100, ge=1, le=500, description="Max results (max 500)"),
    applicant_service: ApplicantServiceDep = Depends(),
    semester_service: SemesterServiceDep = Depends(),
    user: UserInDB = Depends(get_current_user)
):
    """
    List applicants for semester
    
    Query parameters:
    - semester_id: Required semester ID
    - role: Optional filter (mentor or mentee)
    - skip: Pagination offset
    - limit: Maximum results (max 500)
    """
    # Verify semester exists and user has access
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Validate role filter
    if role and role not in ["mentor", "mentee"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'mentor' or 'mentee'"
        )
    
    applicants = applicant_service.get_applicants(
        semester_id=semester_id,
        role=role,
        skip=skip,
        limit=limit
    )
    
    applicant_responses = [
        ApplicantResponse(
            applicant_id=app["applicant_id"],
            ufl_email=app["ufl_email"],
            full_name=app["full_name"],
            role=app["role"],
            status=app["status"],
            submitted_at=app["submitted_at"].isoformat() if hasattr(app["submitted_at"], "isoformat") else app["submitted_at"],
            survey_responses=app["survey_responses"]
        )
        for app in applicants
    ]
    
    return ApplicantListResponse(
        applicants=applicant_responses,
        count=len(applicant_responses),
        semester_id=semester_id,
        filter_role=role
    )


@router.get(
    "/{applicant_id}",
    response_model=ApplicantResponse,
    summary="Get applicant details",
    description="Get detailed information for a specific applicant"
)
async def get_applicant(
    applicant_id: str,
    semester_id: str = Query(..., description="Semester ID"),
    applicant_service: ApplicantServiceDep = Depends(),
    semester_service: SemesterServiceDep = Depends(),
    user: UserInDB = Depends(get_current_user)
):
    """
    Get applicant details
    
    Returns full applicant profile including survey responses.
    """
    # Verify semester access
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Get applicant
    applicants = applicant_service.get_applicants(
        semester_id=semester_id,
        skip=0,
        limit=1
    )
    
    # Filter by applicant_id (service returns list)
    applicant = next((a for a in applicants if a["applicant_id"] == applicant_id), None)
    
    if not applicant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Applicant {applicant_id} not found in semester {semester_id}"
        )
    
    return ApplicantResponse(
        applicant_id=applicant["applicant_id"],
        ufl_email=applicant["ufl_email"],
        full_name=applicant["full_name"],
        role=applicant["role"],
        status=applicant["status"],
        submitted_at=applicant["submitted_at"].isoformat() if hasattr(applicant["submitted_at"], "isoformat") else applicant["submitted_at"],
        survey_responses=applicant["survey_responses"]
    )


@router.patch(
    "/{applicant_id}",
    response_model=dict,
    dependencies=[Depends(PermissionChecker("can_upload_applicants"))],
    summary="Update applicant",
    description="Update applicant survey responses"
)
async def update_applicant(
    applicant_id: str,
    semester_id: str = Query(..., description="Semester ID"),
    request: ApplicantUpdateRequest = ...,
    applicant_service: ApplicantServiceDep = Depends(),
    semester_service: SemesterServiceDep = Depends(),
    user: UserInDB = Depends(get_current_user)
):
    """
    Update applicant survey responses
    
    Protected fields (role, submitted_at, embeddings, status) cannot be updated.
    Only survey response fields can be modified.
    """
    # Verify semester access
    try:
        semester = semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Check semester status
    if semester["status"] not in ["draft", "active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update applicants in locked semester"
        )
    
    # Build updates dict
    updates = request.model_dump(exclude_unset=True)
    
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )
    
    updated = applicant_service.update_applicant(
        applicant_id=applicant_id,
        semester_id=semester_id,
        updates=updates
    )
    
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Applicant not found or no changes made"
        )
    
    return {
        "message": "Applicant updated successfully",
        "applicant_id": applicant_id,
        "updated_fields": list(updates.keys())
    }


@router.delete(
    "/{applicant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(PermissionChecker("can_upload_applicants"))],
    summary="Remove applicant from semester",
    description="Remove applicant's application for a specific semester"
)
async def delete_applicant(
    applicant_id: str,
    semester_id: str = Query(..., description="Semester ID"),
    applicant_service: ApplicantServiceDep = Depends(),
    semester_service: SemesterServiceDep = Depends(),
    user: UserInDB = Depends(get_current_user)
):
    """
    Remove applicant from semester
    
    Note: This does not delete the applicant document, only removes
    the application for this specific semester.
    """
    # Verify semester access
    try:
        semester = semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Check semester status
    if semester["status"] not in ["draft", "active"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete applicants from locked semester"
        )
    
    deleted = applicant_service.delete_applicant(
        applicant_id=applicant_id,
        semester_id=semester_id
    )
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Applicant not found in this semester"
        )
    
    return None  # 204 No Content


@router.get(
    "/validate-csv",
    summary="Validate CSV without uploading",
    description="Dry-run CSV validation to check for errors before actual upload"
)
async def validate_csv_file(
    semester_id: str = Query(..., description="Target semester ID"),
    file: UploadFile = File(..., description="CSV file to validate"),
    applicant_service: ApplicantServiceDep = Depends(),
    semester_service: SemesterServiceDep = Depends(),
    user: UserInDB = Depends(get_current_user)
):
    """
    Validate CSV without uploading
    
    Performs all validation checks but does not insert data.
    Useful for checking data quality before actual upload.
    """
    # Verify semester access
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a CSV (.csv extension)"
        )
    
    try:
        # Read CSV
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))
        
        # Validate only (no upload)
        is_valid, errors = applicant_service.validate_csv(
            df=df,
            semester_id=semester_id,
            organization_id=user.organization_id
        )
        
        return {
            "is_valid": is_valid,
            "total_rows": len(df),
            "errors": errors if errors else [],
            "semester_id": semester_id
        }
        
    except Exception as e:
        logger.error(f"Error validating CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading/validating CSV: {str(e)}"
        )