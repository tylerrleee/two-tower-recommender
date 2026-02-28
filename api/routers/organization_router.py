"""
Docstring for api.routers.organization_router

Endpoints:
POST   /organizations/                      # Create org (admin)
GET    /organizations/                      # List orgs (admin)
GET    /organizations/{org_id}              # Get org details
PATCH  /organizations/{org_id}              # Update org
GET    /organizations/{org_id}/stats        # Get statistics
POST   /organizations/{org_id}/check-quota  # Check quota

"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from api.auth import get_current_user, UserInDB, RoleChecker
from api.dependencies import OrganizationServiceDep
from api.models import ErrorResponse

router = APIRouter()

# PYDANTIC

class OrganizationCreateRequest(BaseModel):
    """Request model for creating organization"""
    name: str = Field(..., description="Organization name", min_length=1, max_length=200)
    subdomain: str = Field(..., description="Unique subdomain", pattern="^[a-z0-9-]+$", min_length=3, max_length=50)
    owner_email: str = Field(..., description="Owner's email address")
    plan: str = Field(default="free", description="Subscription plan: free, premium, enterprise")


class OrganizationUpdateRequest(BaseModel):
    """Request model for updating organization"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    plan: Optional[str] = Field(None, description="Subscription plan")
    is_active: Optional[bool] = None


class OrganizationResponse(BaseModel):
    """Response model for organization details"""
    organization_id: str = Field(..., alias="_id")
    name: str
    subdomain: str
    plan: str
    owner_email: str
    is_active: bool
    created_at: str
    updated_at: str
    
    class Config:
        populate_by_name = True


class OrganizationStatsResponse(BaseModel):
    """Response model for organization statistics"""
    total_users: int
    total_semesters: int
    total_applicants: int
    total_matches: int
    quota_usage: dict


class QuotaCheckRequest(BaseModel):
    """Request model for quota checking"""
    resource_type: str = Field(..., description="Resource type: applicants, semesters, matching_jobs")
    requested_amount: int = Field(default=1, ge=1)


class QuotaCheckResponse(BaseModel):
    """Response model for quota check"""
    resource_type: str
    requested_amount: int
    quota_available: bool
    current_usage: Optional[int] = None
    max_allowed: Optional[int] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(RoleChecker(["admin"]))],
    summary="Create new organization",
    description="Create a new organization. Admin access required."
)
async def create_organization(
    request: OrganizationCreateRequest,
    org_service: OrganizationServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Create new organization
    
    Only admin users can create organizations.
    Subdomain must be unique and follow naming conventions (lowercase, alphanumeric, hyphens).
    """
    try:
        org_id = org_service.create_organization(
            name=request.name,
            subdomain=request.subdomain,
            owner_email=request.owner_email,
            plan=request.plan
        )
        
        org = org_service.get_organization(org_id)
        
        return OrganizationResponse(
            _id=org["_id"],
            name=org["name"],
            subdomain=org["subdomain"],
            plan=org["plan"],
            owner_email=org["owner_email"],
            is_active=org["is_active"],
            created_at=org["created_at"].isoformat(),
            updated_at=org["updated_at"].isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/",
    response_model=List[OrganizationResponse],
    dependencies=[Depends(RoleChecker(["admin"]))],
    summary="List all organizations",
    description="List all organizations with pagination. Admin access required."
)
async def list_organizations(
    org_service: OrganizationServiceDep,
    user: UserInDB = Depends(get_current_user),
    skip: int = 0,
    limit: int = 50,
    is_active: Optional[bool] = None
):
    """
    List all organizations
    
    Query parameters:
    - skip: Number of records to skip (pagination)
    - limit: Maximum records to return (max 100)
    - is_active: Filter by active status
    """
    if limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit cannot exceed 100"
        )
    
    orgs = org_service.list_organizations(skip=skip, limit=limit, is_active=is_active)
    
    return [
        OrganizationResponse(
            _id=org["_id"],
            name=org["name"],
            subdomain=org["subdomain"],
            plan=org["plan"],
            owner_email=org["owner_email"],
            is_active=org["is_active"],
            created_at=org["created_at"].isoformat() if isinstance(org["created_at"], datetime) else org["created_at"],
            updated_at=org["updated_at"].isoformat() if isinstance(org["updated_at"], datetime) else org["updated_at"]
        )
        for org in orgs
    ]


@router.get(
    "/{organization_id}",
    response_model=OrganizationResponse,
    summary="Get organization details",
    description="Get organization details. Users can only access their own organization unless admin."
)
async def get_organization(
    organization_id: str,
    org_service: OrganizationServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Get organization details
    
    Users can only access their own organization unless they have admin role.
    """
    # Authorization: users can only access their own org, admins can access all
    if user.role != "admin" and user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You can only access your own organization."
        )
    
    try:
        org = org_service.get_organization(organization_id)
        
        return OrganizationResponse(
            _id=org["_id"],
            name=org["name"],
            subdomain=org["subdomain"],
            plan=org["plan"],
            owner_email=org["owner_email"],
            is_active=org["is_active"],
            created_at=org["created_at"].isoformat() if isinstance(org["created_at"], datetime) else org["created_at"],
            updated_at=org["updated_at"].isoformat() if isinstance(org["updated_at"], datetime) else org["updated_at"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{organization_id}",
    response_model=dict,
    summary="Update organization",
    description="Update organization settings. Only admins or organization owners can update."
)
async def update_organization(
    organization_id: str,
    request: OrganizationUpdateRequest,
    org_service: OrganizationServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Update organization settings
    
    Only admins or organization owners can update settings.
    Protected fields (subdomain, created_at, owner_email) cannot be updated.
    """
    # Authorization check
    if user.role != "admin" and user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Only admins or organization owners can update."
        )
    
    # Build update dict (only non-None fields)
    updates = request.model_dump(exclude_unset=True)
    
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )
    
    updated = org_service.update_organization(organization_id, updates)
    
    if not updated:
        return {"message": "No changes made (all fields protected or unchanged)"}
    
    return {"message": "Organization updated successfully", "updated_fields": list(updates.keys())}


@router.get(
    "/{organization_id}/stats",
    response_model=OrganizationStatsResponse,
    summary="Get organization statistics",
    description="Get usage statistics and quota information"
)
async def get_organization_stats(
    organization_id: str,
    org_service: OrganizationServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Get organization statistics
    
    Returns:
    - Total users, semesters, applicants, matches
    - Quota usage percentages
    """
    # Authorization
    if user.role != "admin" and user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )
    
    try:
        stats = org_service.get_organization_stats(organization_id)
        return OrganizationStatsResponse(**stats)
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{organization_id}/check-quota",
    response_model=QuotaCheckResponse,
    summary="Check quota availability",
    description="Check if organization has quota available for resource"
)
async def check_quota(
    organization_id: str,
    request: QuotaCheckRequest,
    org_service: OrganizationServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Check quota availability
    
    Validates if organization can add requested amount of resource.
    Resource types: applicants, semesters, matching_jobs
    """
    # Authorization
    if user.role != "admin" and user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied."
        )
    
    if request.resource_type not in ["applicants", "semesters", "matching_jobs"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid resource_type. Must be: applicants, semesters, or matching_jobs"
        )
    
    quota_available = org_service.check_quota(
        organization_id,
        request.resource_type,
        request.requested_amount
    )
    
    # Get org for quota details
    org = org_service.get_organization(organization_id)
    
    response_data = {
        "resource_type": request.resource_type,
        "requested_amount": request.requested_amount,
        "quota_available": quota_available
    }
    
    # Add current usage and max_allowed info
    if request.resource_type == "semesters":
        response_data["max_allowed"] = org["settings"]["allowed_semesters"]
    elif request.resource_type == "applicants":
        response_data["max_allowed"] = org["settings"]["max_applicants_per_semester"]
    elif request.resource_type == "matching_jobs":
        response_data["max_allowed"] = org["settings"]["max_concurrent_matching_jobs"]
    
    return QuotaCheckResponse(**response_data)