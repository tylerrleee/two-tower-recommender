"""
Docstring for api.routers.auth_router

Auth Router - Authentication endpoints

# Endpoints for /auth
POST   /auth/login            # JWT authentication
POST   /auth/register         # User registration
GET    /auth/me               # Current user info
POST   /auth/refresh          # Refresh token
POST   /auth/change-password  # Password change
POST   /auth/logout           # Logout


"""


from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime, timedelta, timezone
import logging

from api.auth import (
    get_current_user,
    UserInDB,
    create_access_token,
    get_password_hash,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from api.dependencies import DatabaseDep
# DatabaseDep = Annotated[Database, Depends(get_database)]

router = APIRouter()
logger = logging.getLogger(__name__)


# PYDANTIC MODELS
class Token(BaseModel):
    """Response model for JWT token"""
    access_token: str
    token_type  : str = "bearer"
    expires_in  : int  # seconds


class UserRegisterRequest(BaseModel):
    """Request model for user registration"""
    email           : EmailStr = Field(..., description="User email address")
    password        : str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name       : str = Field(..., min_length=1, max_length=200)
    organization_id : str = Field(..., description="Organization ID")
    role            : str = Field(default="coordinator", description="User role: admin, coordinator, viewer")


class UserResponse(BaseModel):
    """Response model for user details"""
    email           : str
    full_name       : str
    organization_id : str
    role            : str
    is_active       : bool
    permissions     : dict
    created_at      : str


class ChangePasswordRequest(BaseModel):
    """Request model for password change
    """
    current_password: str
    new_password    : str = Field(..., min_length=8)


# POST   /auth/login            # JWT authentication

@router.post(
    "/login",
    response_model  = Token,
    summary         = "User login",
    description     = "Authenticate user and return JWT token"
)
async def login(
    db: DatabaseDep,
    form_data: OAuth2PasswordRequestForm = Depends(),

):
    """
    User login
    
    Authenticates user with email/password and returns JWT access token.
    
    Form data:
    - username: User email
    - password: User password
    
    Returns:
    - access_token: JWT token for API authentication
    - token_type: "bearer"
    - expires_in: Token expiration in seconds
    """
    # Find user
    user = db.users.find_one({"email": form_data.username.lower()})
    
    if not user:
        logger.warning(f"Login failed: user {form_data.username} not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user["hashed_password"]):
        logger.warning(f"Login failed: incorrect password for {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user["email"],
            "organization_id": str(user["organization_id"]),
            "role": user["role"]
        }
    )
    
    logger.info(f"User {user['email']} logged in successfully")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.post(
    "/register",
    response_model  = UserResponse,
    status_code     = status.HTTP_201_CREATED,
    summary         = "Register new user",
    description     = "Create new user account"
)
async def register_user(
    request: UserRegisterRequest,
    db: DatabaseDep
):
    """
    Register new user
    
    Creates a new user account with specified role and organization.
    
    Note: In production, this endpoint should:
    1. Require admin authentication or invitation token
    2. Send email verification
    3. Validate organization exists
    
    For now, allows open registration for development.
    """
    # Check if user already exists
    existing = db.users.find_one({"email": request.email.lower()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Validate organization exists
    from bson import ObjectId
    org = db.organizations.find_one({"_id": ObjectId(request.organization_id)})
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    # Validate role
    valid_roles = ["admin", "coordinator", "viewer"]
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    # Set default permissions based on role
    permissions = {
        "admin": {
            "can_upload_applicants": True,
            "can_trigger_matching": True,
            "can_view_results": True,
            "can_manage_users": True,
            "can_create_semester": True
        },
        "coordinator": {
            "can_upload_applicants": True,
            "can_trigger_matching": True,
            "can_view_results": True,
            "can_manage_users": False,
            "can_create_semester": True
        },
        "viewer": {
            "can_upload_applicants": False,
            "can_trigger_matching": False,
            "can_view_results": True,
            "can_manage_users": False,
            "can_create_semester": False
        }
    }
    
    # Create user document
    user_doc = {
        "email": request.email.lower(),
        "full_name": request.full_name,
        "hashed_password": get_password_hash(request.password),
        "organization_id": ObjectId(request.organization_id),
        "role": request.role,
        "is_active": True,
        "permissions": permissions[request.role],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    
    result = db.users.insert_one(user_doc)
    
    logger.info(f"Created new user: {request.email} ({request.role})")
    
    return UserResponse(
        email=user_doc["email"],
        full_name=user_doc["full_name"],
        organization_id=str(user_doc["organization_id"]),
        role=user_doc["role"],
        is_active=user_doc["is_active"],
        permissions=user_doc["permissions"],
        created_at=user_doc["created_at"].isoformat()
    )

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get authenticated user's information"
)
async def get_current_user_info(
    user: UserInDB = Depends(get_current_user)
):
    """
    Get current user information
    
    Returns details of the authenticated user.
    Requires valid JWT token.
    """
    return UserResponse(
        email=user.email,
        full_name=user.full_name,
        organization_id=user.organization_id,
        role=user.role,
        is_active=user.is_active,
        permissions=user.permissions,
        created_at=user.created_at.isoformat() if hasattr(user.created_at, "isoformat") else user.created_at
    )


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token",
    description="Generate new access token from valid token"
)
async def refresh_token(
    user: UserInDB = Depends(get_current_user)
):
    """
    Refresh access token
    
    Generates a new JWT token for authenticated user.
    Useful for extending session without re-login.
    """
    access_token = create_access_token(
        data={
            "sub": user.email,
            "organization_id": user.organization_id,
            "role": user.role
        }
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.post(
    "/change-password",
    response_model=dict,
    summary="Change password",
    description="Change user's password"
)
async def change_password(
    request: ChangePasswordRequest,
    db: DatabaseDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Change password
    
    Allows authenticated user to change their password.
    Requires current password verification.
    """
    # Get user from database
    db_user = db.users.find_one({"email": user.email.lower()})
    
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # Verify current password
    if not verify_password(request.current_password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Update password
    new_hashed_password = get_password_hash(request.new_password)
    
    db.users.update_one(
        {"email": user.email.lower()},
        {
            "$set": {
                "hashed_password": new_hashed_password,
                "updated_at": datetime.now(timezone.utc)
            }
        }
    )
    
    logger.info(f"User {user.email} changed password")
    
    return {"message": "Password changed successfully"}

@router.post(
    "/logout",
    response_model=dict,
    summary="Logout",
    description="Logout user (client-side token deletion)"
)
async def logout(
    user: UserInDB = Depends(get_current_user)
):
    """
    Logout user
    
    Note: Since we're using stateless JWT tokens, logout is handled client-side
    by deleting the token. This endpoint serves as a confirmation.
    
    For production, consider:
    1. Token blacklisting (requires Redis/database)
    2. Short token expiration times
    3. Refresh token rotation
    """
    logger.info(f"User {user.email} logged out")
    
    return {
        "message": "Logged out successfully",
        "note": "Please delete your access token on the client side"
    }
