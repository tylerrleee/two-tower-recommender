import datetime
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import os

from database.connection import get_database


# Config
# TODO update .env for these configs

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24h

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# Pydantic models
class TokenData(BaseModel):
    email: Optional[str]            = None
    organization_id: Optional[str]  = None
    role: Optional[str]             = None

class Token(BaseModel):
    access_token: str
    token_type  : str
    expires_in  : int

class UserInDB(BaseModel):
    email           : str
    hashed_password : str
    full_name       : str
    organization_id : str
    role            : str
    is_active       : bool
    permissions     : dict

# CORE FUNCTIONS

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    Return True if matched
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()

    # Standardize subject -- map to sub through email
    if "sub" not in to_encode and "email" in to_encode:
         to_encode["sub"] = to_encode["email"]

    if expires_delta:
        expire = datetime.datetime.now(datetime.UTC) + expires_delta
    else:
        expire = datetime.datetime.now(datetime.UTC) \
                + datetime.timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, 
                             algorithm = ALGORITHM)
    
    return encoded_jwt

# send multiple get user simultaneously 
async def get_current_user(
            token: str = Depends(oauth2_scheme),
            db         = Depends(get_database)  # Inject MongoDB
        ) -> UserInDB:
    """
    Extract and validate user from JWT token
    
    Used as dependency in protected endpoints:
    @app.get("/match/results")
    async def get_results(user: UserInDB = Depends(get_current_user)):
        ...
    """
    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail      = "Could not validate credentials",
        headers     = {"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode token
        payload     = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str  = payload.get("sub")
        
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)

        # Fetch user from database
        user_doc = db.users.find_one({"email": email})
        
        if user_doc is None:
            raise credentials_exception
        
        # Check if user is active | raise status code if they're not active
        if not user_doc.get("is_active", False):
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = "User account is disabled"
            )
        
        return UserInDB(
            email           = user_doc["email"],
            hashed_password = user_doc["hashed_password"],
            full_name       = user_doc["full_name"],
            organization_id = str(user_doc["organization_id"]),
            role            = user_doc["role"],
            is_active       = user_doc["is_active"],
            permissions     = user_doc.get("permissions", {})
        )
        
    except JWTError:
        raise credentials_exception

# ROLE ACCESS CONTROL
class RoleChecker:
    """
    Dependency class for role-based access control
    
    Usage:
    @app.post("/match/batch")
    async def trigger_match(
        user: UserInDB = Depends(get_current_user),
        _: None = Depends(RoleChecker(["admin", "coordinator"]))
    ):
        # Only admins and coordinators can trigger matching
        ...
    """
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: UserInDB = Depends(get_current_user)):
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = f"Operation not permitted for role: {user.role}"
            )
        return None

class PermissionChecker:
    """
    Dependency class for granular permission checks
    
    Usage:
    @app.post("/applicants/upload")
    async def upload_csv(
        _: None = Depends(PermissionChecker("can_upload_applicants"))
    ):
        ...
    """
    def __init__(self, required_permission: str):
        self.required_permission = required_permission
    
    def __call__(self, user: UserInDB = Depends(get_current_user)):
        if not user.permissions.get(self.required_permission, False):           
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = f"Missing permission: {self.required_permission}"
            )
        return None