"""
Mocking:
1. database
2. authentication
3. dependency injection
"""
import sys
import os

# Set key before importing auth
os.environ["JWT_SECRET_KEY"] = "test_secret_key_123"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from api.main import app
from api.dependencies import get_database, get_current_user
from api.auth import UserInDB



# 1. Mock Database Fixture
@pytest.fixture
def mock_db():
    """Returns a mock MongoDB object"""
    db              = MagicMock()
    # Mock collections
    db.users        = MagicMock()
    db.semesters    = MagicMock()
    db.matching_jobs = MagicMock()
    db.feedback     = MagicMock()
    return db

# 2. Override Database Dependency
@pytest.fixture
def client(mock_db):
    """
    Creates a TestClient with the database dependency overridden.
    This ensures API calls use our 'mock_db' instead of trying to connect to Atlas.
    """
    app.dependency_overrides[get_database] = lambda: mock_db
    
    with TestClient(app) as c:
        yield c
    
    # Cleanup
    app.dependency_overrides.clear()

# 3. Mock User (Authenticated)
@pytest.fixture
def mock_user_admin():
    return UserInDB(
        email           = "admin@ufl.edu",
        hashed_password = "fakehash",
        full_name       = "Admin User",
        organization_id = "org_123",
        role            = "admin",
        is_active       = True,
        permissions     = {"can_create_semester": True}
    )

@pytest.fixture
def authorized_client(client, mock_user_admin):
    """
    Overrides the authentication dependency to simulate a logged-in user.
    No need to generate real JWTs for checking logic
    """
    app.dependency_overrides[get_current_user] = lambda: mock_user_admin
    return client