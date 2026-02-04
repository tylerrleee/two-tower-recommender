import pytest
from fastapi.testclient import TestClient
from fastapi import status
from api.main import app
from datetime import datetime

client = TestClient(app)

def test_health_check():
    """ Validate health check 
    Reference api/models.py 
    """
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_batch_matching_success():
    payload = {
        "applicants": [
            {
                "role": 0,
                "name": "Mentor 1",
                "ufl_email": "mentor1@ufl.edu",
                "major": "CS",
                "year": "Junior",
                "bio": "Loves AI",
                "interests": "ML, research",
                "goals": "Help students"
            },
            {
                "role": 1,
                "name": "Mentee 1",
                "ufl_email": "mentee1@ufl.edu",
                "major": "CS",
                "year": "Freshman",
                "bio": "Interested in AI",
                "interests": "ML, coding",
                "goals": "Learn AI"
            },
            {
                "role": 1,
                "name": "Mentee 2",
                "ufl_email": "mentee2@ufl.edu",
                "major": "Math",
                "year": "Freshman",
                "bio": "Math enthusiast",
                "interests": "algorithms, theory",
                "goals": "Research"
            }
        ],
        "use_faiss": False
    }
    
    response = client.post("/match/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_groups"] >= 1

def test_batch_matching_insufficient_data():
    payload = {
        "applicants": [
            {
                "role": 0,
                "name": "Mentor 1",
                "ufl_email": "mentor1@ufl.edu",
                "major": "CS",
                "year": "Junior"
            }
            # Missing mentees
        ]
    }
    
    response = client.post("/match/batch", json=payload)
    assert response.status_code == 422  # Validation error


class TestSemesterEndpoints:
    def test_create_semester_success(self, authorized_client, mock_db):
        """Test happy path for creating a semester"""
        # Mock DB insert return
        mock_db.semesters.insert_one.return_value.inserted_id = "new_sem_id"
        mock_db.semesters.find_one.return_value = {
            "_id": "new_sem_id",
            "name": "Fall 2026",
            "status": "active",
            "created_at": datetime.now()
        }

        payload = {
            "name": "Fall 2026",
            "start_date": "2026-08-20T00:00:00",
            "end_date": "2026-12-15T00:00:00",
            "mentor_quota": 50,
            "mentee_quota": 100
        }

        # Action
        response = authorized_client.post("/semesters/", json=payload)

        # Assert
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["semester_id"] == "new_sem_id"

    def test_create_semester_unauthorized(self, client):
        """Test without auth headers"""
        response = client.post("/semesters/", json={})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

class TestRateLimiter:
    def test_rate_limit_exceeded(self, authorized_client, mock_user_admin):
        """
        To test this properly, we need to inspect the middleware or 
        mock the limit to be very low.
        """
        # Simulate 101 requests
        # api.middleware.rate_limit.RateLimitMiddleware._check_request_limit
        
        from api.middleware.rate_limit import RateLimitMiddleware
        
        # Create a clean middleware instance just for testing logic
        mw = RateLimitMiddleware(None)
        
        # Hit limit
        for _ in range(100):
            assert mw._check_request_limit("user@test.com") is True
            
        # 101st request should fail
        assert mw._check_request_limit("user@test.com") is False