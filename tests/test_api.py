import pytest
from fastapi.testclient import TestClient
from api.main import app

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