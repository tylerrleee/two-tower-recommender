"""
Pydantic Schemas for API JSON requests and responses

"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Literal, Optional
from enum import Enum

class YearLevel(str, Enum):
    FRESHMAN    = "Freshman"
    SOPHOMORE   = "Sophomore"
    JUNIOR      = "Junior"
    SENIOR      = "Senior"
    GRADUATE    = "Graduate"

class Role(int, Enum):
    MENTOR = 0
    MENTEE = 1

class ApplicantInput(BaseModel):
    """Single applicant data """
    role    : Role
    name    : str = Field(..., min_length=1, max_length=100)
    email   : EmailStr
    major   : str = Field(..., min_length=1, max_length=100)
    year    : YearLevel
    
    # Optional fields for matching
    bio             : Optional[str] = Field(None, max_length=1000)
    interests       : Optional[str] = Field(None, max_length=500)
    goals           : Optional[str] = Field(None, max_length=500)

    # Scales
    extroversion    : Optional[int] = Field(None, ge=1, le=5)
    study_frequency : Optional[int] = Field(None, ge=1, le=5)
    gym_frequency   : Optional[int] = Field(None, ge=1, le=5)

    class Config:
        json_schema_extra = {
            "example": {
                "role"      : 0,
                "name"      : "David Chen",
                "email"     : "david.chen@ufl.edu",
                "major"     : "Computer Science",
                "year"      :  "Junior",
                "bio"       : "Passionate about AI and machine learning",
                "interests" : "AI research, entrepreneurship, Vietnamese culture",
                "goals"     : "Help freshmen navigate CS curriculum",
                "extroversion"      : 4,
                "study_frequency"   : 5,
                "gym_frequency"     : 3
            }
        }
    # TODO: Update the input 

class BatchMatchingRequest(BaseModel):
    """Request for batch matching from CSV data"""

    applicants: List[ApplicantInput] = Field(..., min_items=3)
    use_faiss: bool = Field(False, description="Use FAISS for faster approximate matching")
    top_k: int = Field(10, ge=5, le=50, description="Number of candidates for FAISS")
    
    @field_validator('applicants')
    def validate_mentor_mentee_ratio(cls, v):
        mentors = sum(1 for app in v if app.role == Role.MENTOR)
        mentees = sum(1 for app in v if app.role == Role.MENTEE)
        
        if mentors == 0:
            raise ValueError("At least one mentor required")
        if mentees == 0:
            raise ValueError("At least one mentee required")
        if mentees < 2 * mentors:
            raise ValueError(f"Need at least 2 mentees per mentor. Got {mentees} mentees for {mentors} mentors")
        
        return v

class CSVUploadRequest(BaseModel):
    """Request for CSV file upload matching"""
    csv_content: str    = Field(..., description="Base64 encoded CSV file content")
    use_faiss: bool     = False
    top_k: int          = 10

class MentorInfo(BaseModel):
    """Mentor information in response"""
    name : str
    major: str
    email: EmailStr

class MenteeInfo(BaseModel):
    """Mentee information in response"""
    name : str
    major: str
    year : YearLevel

class MatchGroup(BaseModel):
    """Single mentor-mentee group match"""
    group_id: int
    mentor: MentorInfo
    mentees: List[MenteeInfo]
    compatibility_score: float = Field(..., ge=0, le=1)
    individual_scores: List[float]

class MatchingResponse(BaseModel):
    """Complete matching results"""
    status: Literal["success", "partial_success", "failure"]
    total_groups: int
    average_compatibility: float
    groups: List[MatchGroup]
    warnings: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_groups": 2,
                "average_compatibility": 0.82,
                "groups": [
                    {
                        "group_id": 0,
                        "mentor": {
                            "name": "David Chen",
                            "major": "Computer Science",
                            "email": "david.chen@ufl.edu"
                        },
                        "mentees": [
                            {"name": "Anna Nguyen", "major": "Biology", "year": "Freshman"},
                            {"name": "John Smith", "major": "Computer Science", "year": "Freshman"}
                        ],
                        "compatibility_score": 0.85,
                        "individual_scores": [0.83, 0.87]
                    }
                ]
            }
        }

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status      : Literal["healthy", "unhealthy"]
    model_loaded: bool
    version     : str
    timestamp   : str

class ErrorResponse(BaseModel):
    """Error response schema"""
    error       : str
    detail      : Optional[str] = None
    timestamp   : str