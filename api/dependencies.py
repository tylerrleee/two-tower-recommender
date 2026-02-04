"""
Dependency Injection Container

Provides centralized dependency management for:
- Database connections
- Service instances
- Inference engine
- User authentication
"""

from functools import lru_cache
from fastapi import Depends
from pymongo.database import Database
from typing import Annotated

from database.connection import get_database
from api.inference import MatchingInference
from api.services.organization_service import OrganizationService
from api.services.semester_service import SemesterService
from api.services.applicant_service import ApplicantService
from api.services.matching_service import MatchingService
from api.services.feedback_service import FeedbackService

# --- Database Dependency ---
DatabaseDep = Annotated[Database, Depends(get_database)]


# --- Inference Engine (Singleton) ---
@lru_cache()

def get_inference_engine() -> MatchingInference:
    """
    Get or create inference engine (singleton)
    
    Loaded once on first request, reused for all subsequent requests
    """
    engine = MatchingInference(
        model_path="models/best_model.pt",
        sbert_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384
    )
    engine.load_model()
    return engine

# --- Service Dependencies ---
def get_organization_service(db: DatabaseDep) -> OrganizationService:
    """Get organization service with injected database"""
    return OrganizationService(db)

def get_semester_service(db: DatabaseDep) -> SemesterService:
    """Get semester service with injected database"""
    return SemesterService(db)

def get_applicant_service(db: DatabaseDep) -> ApplicantService:
    """Get applicant service with injected database"""
    return ApplicantService(db)

def get_matching_service(db: DatabaseDep) -> MatchingService:
    """Get matching service with injected database"""
    return MatchingService(db)

def get_feedback_service(db: DatabaseDep) -> FeedbackService:
    """Get feedback service with injected database"""
    return FeedbackService(db)

# --- Type Aliases for Clean Signatures ---
InferenceEngineDep      = Annotated[MatchingInference, Depends(get_inference_engine)]
OrganizationServiceDep  = Annotated[OrganizationService, Depends(get_organization_service)]
SemesterServiceDep      = Annotated[SemesterService, Depends(get_semester_service)]
ApplicantServiceDep     = Annotated[ApplicantService, Depends(get_applicant_service)]
MatchingServiceDep      = Annotated[MatchingService, Depends(get_matching_service)]
FeedbackServiceDep      = Annotated[FeedbackService, Depends(get_feedback_service)]
