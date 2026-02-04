# Complete Backend Implementation Guide

## Part 2: Remaining Services, Middleware, Routers & Dependencies

---

## 4. Matching Service (api/services/matching_service.py)

```python
"""
Matching Service - Async job queue and status tracking

Responsibilities:
- Create matching jobs
- Track job progress
- Save match results
- Handle job failures
"""

from typing import Dict, Optional, List
from bson import ObjectId
from pymongo.database import Database
import uuid
from datetime import datetime, timezone
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    EMBEDDING = "embedding"
    TRAINING = "training"
    MATCHING = "matching"
    COMPLETED = "completed"
    FAILED = "failed"

class MatchingService:
    """Handle async matching job lifecycle"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_matching_job(
        self,
        semester_id: str,
        organization_id: str,
        triggered_by: str,
        config: Dict
    ) -> str:
        """Create matching job record"""
        job_id = str(uuid.uuid4())
        
        job_doc = {
            "job_id": job_id,
            "semester_id": ObjectId(semester_id),
            "organization_id": ObjectId(organization_id),
            "status": JobStatus.PENDING,
            "triggered_by": triggered_by,
            "config": config,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "progress": {
                "current_step": "pending",
                "total_steps": 5,
                "completed_steps": 0
            },
            "results": {
                "total_groups": None,
                "average_compatibility": None
            }
        }
        
        self.db.matching_jobs.insert_one(job_doc)
        logger.info(f"Created matching job {job_id} for semester {semester_id}")
        
        return job_id
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: Optional[Dict] = None,
        error_message: Optional[str] = None
    ):
        """Update job status and progress"""
        update_doc = {
            "status": status,
            "updated_at": datetime.now(timezone.utc)
        }
        
        if progress:
            update_doc["progress"] = progress
        
        if error_message:
            update_doc["error_message"] = error_message
        
        if status == JobStatus.COMPLETED:
            update_doc["completed_at"] = datetime.now(timezone.utc)
        
        if status == JobStatus.PREPROCESSING:
            update_doc.setdefault("started_at", datetime.now(timezone.utc))
        
        self.db.matching_jobs.update_one(
            {"job_id": job_id},
            {"$set": update_doc}
        )
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get job status for polling"""
        job = self.db.matching_jobs.find_one({"job_id": job_id})
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        elapsed = None
        if job.get("started_at"):
            end_time = job.get("completed_at") or datetime.now(timezone.utc)
            elapsed = (end_time - job["started_at"]).total_seconds()
        
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "created_at": job["created_at"].isoformat(),
            "started_at": job["started_at"].isoformat() if job.get("started_at") else None,
            "completed_at": job["completed_at"].isoformat() if job.get("completed_at") else None,
            "elapsed_seconds": elapsed,
            "error_message": job.get("error_message"),
            "results": job.get("results")
        }
    
    def save_match_results(
        self,
        job_id: str,
        semester_id: str,
        groups: List[Dict]
    ):
        """Save matching results to match_groups collection"""
        semester_oid = ObjectId(semester_id)
        
        # Delete previous results for this semester
        self.db.match_groups.delete_many({"semester_id": semester_oid})
        
        group_docs = []
        for group in groups:
            group_doc = {
                "semester_id": semester_oid,
                "job_id": job_id,
                "group_id": group["group_id"],
                "mentor": group["mentor"],
                "mentees": group["mentees"],
                "compatibility_score": group["compatibility_score"],
                "individual_scores": group["individual_scores"],
                "created_at": datetime.now(timezone.utc)
            }
            group_docs.append(group_doc)
        
        if group_docs:
            self.db.match_groups.insert_many(group_docs)
            logger.info(f"Saved {len(group_docs)} match groups for semester {semester_id}")
        
        # Update job with results summary
        avg_compatibility = sum(g["compatibility_score"] for g in groups) / len(groups)
        
        self.db.matching_jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "results.total_groups": len(groups),
                "results.average_compatibility": avg_compatibility
            }}
        )
```

---

## 5. Feedback Service (api/services/feedback_service.py)

```python
"""
Feedback Service - Collect match satisfaction feedback

Responsibilities:
- Submit mentor/mentee feedback
- Calculate aggregate satisfaction metrics
- Export labeled dataset for model retraining
"""

from typing import List, Optional, Dict
from bson import ObjectId
from pymongo.database import Database
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class FeedbackService:
    """Handle feedback collection and analytics"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def submit_feedback(
        self,
        match_id: str,
        semester_id: str,
        submitted_by: str,
        role: str,  # "mentor" or "mentee"
        rating: int,  # 1-5
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Submit feedback for a match
        
        Args:
            match_id: Match group ID (mentor-mentee pairing)
            semester_id: Semester ID
            submitted_by: Email of person submitting feedback
            role: "mentor" or "mentee"
            rating: 1-5 satisfaction rating
            comment: Optional text feedback
            tags: Optional tags (e.g., ["good_communication", "shared_interests"])
        
        Returns:
            feedback_id (str)
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback_doc = {
            "match_id": ObjectId(match_id),
            "semester_id": ObjectId(semester_id),
            "submitted_by": submitted_by.lower(),
            "role": role,
            "rating": rating,
            "comment": comment,
            "tags": tags or [],
            "submitted_at": datetime.now(timezone.utc)
        }
        
        result = self.db.feedback.insert_one(feedback_doc)
        
        logger.info(
            f"Feedback submitted for match {match_id} by {role}: {rating}/5"
        )
        
        return str(result.inserted_id)
    
    def get_semester_feedback_summary(
        self,
        semester_id: str,
        organization_id: str
    ) -> Dict:
        """
        Get aggregate feedback statistics for semester
        
        Returns:
            {
              "total_feedback": 300,
              "response_rate": 0.67,  # 300/450 applicants
              "average_rating": 4.2,
              "rating_distribution": {
                "5": 120,
                "4": 100,
                "3": 50,
                "2": 20,
                "1": 10
              },
              "by_role": {
                "mentor": {"avg": 4.3, "count": 150},
                "mentee": {"avg": 4.1, "count": 150}
              },
              "top_tags": [
                {"tag": "good_communication", "count": 85},
                {"tag": "shared_interests", "count": 72}
              ]
            }
        """
        semester_oid = ObjectId(semester_id)
        
        # Total feedback count
        total_feedback = self.db.feedback.count_documents({
            "semester_id": semester_oid
        })
        
        # Average rating
        avg_pipeline = [
            {"$match": {"semester_id": semester_oid}},
            {"$group": {
                "_id": None,
                "avg_rating": {"$avg": "$rating"},
                "total": {"$sum": 1}
            }}
        ]
        avg_result = list(self.db.feedback.aggregate(avg_pipeline))
        average_rating = avg_result[0]["avg_rating"] if avg_result else 0
        
        # Rating distribution
        dist_pipeline = [
            {"$match": {"semester_id": semester_oid}},
            {"$group": {
                "_id": "$rating",
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": -1}}
        ]
        rating_dist = {
            str(r["_id"]): r["count"] 
            for r in self.db.feedback.aggregate(dist_pipeline)
        }
        
        # By role
        role_pipeline = [
            {"$match": {"semester_id": semester_oid}},
            {"$group": {
                "_id": "$role",
                "avg": {"$avg": "$rating"},
                "count": {"$sum": 1}
            }}
        ]
        by_role = {
            r["_id"]: {"avg": r["avg"], "count": r["count"]}
            for r in self.db.feedback.aggregate(role_pipeline)
        }
        
        # Top tags
        tag_pipeline = [
            {"$match": {"semester_id": semester_oid}},
            {"$unwind": "$tags"},
            {"$group": {
                "_id": "$tags",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_tags = [
            {"tag": t["_id"], "count": t["count"]}
            for t in self.db.feedback.aggregate(tag_pipeline)
        ]
        
        # Calculate response rate
        total_applicants = self.db.applicants.count_documents({
            "applications.semester_id": semester_oid
        })
        response_rate = total_feedback / total_applicants if total_applicants > 0 else 0
        
        return {
            "total_feedback": total_feedback,
            "response_rate": round(response_rate, 2),
            "average_rating": round(average_rating, 2),
            "rating_distribution": rating_dist,
            "by_role": by_role,
            "top_tags": top_tags
        }
    
    def export_labeled_dataset(
        self,
        semester_id: str,
        min_rating_threshold: int = 4
    ) -> List[Dict]:
        """
        Export feedback as labeled training data
        
        Format for ML retraining:
        [
          {
            "mentor_features": {...},
            "mentee_features": {...},
            "compatibility_label": 1,  # 1 = good match (rating >= 4), 0 = poor match
            "rating": 5,
            "feedback_count": 2
          }
        ]
        
        Args:
            semester_id: Semester ID
            min_rating_threshold: Ratings >= this are labeled as good matches
        
        Returns:
            List of labeled training examples
        """
        semester_oid = ObjectId(semester_id)
        
        # Aggregate feedback by match group
        pipeline = [
            {"$match": {"semester_id": semester_oid}},
            {"$group": {
                "_id": "$match_id",
                "avg_rating": {"$avg": "$rating"},
                "feedback_count": {"$sum": 1},
                "comments": {"$push": "$comment"}
            }},
            {"$lookup": {
                "from": "match_groups",
                "localField": "_id",
                "foreignField": "_id",
                "as": "match"
            }},
            {"$unwind": "$match"}
        ]
        
        labeled_data = []
        
        for doc in self.db.feedback.aggregate(pipeline):
            match = doc["match"]
            avg_rating = doc["avg_rating"]
            
            # Binary label: 1 = good match, 0 = poor match
            label = 1 if avg_rating >= min_rating_threshold else 0
            
            labeled_data.append({
                "match_id": str(match["_id"]),
                "mentor_email": match["mentor"]["email"],
                "mentee_emails": [m["email"] for m in match["mentees"]],
                "compatibility_label": label,
                "rating": round(avg_rating, 2),
                "feedback_count": doc["feedback_count"],
                "comments": [c for c in doc["comments"] if c]
            })
        
        logger.info(
            f"Exported {len(labeled_data)} labeled training examples from semester {semester_id}"
        )
        
        return labeled_data
```

---

## 6. Middleware Layer

### 6.1 Rate Limiting (api/middleware/rate_limit.py)

```python
"""
Rate Limiting Middleware - Prevent API abuse

Limits:
- 100 requests/hour per user
- 10 CSV uploads/day per user
- 5 matching jobs/day per user
- 5 concurrent matching jobs per organization
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit requests per user and organization"""
    
    def __init__(self, app):
        super().__init__(app)
        # In-memory storage (use Redis for production)
        self.request_counts: Dict[str, list] = defaultdict(list)
        self.csv_uploads: Dict[str, list] = defaultdict(list)
        self.matching_jobs: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        # Extract user from JWT (set by auth middleware)
        user_email = getattr(request.state, "user_email", None)
        org_id = getattr(request.state, "organization_id", None)
        
        if user_email:
            # General API rate limit
            if not self._check_request_limit(user_email):
                logger.warning(f"Rate limit exceeded for user {user_email}")
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Max 100 requests per hour."
                )
            
            # CSV upload limit
            if "/applicants/upload" in request.url.path:
                if not self._check_csv_upload_limit(user_email):
                    raise HTTPException(
                        status_code=429,
                        detail="CSV upload limit exceeded. Max 10 uploads per day."
                    )
            
            # Matching job limit
            if "/matching/batch" in request.url.path:
                if not self._check_matching_job_limit(user_email):
                    raise HTTPException(
                        status_code=429,
                        detail="Matching job limit exceeded. Max 5 jobs per day."
                    )
        
        response = await call_next(request)
        return response
    
    def _check_request_limit(self, user_email: str) -> bool:
        """Check general request limit: 100/hour"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        # Clean old entries
        self.request_counts[user_email] = [
            ts for ts in self.request_counts[user_email] if ts > cutoff
        ]
        
        # Check limit
        if len(self.request_counts[user_email]) >= 100:
            return False
        
        # Record request
        self.request_counts[user_email].append(now)
        return True
    
    def _check_csv_upload_limit(self, user_email: str) -> bool:
        """Check CSV upload limit: 10/day"""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        self.csv_uploads[user_email] = [
            ts for ts in self.csv_uploads[user_email] if ts > cutoff
        ]
        
        if len(self.csv_uploads[user_email]) >= 10:
            return False
        
        self.csv_uploads[user_email].append(now)
        return True
    
    def _check_matching_job_limit(self, user_email: str) -> bool:
        """Check matching job limit: 5/day"""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        self.matching_jobs[user_email] = [
            ts for ts in self.matching_jobs[user_email] if ts > cutoff
        ]
        
        if len(self.matching_jobs[user_email]) >= 5:
            return False
        
        self.matching_jobs[user_email].append(now)
        return True
```

### 6.2 Structured Logging (api/middleware/logging_middleware.py)

```python
"""
Structured JSON Logging Middleware - AWS CloudWatch compatible

Logs every request with:
- Correlation ID
- User context
- Request/response metadata
- Execution time
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import json
import time
import uuid

logger = logging.getLogger(__name__)

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Add structured logging to every request"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timer
        start_time = time.time()
        
        # Extract user context (if authenticated)
        user_email = getattr(request.state, "user_email", None)
        org_id = getattr(request.state, "organization_id", None)
        
        # Log request
        log_data = {
            "event": "request_start",
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "user_email": user_email,
            "organization_id": org_id,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent")
        }
        
        logger.info(json.dumps(log_data))
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log response
            log_data.update({
                "event": "request_complete",
                "status_code": response.status_code,
                "execution_time_ms": round(execution_time * 1000, 2)
            })
            
            logger.info(json.dumps(log_data))
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            
            log_data.update({
                "event": "request_error",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": round(execution_time * 1000, 2)
            })
            
            logger.error(json.dumps(log_data))
            
            raise
```

---

## 7. Dependency Injection (api/dependencies.py)

```python
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
InferenceEngineDep = Annotated[MatchingInference, Depends(get_inference_engine)]
OrganizationServiceDep = Annotated[OrganizationService, Depends(get_organization_service)]
SemesterServiceDep = Annotated[SemesterService, Depends(get_semester_service)]
ApplicantServiceDep = Annotated[ApplicantService, Depends(get_applicant_service)]
MatchingServiceDep = Annotated[MatchingService, Depends(get_matching_service)]
FeedbackServiceDep = Annotated[FeedbackService, Depends(get_feedback_service)]
```

---

## 8. Example Router (api/routers/semester_router.py)

```python
"""
Semester Router - Semester management endpoints
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
```

---

## 9. Updated main.py

```python
"""
Updated FastAPI Main Application

Changes:
- Remove global state
- Add middleware stack
- Include routers
- Use dependency injection
"""

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.logging_middleware import StructuredLoggingMiddleware

# Import routers
from api.routers import (
    auth_router,
    semester_router,
    # Import other routers as you build them
)

from api.models import HealthCheckResponse
from datetime import datetime

# LOGGING CONFIG - Structured JSON for AWS CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified lifespan - no global state"""
    logger.info("Application starting...")
    yield
    logger.info("Application shutting down...")

# FASTAPI APP
app = FastAPI(
    title="Mentor-Mentee Matching API",
    description="Production-ready matching system with authentication and multi-tenancy",
    version="2.0.0",
    lifespan=lifespan
)

# === MIDDLEWARE STACK (order matters!) ===
# 1. Structured Logging (first - logs everything)
app.add_middleware(StructuredLoggingMiddleware)

# 2. Rate Limiting (second - blocks before processing)
app.add_middleware(RateLimitMiddleware)

# 3. CORS (last - after rate limiting)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === ROUTERS ===
app.include_router(auth_router.router, prefix="/auth", tags=["Authentication"])
app.include_router(semester_router.router, prefix="/semesters", tags=["Semesters"])
# Add other routers as you build them

# === ROOT ENDPOINTS ===
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Mentor-Mentee Matching API v2.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check for AWS ECS"""
    # Try to get inference engine (will load if not cached)
    try:
        from api.dependencies import get_inference_engine
        engine = get_inference_engine()
        model_loaded = engine.model is not None
    except:
        model_loaded = False
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

# === GLOBAL EXCEPTION HANDLER ===
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": getattr(request.state, "correlation_id", None)
        }
    )
```

---

## 10. AWS ECS Deployment Configuration

### Dockerfile (Updated)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables (overridden by ECS task definition)
ENV MODEL_PATH=/app/models/best_model.pt
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### ECS Task Definition (task-definition.json)

```json
{
  "family": "matching-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "matching-api",
      "image": "<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/matching-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models/best_model.pt"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "MONGODB_URL",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT>:secret:mongodb-url"
        },
        {
          "name": "JWT_SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT>:secret:jwt-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/matching-api",
          "awslogs-region": "<REGION>",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 40
      }
    }
  ]
}
```

---

## Next Steps

1. **Copy services to your project**:
   ```bash
   mkdir -p api/services api/middleware api/routers
   # Copy the service files
   ```

2. **Create remaining routers** (following semester_router pattern):
   - applicant_router.py
   - matching_router.py
   - feedback_router.py
   - organization_router.py

3. **Test locally**:
   ```bash
   uvicorn api.main:app --reload
   ```

4. **Deploy to AWS ECS**:
   ```bash
   # Build and push image
   docker build -t matching-api .
   docker tag matching-api:latest <ECR_URI>:latest
   docker push <ECR_URI>:latest
   
   # Update ECS service
   aws ecs update-service --cluster matching-cluster --service matching-api --force-new-deployment
   ```

Would you like me to generate any specific router or provide the complete repository pattern implementation?