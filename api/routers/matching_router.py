"""
Docstring for api.routers.matching_router
Matching Router - Matching job management endpoints


Endpoints:
POST   /matching/batch-async              # Trigger async job
GET    /matching/jobs/{job_id}/status     # Poll job status
GET    /matching/jobs                     # List jobs
GET    /matching/results/{semester_id}    # Get results
POST   /matching/batch                    # Sync matching (test)
"""


from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timezone
import logging

from api.auth import get_current_user, UserInDB, PermissionChecker
from api.dependencies import (
    MatchingServiceDep, 
    SemesterServiceDep, 
    InferenceEngineDep,
    DatabaseDep
)
from api.services.matching_service import JobStatus
from database.adapter import DataAdapter

router = APIRouter()
logger = logging.getLogger(__name__)


# PYDANTIC MODELS

class BatchMatchingRequest(BaseModel):
    """Request model for batch matching"""
    semester_id: str = Field(..., description="Semester ID to match")
    use_faiss: bool = Field(default=False, description="Use FAISS for faster matching")
    top_k: int = Field(default=10, ge=1, le=50, description="Top K candidates to consider")
    save_results: bool = Field(default=True, description="Save results to database")


class MatchingJobResponse(BaseModel):
    """Response model for matching job"""
    job_id: str
    semester_id: str
    status: str
    created_at: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: dict
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    elapsed_seconds: Optional[float]
    error_message: Optional[str]
    results: Optional[dict]


class MatchGroupResponse(BaseModel):
    """Response model for match group"""
    group_id: int
    mentor: dict
    mentees: List[dict]
    compatibility_score: float
    individual_scores: List[float]
    created_at: str


class MatchResultsResponse(BaseModel):
    """Response model for match results"""
    semester_id: str
    total_groups: int
    average_compatibility: Optional[float]
    groups: List[MatchGroupResponse]


# ============================================================================
# BACKGROUND JOB FUNCTION
# ============================================================================

async def run_matching_job_async(
    job_id: str,
    semester_id: str,
    config: dict,
    matching_service,
    inference_engine,
    db
):
    """
    Run matching job asynchronously
    
    Steps:
    1. Load applicant data
    2. Preprocess and generate features
    3. Run matching algorithm
    4. Save results
    """
    try:
        # Step 1: Preprocessing
        logger.info(f"Job {job_id}: Starting preprocessing")
        matching_service.update_job_status(
            job_id, 
            JobStatus.PREPROCESSING,
            progress={"current_step": "loading_data", "completed_steps": 1, "total_steps": 5}
        )
        
        adapter = DataAdapter(db)
        df_mentors, df_mentees = adapter.fetch_training_data(
            semester_id=semester_id,
            include_embeddings=True
        )
        
        logger.info(f"Job {job_id}: Loaded {len(df_mentors)} mentors, {len(df_mentees)} mentees")
        
        # Validate sufficient applicants
        if len(df_mentors) == 0:
            raise ValueError("No mentors found in semester")
        if len(df_mentees) == 0:
            raise ValueError("No mentees found in semester")
        
        # Step 2: Feature engineering
        logger.info(f"Job {job_id}: Starting feature engineering")
        matching_service.update_job_status(
            job_id,
            JobStatus.EMBEDDING,
            progress={"current_step": "feature_engineering", "completed_steps": 2, "total_steps": 5}
        )
        
        import pandas as pd
        df_all = pd.concat([df_mentors, df_mentees], ignore_index=True)
        
        # Step 3: Matching
        logger.info(f"Job {job_id}: Starting matching algorithm")
        matching_service.update_job_status(
            job_id,
            JobStatus.MATCHING,
            progress={"current_step": "hungarian_algorithm", "completed_steps": 4, "total_steps": 5}
        )
        
        results = inference_engine.match(
            df=df_all,
            use_faiss=config.get("use_faiss", False),
            top_k=config.get("top_k", 10)
        )
        
        logger.info(f"Job {job_id}: Matching completed, {len(results['groups'])} groups created")
        
        # Step 4: Save results
        if config.get("save_results", True):
            matching_service.save_match_results(
                job_id=job_id,
                semester_id=semester_id,
                groups=results["groups"]
            )
        
        # Step 5: Complete
        matching_service.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress={"current_step": "done", "completed_steps": 5, "total_steps": 5}
        )
        
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        matching_service.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(e)
        )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/batch-async",
    response_model=MatchingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(PermissionChecker("can_trigger_matching"))],
    summary="Trigger async matching job",
    description="Start asynchronous matching job for a semester"
)
async def trigger_async_matching(
    request: BatchMatchingRequest,
    background_tasks: BackgroundTasks,
    matching_service: MatchingServiceDep,
    semester_service: SemesterServiceDep,
    inference_engine: InferenceEngineDep,
    db: DatabaseDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Trigger asynchronous matching job
    
    Creates a matching job and processes it in the background.
    Poll /matching/jobs/{job_id}/status to check progress.
    
    The job will:
    1. Load applicant data from semester
    2. Generate embeddings (if not cached)
    3. Run matching algorithm
    4. Save results to database
    
    Returns job_id for status tracking.
    """
    # Verify semester exists and user has access
    try:
        semester = semester_service.get_semester(request.semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Verify semester is in correct status
    if semester["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot run matching on semester in '{semester['status']}' status. Must be 'active'."
        )
    
    # Check organization quota for concurrent jobs
    from api.dependencies import get_organization_service
    org_service = get_organization_service(db)
    
    if not org_service.check_quota(user.organization_id, "matching_jobs", 1):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Maximum concurrent matching jobs reached. Please wait for existing jobs to complete."
        )
    
    # Create matching job
    config = {
        "use_faiss": request.use_faiss,
        "top_k": request.top_k,
        "save_results": request.save_results
    }
    
    job_id = matching_service.create_matching_job(
        semester_id=request.semester_id,
        organization_id=user.organization_id,
        triggered_by=user.email,
        config=config
    )
    
    # Update semester status
    semester_service.update_semester_status(
        request.semester_id,
        user.organization_id,
        "matching"
    )
    
    # Add background task
    background_tasks.add_task(
        run_matching_job_async,
        job_id=job_id,
        semester_id=request.semester_id,
        config=config,
        matching_service=matching_service,
        inference_engine=inference_engine,
        db=db
    )
    
    return MatchingJobResponse(
        job_id=job_id,
        semester_id=request.semester_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc).isoformat(),
        message="Matching job created. Poll /matching/jobs/{job_id}/status for updates."
    )


@router.get(
    "/jobs/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Get status and progress of matching job"
)
async def get_job_status(
    job_id: str,
    matching_service: MatchingServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Get matching job status
    
    Returns current status, progress, and results (if completed).
    Poll this endpoint to track job progress.
    
    Status flow:
    pending → preprocessing → embedding → matching → completed (or failed)
    """
    try:
        status_data = matching_service.get_job_status(job_id)
        
        # Note: In production, add authorization check to ensure user can access this job
        # For now, any authenticated user can check any job status
        
        return JobStatusResponse(**status_data)
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get(
    "/jobs",
    response_model=List[JobStatusResponse],
    summary="List matching jobs",
    description="List matching jobs for organization"
)
async def list_matching_jobs(
    matching_service: MatchingServiceDep,
    db: DatabaseDep,
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    semester_id: Optional[str] = Query(None, description="Filter by semester"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    user: UserInDB = Depends(get_current_user)
):
    """
    List matching jobs
    
    Returns list of matching jobs for user's organization.
    Can filter by status or semester.
    """
    from bson import ObjectId
    
    # Build query
    query = {"organization_id": ObjectId(user.organization_id)}
    
    if status_filter:
        if status_filter not in [s.value for s in JobStatus]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.value for s in JobStatus]}"
            )
        query["status"] = status_filter
    
    if semester_id:
        query["semester_id"] = ObjectId(semester_id)
    
    # Query jobs
    jobs = list(
        db.matching_jobs
        .find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    
    # Convert to response models
    job_responses = []
    for job in jobs:
        elapsed = None
        if job.get("started_at"):
            end_time = job.get("completed_at") or datetime.now(timezone.utc)
            elapsed = (end_time - job["started_at"]).total_seconds()
        
        job_responses.append(JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=job["progress"],
            created_at=job["created_at"].isoformat(),
            started_at=job["started_at"].isoformat() if job.get("started_at") else None,
            completed_at=job["completed_at"].isoformat() if job.get("completed_at") else None,
            elapsed_seconds=elapsed,
            error_message=job.get("error_message"),
            results=job.get("results")
        ))
    
    return job_responses


@router.get(
    "/results/{semester_id}",
    response_model=MatchResultsResponse,
    summary="Get match results",
    description="Get matching results for a semester"
)
async def get_match_results(
    semester_id: str,
    semester_service: SemesterServiceDep,
    db: DatabaseDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Get match results for semester
    
    Returns all match groups with mentor-mentee pairings and compatibility scores.
    """
    # Verify semester access
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    from bson import ObjectId
    
    # Get match groups
    groups = list(
        db.match_groups
        .find({"semester_id": ObjectId(semester_id)})
        .sort("group_id", 1)
    )
    
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No match results found for this semester"
        )
    
    # Convert to response
    group_responses = [
        MatchGroupResponse(
            group_id=g["group_id"],
            mentor=g["mentor"],
            mentees=g["mentees"],
            compatibility_score=g["compatibility_score"],
            individual_scores=g["individual_scores"],
            created_at=g["created_at"].isoformat()
        )
        for g in groups
    ]
    
    # Calculate average
    avg_compatibility = sum(g.compatibility_score for g in group_responses) / len(group_responses)
    
    return MatchResultsResponse(
        semester_id=semester_id,
        total_groups=len(group_responses),
        average_compatibility=avg_compatibility,
        groups=group_responses
    )


@router.post(
    "/batch",
    dependencies=[Depends(PermissionChecker("can_trigger_matching"))],
    summary="Synchronous matching (testing only)",
    description="Run matching synchronously. Use /batch-async for production."
)
async def trigger_sync_matching(
    request: BatchMatchingRequest,
    matching_service: MatchingServiceDep,
    semester_service: SemesterServiceDep,
    inference_engine: InferenceEngineDep,
    db: DatabaseDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Synchronous matching (for testing)
    
    WARNING: This endpoint blocks until matching completes.
    For production, use /batch-async instead.
    
    Useful for:
    - Testing with small datasets
    - Development/debugging
    - Quick matching runs (<100 applicants)
    """
    # Verify semester
    try:
        semester_service.get_semester(request.semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    try:
        # Load data
        adapter = DataAdapter(db)
        df_mentors, df_mentees = adapter.fetch_training_data(
            semester_id=request.semester_id,
            include_embeddings=True
        )
        
        import pandas as pd
        df_all = pd.concat([df_mentors, df_mentees], ignore_index=True)
        
        # Run matching
        results = inference_engine.match(
            df=df_all,
            use_faiss=request.use_faiss,
            top_k=request.top_k
        )
        
        # Save results if requested
        if request.save_results:
            job_id = matching_service.create_matching_job(
                semester_id=request.semester_id,
                organization_id=user.organization_id,
                triggered_by=user.email,
                config={"sync": True}
            )
            
            matching_service.save_match_results(
                job_id=job_id,
                semester_id=request.semester_id,
                groups=results["groups"]
            )
            
            matching_service.update_job_status(job_id, JobStatus.COMPLETED)
        
        return {
            "message": "Matching completed",
            "total_groups": len(results["groups"]),
            "average_compatibility": sum(g["compatibility_score"] for g in results["groups"]) / len(results["groups"]),
            "groups": results["groups"][:5]  # Return first 5 groups as preview
        }
        
    except Exception as e:
        logger.error(f"Sync matching failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}"
        )