"""
Docstring for api.routers.feedback_router
Feedback Router - Feedback collection and analytics endpoints

Endpoints:
POST   /feedback/                          # Submit feedback
GET    /feedback/semester/{id}/summary     # Get summary
GET    /feedback/match/{id}                # Get match feedback
GET    /feedback/export/{semester_id}      # Export dataset
GET    /feedback/my-feedback               # My feedback
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging

from api.auth import get_current_user, UserInDB
from api.dependencies import FeedbackServiceDep, SemesterServiceDep, DatabaseDep

router = APIRouter()
logger = logging.getLogger(__name__)


# PYDANTIC MODELS

class FeedbackSubmitRequest(BaseModel):
    """Request model for submitting feedback"""
    match_id: str = Field(..., description="Match group ID")
    semester_id: str = Field(..., description="Semester ID")
    rating: int = Field(..., ge=1, le=5, description="Satisfaction rating (1-5)")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional text feedback")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    feedback_id: str
    match_id: str
    semester_id: str
    submitted_by: str
    role: str
    rating: int
    comment: Optional[str]
    tags: List[str]
    submitted_at: str


class FeedbackSummaryResponse(BaseModel):
    """Response model for semester feedback summary"""
    total_feedback: int
    response_rate: float
    average_rating: float
    rating_distribution: dict
    by_role: dict
    top_tags: List[dict]


class LabeledDataPoint(BaseModel):
    """Response model for labeled training data"""
    match_id: str
    mentor_email: str
    mentee_emails: List[str]
    compatibility_label: int  # 1 = good, 0 = poor
    rating: float
    feedback_count: int
    comments: List[str]


class LabeledDatasetResponse(BaseModel):
    """Response model for labeled dataset export"""
    semester_id: str
    total_examples: int
    positive_examples: int  # label = 1
    negative_examples: int  # label = 0
    data: List[LabeledDataPoint]


class MatchFeedbackResponse(BaseModel):
    """Response model for match-specific feedback"""
    match_id: str
    total_feedback: int
    average_rating: Optional[float]
    feedback_items: List[FeedbackResponse]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback",
    description="Submit satisfaction feedback for a match"
)
async def submit_feedback(
    request: FeedbackSubmitRequest,
    feedback_service: FeedbackServiceDep,
    semester_service: SemesterServiceDep,
    db: DatabaseDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Submit feedback for a match
    
    Requirements:
    - User must be part of the match (mentor or mentee)
    - Rating must be between 1-5
    - Can submit multiple feedback entries (e.g., mentor + mentee)
    
    Tags (optional):
    - good_communication
    - shared_interests
    - helpful_mentor
    - engaged_mentee
    - cultural_connection
    - academic_support
    - poor_communication
    - mismatched_interests
    - (custom tags allowed)
    """
    # Verify semester access
    try:
        semester_service.get_semester(request.semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Verify match exists
    from bson import ObjectId
    match = db.match_groups.find_one({"_id": ObjectId(request.match_id)})
    
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found"
        )
    
    # Determine role (mentor or mentee)
    user_email = user.email.lower()
    role = None
    
    if match["mentor"]["email"].lower() == user_email:
        role = "mentor"
    elif any(m["email"].lower() == user_email for m in match["mentees"]):
        role = "mentee"
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not part of this match"
        )
    
    # Submit feedback
    try:
        feedback_id = feedback_service.submit_feedback(
            match_id=request.match_id,
            semester_id=request.semester_id,
            submitted_by=user.email,
            role=role,
            rating=request.rating,
            comment=request.comment,
            tags=request.tags or []
        )
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            match_id=request.match_id,
            semester_id=request.semester_id,
            submitted_by=user.email,
            role=role,
            rating=request.rating,
            comment=request.comment,
            tags=request.tags or [],
            submitted_at=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/semester/{semester_id}/summary",
    response_model=FeedbackSummaryResponse,
    summary="Get feedback summary",
    description="Get aggregate feedback statistics for semester"
)
async def get_semester_feedback_summary(
    semester_id: str,
    feedback_service: FeedbackServiceDep,
    semester_service: SemesterServiceDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Get feedback summary for semester
    
    Returns:
    - Total feedback count
    - Response rate (feedback / total applicants)
    - Average rating
    - Rating distribution (1-5 stars)
    - Breakdown by role (mentor vs mentee)
    - Top 10 most common tags
    """
    # Verify semester access
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    summary = feedback_service.get_semester_feedback_summary(
        semester_id, user.organization_id
    )
    
    return FeedbackSummaryResponse(**summary)


@router.get(
    "/match/{match_id}",
    response_model=MatchFeedbackResponse,
    summary="Get feedback for match",
    description="Get all feedback for a specific match"
)
async def get_match_feedback(
    match_id: str,
    semester_service: SemesterServiceDep,
    db: DatabaseDep,
    user: UserInDB = Depends(get_current_user)
):
    """
    Get feedback for specific match
    
    Returns all feedback entries for a match group.
    """
    from bson import ObjectId
    
    # Get match to verify it exists and get semester
    match = db.match_groups.find_one({"_id": ObjectId(match_id)})
    
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Match not found"
        )
    
    # Verify semester access
    semester_id = str(match["semester_id"])
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    
    # Get feedback
    feedback_items = list(db.feedback.find({"match_id": ObjectId(match_id)}))
    
    feedback_responses = [
        FeedbackResponse(
            feedback_id=str(f["_id"]),
            match_id=match_id,
            semester_id=str(f["semester_id"]),
            submitted_by=f["submitted_by"],
            role=f["role"],
            rating=f["rating"],
            comment=f.get("comment"),
            tags=f.get("tags", []),
            submitted_at=f["submitted_at"].isoformat()
        )
        for f in feedback_items
    ]
    
    avg_rating = None
    if feedback_responses:
        avg_rating = sum(f.rating for f in feedback_responses) / len(feedback_responses)
    
    return MatchFeedbackResponse(
        match_id=match_id,
        total_feedback=len(feedback_responses),
        average_rating=avg_rating,
        feedback_items=feedback_responses
    )


@router.get(
    "/export/{semester_id}",
    response_model=LabeledDatasetResponse,
    summary="Export labeled dataset",
    description="Export feedback as labeled training data for ML retraining"
)
async def export_labeled_dataset(
    semester_id: str,
    feedback_service: FeedbackServiceDep,
    semester_service: SemesterServiceDep,
    min_rating_threshold: int = Query(4, ge=1, le=5, description="Min rating for positive label"),
    user: UserInDB = Depends(get_current_user)
):
    """
    Export labeled dataset for ML retraining
    
    Aggregates feedback by match and creates labeled training examples:
    - Label 1 (good match): average rating >= threshold
    - Label 0 (poor match): average rating < threshold
    
    Use this data to retrain the two-tower model with human feedback.
    
    Query parameters:
    - min_rating_threshold: Minimum rating for positive label (default 4)
    """
    # Verify semester access
    try:
        semester_service.get_semester(semester_id, user.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
    # Export dataset
    labeled_data = feedback_service.export_labeled_dataset(
        semester_id, min_rating_threshold
    )
    
    # Convert to response models
    data_points = [
        LabeledDataPoint(**item)
        for item in labeled_data
    ]
    
    # Calculate positive/negative counts
    positive = sum(1 for d in data_points if d.compatibility_label == 1)
    negative = len(data_points) - positive
    
    return LabeledDatasetResponse(
        semester_id=semester_id,
        total_examples=len(data_points),
        positive_examples=positive,
        negative_examples=negative,
        data=data_points
    )


@router.get(
    "/tags",
    summary="Get available feedback tags",
    description="Get list of suggested feedback tags"
)
async def get_feedback_tags(
    user: UserInDB = Depends(get_current_user)
):
    """
    Get suggested feedback tags
    
    Returns list of common tags users can select from.
    Users can also create custom tags.
    """
    return {
        "positive_tags": [
            "good_communication",
            "shared_interests",
            "helpful_mentor",
            "engaged_mentee",
            "cultural_connection",
            "academic_support",
            "career_guidance",
            "friendly",
            "responsive",
            "motivating"
        ],
        "negative_tags": [
            "poor_communication",
            "mismatched_interests",
            "unresponsive",
            "different_goals",
            "scheduling_conflicts",
            "lack_of_engagement",
            "unhelpful",
            "incompatible_personalities"
        ],
        "note": "Users can also create custom tags"
    }


@router.get(
    "/my-feedback",
    response_model=List[FeedbackResponse],
    summary="Get my submitted feedback",
    description="Get all feedback submitted by current user"
)
async def get_my_feedback(
    db: DatabaseDep,
    semester_id: Optional[str] = Query(None, description="Filter by semester"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    user: UserInDB = Depends(get_current_user)
):
    """
    Get feedback submitted by current user
    
    Optionally filter by semester.
    """
    from bson import ObjectId
    
    query = {"submitted_by": user.email.lower()}
    
    if semester_id:
        query["semester_id"] = ObjectId(semester_id)
    
    feedback_items = list(
        db.feedback
        .find(query)
        .sort("submitted_at", -1)
        .skip(skip)
        .limit(limit)
    )
    
    return [
        FeedbackResponse(
            feedback_id=str(f["_id"]),
            match_id=str(f["match_id"]),
            semester_id=str(f["semester_id"]),
            submitted_by=f["submitted_by"],
            role=f["role"],
            rating=f["rating"],
            comment=f.get("comment"),
            tags=f.get("tags", []),
            submitted_at=f["submitted_at"].isoformat()
        )
        for f in feedback_items
    ]