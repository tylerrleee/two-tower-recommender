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
            "match_id"      : ObjectId(match_id),
            "semester_id"   : ObjectId(semester_id),
            "submitted_by"  : submitted_by.lower(),
            "role"          : role,
            "rating"        : rating,
            "comment"       : comment,
            "tags"          : tags or [],
            "submitted_at"  : datetime.now(timezone.utc)
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