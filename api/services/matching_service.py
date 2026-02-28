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
    PENDING         = "pending"
    PREPROCESSING   = "preprocessing"
    EMBEDDING       = "embedding"
    TRAINING        = "training"
    MATCHING        = "matching"
    COMPLETED       = "completed"
    FAILED          = "failed"


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
            "job_id"            : job_id,
            "semester_id"       : ObjectId(semester_id),
            "organization_id"   : ObjectId(organization_id),
            "status"            : JobStatus.PENDING,
            "triggered_by"      : triggered_by,
            "config"            : config,
            "created_at"        : datetime.now(timezone.utc),
            "updated_at"        : datetime.now(timezone.utc),
            "started_at"        : None,
            "completed_at"      : None,
            "error_message"     : None,
            "progress"          : {
                                    "current_step": "pending",
                                    "total_steps": 5,
                                    "completed_steps": 0
                                },
            "results"           : {
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
            "job_id"        : job["job_id"],
            "status"        : job["status"],
            "progress"      : job["progress"],
            "created_at"    : job["created_at"].isoformat(),
            "started_at"    : job["started_at"].isoformat() if job.get("started_at") else None,
            "completed_at"  : job["completed_at"].isoformat() if job.get("completed_at") else None,
            "elapsed_seconds": elapsed,
            "error_message" : job.get("error_message"),
            "results"       : job.get("results")
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
                "semester_id"   : semester_oid,
                "job_id"        : job_id,
                "group_id"      : group["group_id"],
                "mentor"        : group["mentor"],
                "mentees"       : group["mentees"],
                "compatibility_score"   : group["compatibility_score"],
                "individual_scores"     : group["individual_scores"],
                "created_at"            : datetime.now(timezone.utc)
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