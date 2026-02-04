"""
Semester Service - Business logic for semester lifecycle management

Responsibilities:
- Create/Read/Update/Delete semesters
- Validate semester constraints (dates, quotas)
- Track semester statistics
- Enforce organization-scoped access
"""

from typing import List, Optional, Dict
from bson import ObjectId
from pymongo.database import Database
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class SemesterService:
    """
    Handle semester-level operations
    
    Semester lifecycle:
    1. draft -> Coordinator creates semester
    2. active -> Accepting applicant uploads
    3. matching -> Matching job running
    4. completed -> Matches finalized
    5. archived -> Historical record
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_semester(
        self,
        organization_id: str,
        name: str,
        start_date: datetime,
        end_date: datetime,
        mentor_quota: int,
        mentee_quota: int,
        created_by: str  # User email
    ) -> str:
        """
        Create new semester
        
        Args:
            organization_id: Organization ID
            name: Semester name (e.g., "Fall 2024")
            start_date: Semester start date
            end_date: Semester end date
            mentor_quota: Expected number of mentors
            mentee_quota: Expected number of mentees
            created_by: Email of user creating semester
        
        Returns:
            semester_id (str): MongoDB ObjectId as string
        
        Raises:
            ValueError: If validation fails
        """
        # Validate dates
        if end_date <= start_date:
            raise ValueError("End date must be after start date")
        
        # Check quota ratio warning
        if mentee_quota < mentor_quota * 2:
            logger.warning(
                f"Mentee quota ({mentee_quota}) is less than 2x mentor quota ({mentor_quota}). "
                f"Matching may not fill all groups."
            )
        
        # Check for duplicate semester name in same org
        existing = self.db.semesters.find_one({
            "organization_id": ObjectId(organization_id),
            "name": name
        })
        
        if existing:
            raise ValueError(f"Semester '{name}' already exists in this organization")
        
        # Create semester document
        semester_doc = {
            "organization_id": ObjectId(organization_id),
            "name": name,
            "start_date": start_date,
            "end_date": end_date,
            "quotas": {
                "mentors": mentor_quota,
                "mentees": mentee_quota
            },
            "status": "draft",  # draft, active, matching, completed, archived
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "matching_job_id": None,
            "matching_completed_at": None,
            "statistics": {
                "total_applicants": 0,
                "mentors": 0,
                "mentees": 0,
                "matched_groups": 0,
                "average_compatibility": None
            }
        }
        
        result = self.db.semesters.insert_one(semester_doc)
        
        logger.info(f"Created semester '{name}' with ID {result.inserted_id} in org {organization_id}")
        
        return str(result.inserted_id)
    
    def get_semester(
        self,
        semester_id: str,
        organization_id: str
    ) -> Dict:
        """
        Get semester details with organization isolation
        
        Args:
            semester_id: Semester ID
            organization_id: User's organization ID (for access control)
        
        Returns:
            Semester document
        
        Raises:
            ValueError: If semester not found or access denied
        """
        semester = self.db.semesters.find_one({
            "_id": ObjectId(semester_id),
            "organization_id": ObjectId(organization_id)  # Strict isolation
        })
        
        if not semester:
            raise ValueError(
                f"Semester {semester_id} not found or access denied for org {organization_id}"
            )
        
        # Convert ObjectIds to strings
        semester["_id"] = str(semester["_id"])
        semester["organization_id"] = str(semester["organization_id"])
        
        return semester
    
    def list_semesters(
        self,
        organization_id: str,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[Dict]:
        """
        List semesters for organization
        
        Args:
            organization_id: Organization ID
            status: Filter by status (optional)
            skip: Pagination offset
            limit: Max results
        
        Returns:
            List of semester documents
        """
        query = {"organization_id": ObjectId(organization_id)}
        
        if status:
            query["status"] = status
        
        semesters = list(
            self.db.semesters
            .find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        
        # Convert ObjectIds
        for sem in semesters:
            sem["_id"] = str(sem["_id"])
            sem["organization_id"] = str(sem["organization_id"])
        
        return semesters
    
    def update_semester(
        self,
        semester_id: str,
        organization_id: str,
        updates: Dict
    ) -> bool:
        """
        Update semester details
        
        Args:
            semester_id: Semester ID
            organization_id: User's organization ID
            updates: Dict of fields to update
        
        Returns:
            True if updated, False if no changes
        
        Raises:
            ValueError: If semester locked (status = completed/archived)
        """
        # Check if semester exists and user has access
        semester = self.get_semester(semester_id, organization_id)
        
        # Prevent updates to locked semesters
        if semester["status"] in ["completed", "archived"]:
            raise ValueError(
                f"Cannot update semester in '{semester['status']}' status. "
                f"Unlock or create new semester."
            )
        
        # Protected fields
        protected_fields = {
            "_id", "organization_id", "created_at", "created_by",
            "matching_job_id", "matching_completed_at"
        }
        safe_updates = {k: v for k, v in updates.items() if k not in protected_fields}
        
        if not safe_updates:
            return False
        
        safe_updates["updated_at"] = datetime.now(timezone.utc)
        
        result = self.db.semesters.update_one(
            {
                "_id": ObjectId(semester_id),
                "organization_id": ObjectId(organization_id)
            },
            {"$set": safe_updates}
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated semester {semester_id}: {list(safe_updates.keys())}")
            return True
        
        return False
    
    def update_semester_status(
        self,
        semester_id: str,
        organization_id: str,
        new_status: str
    ) -> bool:
        """
        Update semester status
        
        Valid transitions:
        - draft -> active
        - active -> matching (when matching job starts)
        - matching -> completed (when matching job completes)
        - completed -> archived
        
        Args:
            semester_id: Semester ID
            organization_id: Organization ID
            new_status: New status value
        
        Returns:
            True if updated
        
        Raises:
            ValueError: If invalid status transition
        """
        semester = self.get_semester(semester_id, organization_id)
        current_status = semester["status"]
        
        # Validate transition
        valid_transitions = {
            "draft": ["active"],
            "active": ["matching"],
            "matching": ["completed", "active"],  # Can revert if matching fails
            "completed": ["archived"],
            "archived": []  # Terminal state
        }
        
        if new_status not in valid_transitions.get(current_status, []):
            raise ValueError(
                f"Invalid status transition: {current_status} -> {new_status}"
            )
        
        result = self.db.semesters.update_one(
            {
                "_id": ObjectId(semester_id),
                "organization_id": ObjectId(organization_id)
            },
            {
                "$set": {
                    "status": new_status,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        logger.info(f"Semester {semester_id} status: {current_status} -> {new_status}")
        
        return result.modified_count > 0
    
    def delete_semester(
        self,
        semester_id: str,
        organization_id: str,
        soft_delete: bool = True
    ) -> bool:
        """
        Delete semester (soft delete by default)
        
        Args:
            semester_id: Semester ID
            organization_id: Organization ID
            soft_delete: If True, mark as archived; if False, permanently delete
        
        Returns:
            True if deleted
        
        Raises:
            ValueError: If semester has associated match results
        """
        # Check if semester has match results
        match_count = self.db.match_groups.count_documents({
            "semester_id": ObjectId(semester_id)
        })
        
        if match_count > 0 and not soft_delete:
            raise ValueError(
                f"Cannot permanently delete semester with {match_count} match results. "
                f"Use soft delete (archive) instead."
            )
        
        if soft_delete:
            # Soft delete: Mark as archived
            result = self.db.semesters.update_one(
                {
                    "_id": ObjectId(semester_id),
                    "organization_id": ObjectId(organization_id)
                },
                {
                    "$set": {
                        "status": "archived",
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            logger.info(f"Soft deleted (archived) semester {semester_id}")
            return result.modified_count > 0
        else:
            # Hard delete: Remove from database
            result = self.db.semesters.delete_one({
                "_id": ObjectId(semester_id),
                "organization_id": ObjectId(organization_id)
            })
            logger.warning(f"Hard deleted semester {semester_id}")
            return result.deleted_count > 0
    
    def get_semester_stats(self, semester_id: str, organization_id: str) -> Dict:
        """
        Get statistics for a semester
        
        Returns:
            {
              "total_applicants": 450,
              "mentors": 150,
              "mentees": 300,
              "matched_groups": 150,
              "unmatched_mentees": 0,
              "average_compatibility": 0.85,
              "quota_fulfillment": {
                "mentors": 1.0,   # 150/150
                "mentees": 1.0    # 300/300
              }
            }
        """
        semester = self.get_semester(semester_id, organization_id)
        semester_oid = ObjectId(semester_id)
        
        # Count applicants by role using aggregation
        role_counts = list(self.db.applicants.aggregate([
            {"$match": {"applications.semester_id": semester_oid}},
            {"$unwind": "$applications"},
            {"$match": {"applications.semester_id": semester_oid}},
            {"$group": {
                "_id": "$applications.role",
                "count": {"$sum": 1}
            }}
        ]))
        
        mentors = next((r["count"] for r in role_counts if r["_id"] == "mentor"), 0)
        mentees = next((r["count"] for r in role_counts if r["_id"] == "mentee"), 0)
        
        # Count matched groups and get average compatibility
        match_pipeline = [
            {"$match": {"semester_id": semester_oid}},
            {"$group": {
                "_id": None,
                "count": {"$sum": 1},
                "avg_compatibility": {"$avg": "$compatibility_score"}
            }}
        ]
        
        match_stats = list(self.db.match_groups.aggregate(match_pipeline))
        matched_groups = match_stats[0]["count"] if match_stats else 0
        avg_compatibility = match_stats[0]["avg_compatibility"] if match_stats else None
        
        # Calculate quota fulfillment
        mentor_quota = semester["quotas"]["mentors"]
        mentee_quota = semester["quotas"]["mentees"]
        
        quota_fulfillment = {
            "mentors": mentors / mentor_quota if mentor_quota > 0 else 0,
            "mentees": mentees / mentee_quota if mentee_quota > 0 else 0
        }
        
        stats = {
            "total_applicants": mentors + mentees,
            "mentors": mentors,
            "mentees": mentees,
            "matched_groups": matched_groups,
            "unmatched_mentees": max(0, mentees - (matched_groups * 2)),
            "average_compatibility": avg_compatibility,
            "quota_fulfillment": quota_fulfillment
        }
        
        # Update semester statistics in database
        self.db.semesters.update_one(
            {"_id": semester_oid},
            {"$set": {"statistics": stats, "updated_at": datetime.now(timezone.utc)}}
        )
        
        return stats