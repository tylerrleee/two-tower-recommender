"""
Applicant Service - Business logic for applicant management

Responsibilities:
- CSV upload parsing and validation
- Applicant CRUD operations
- Data quality checks
- Duplicate detection
"""

from typing import List, Optional, Dict, Tuple
from bson import ObjectId
from pymongo.database import Database
from datetime import datetime, timezone
import pandas as pd
import logging
import re

logger = logging.getLogger(__name__)

class ApplicantService:
    """
    Handle applicant-level operations
    
    Validation rules:
    - Email uniqueness per semester
    - Required fields based on role
    - Survey response completeness
    - Organization quota enforcement
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def validate_csv(
        self,
        df: pd.DataFrame,
        semester_id: str,
        organization_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate uploaded CSV for applicant data
        
        Args:
            df: Pandas DataFrame from CSV
            semester_id: Target semester ID
            organization_id: Organization ID for quota check
        
        Returns:
            (is_valid, errors_list)
        
        Validation checks:
        1. Required columns present
        2. Email format validity
        3. Email domain restriction (optional)
        4. Duplicate emails within CSV
        5. Duplicate emails with existing semester applicants
        6. Organization quota enforcement
        7. Role distribution (mentor/mentee ratio)
        """
        errors = []
        
        # 1. Check required columns
        required_cols = {'role', 'name', 'ufl_email', 'major', 'year'}
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # 2. Validate email format
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        invalid_emails = df[~df['ufl_email'].str.match(email_pattern, na=False)]
        
        if not invalid_emails.empty:
            errors.append(
                f"Invalid email format for {len(invalid_emails)} rows: "
                f"{invalid_emails['ufl_email'].head(3).tolist()}"
            )
        
        # 3. Email domain restriction (optional - enforce @ufl.edu)
        # Uncomment to enforce domain restriction:
        # non_ufl_emails = df[~df['ufl_email'].str.endswith('@ufl.edu', na=False)]
        # if not non_ufl_emails.empty:
        #     errors.append(
        #         f"Non-UFL emails found: {non_ufl_emails['ufl_email'].head(3).tolist()}"
        #     )
        
        # 4. Duplicate emails within CSV
        duplicate_emails = df[df.duplicated(subset=['ufl_email'], keep=False)]
        if not duplicate_emails.empty:
            errors.append(
                f"Duplicate emails in CSV: "
                f"{duplicate_emails['ufl_email'].unique().tolist()[:5]}"
            )
        
        # 5. Check for existing applicants in semester
        existing_emails = self._get_existing_emails(semester_id)
        overlap = set(df['ufl_email'].str.lower()) & set(existing_emails)
        
        if overlap:
            errors.append(
                f"{len(overlap)} emails already exist in semester: "
                f"{list(overlap)[:5]}"
            )
        
        # 6. Organization quota check
        org = self.db.organizations.find_one({"_id": ObjectId(organization_id)})
        max_applicants = org["settings"]["max_applicants_per_semester"]
        
        if max_applicants > 0:  # -1 means unlimited
            current_count = self.db.applicants.count_documents({
                "applications.semester_id": ObjectId(semester_id)
            })
            
            if (current_count + len(df)) > max_applicants:
                errors.append(
                    f"Organization quota exceeded: "
                    f"Current: {current_count}, Upload: {len(df)}, "
                    f"Max: {max_applicants}"
                )
        
        # 7. Role distribution validation
        role_counts = df['role'].value_counts()
        mentors = role_counts.get(0, 0) + role_counts.get('mentor', 0)
        mentees = role_counts.get(1, 0) + role_counts.get('mentee', 0)
        
        if mentors == 0:
            errors.append("No mentors found in CSV")
        
        if mentees == 0:
            errors.append("No mentees found in CSV")
        
        if mentees < mentors * 2:
            errors.append(
                f"Warning: Insufficient mentees ({mentees}) for mentors ({mentors}). "
                f"Recommend at least 2:1 ratio."
            )
        
        # 8. Check for required survey fields (optional - based on your survey)
        optional_fields = ['bio', 'interests', 'goals']
        for field in optional_fields:
            if field in df.columns:
                empty_count = df[field].isna().sum()
                if empty_count > len(df) * 0.5:  # >50% missing
                    errors.append(
                        f"Warning: {empty_count}/{len(df)} rows missing '{field}'"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _get_existing_emails(self, semester_id: str) -> List[str]:
        """Get list of emails already in semester"""
        semester_oid = ObjectId(semester_id)
        
        pipeline = [
            {"$match": {"applications.semester_id": semester_oid}},
            {"$project": {"ufl_email": 1}}
        ]
        
        existing = list(self.db.applicants.aggregate(pipeline))
        return [doc["ufl_email"].lower() for doc in existing]
    
    def upload_applicants(
        self,
        df: pd.DataFrame,
        semester_id: str,
        organization_id: str,
        uploaded_by: str
    ) -> Dict:
        """
        Upload applicants from CSV to database
        
        Args:
            df: Validated DataFrame
            semester_id: Target semester ID
            organization_id: Organization ID
            uploaded_by: Email of user uploading
        
        Returns:
            {
              "total_uploaded": 450,
              "mentors": 150,
              "mentees": 300,
              "duplicates_skipped": 0,
              "errors": []
            }
        """
        # Validate first
        is_valid, errors = self.validate_csv(df, semester_id, organization_id)
        
        if not is_valid:
            return {
                "total_uploaded": 0,
                "mentors": 0,
                "mentees": 0,
                "duplicates_skipped": 0,
                "errors": errors
            }
        
        semester_oid = ObjectId(semester_id)
        org_oid = ObjectId(organization_id)
        
        uploaded_count = 0
        mentor_count = 0
        mentee_count = 0
        duplicate_count = 0
        
        for _, row in df.iterrows():
            try:
                # Normalize role
                role = "mentor" if row['role'] in [0, 'mentor', 'Mentor'] else "mentee"
                
                # Check if applicant already exists (by email)
                existing = self.db.applicants.find_one({"ufl_email": row['ufl_email'].lower()})
                
                if existing:
                    # Check if already applied for this semester
                    has_application = any(
                        app.get("semester_id") == semester_oid 
                        for app in existing.get("applications", [])
                    )
                    
                    if has_application:
                        duplicate_count += 1
                        continue
                    
                    # Add new application to existing applicant
                    self._add_application_to_existing(
                        existing["_id"],
                        semester_oid,
                        role,
                        row.to_dict()
                    )
                else:
                    # Create new applicant
                    self._create_new_applicant(
                        row.to_dict(),
                        semester_oid,
                        role,
                        uploaded_by
                    )
                
                uploaded_count += 1
                
                if role == "mentor":
                    mentor_count += 1
                else:
                    mentee_count += 1
                
            except Exception as e:
                logger.error(f"Error uploading applicant {row.get('ufl_email')}: {e}")
                errors.append(f"Row error: {str(e)}")
        
        logger.info(
            f"Uploaded {uploaded_count} applicants to semester {semester_id}: "
            f"{mentor_count} mentors, {mentee_count} mentees"
        )
        
        return {
            "total_uploaded": uploaded_count,
            "mentors": mentor_count,
            "mentees": mentee_count,
            "duplicates_skipped": duplicate_count,
            "errors": errors if errors else None
        }
    
    def _create_new_applicant(
        self,
        row_dict: Dict,
        semester_id: ObjectId,
        role: str,
        uploaded_by: str
    ) -> ObjectId:
        """Create new applicant document"""
        
        # Extract survey responses (all fields except metadata)
        metadata_fields = {'role', 'name', 'ufl_email'}
        survey_responses = {
            k: v for k, v in row_dict.items() 
            if k not in metadata_fields and pd.notna(v)
        }
        
        applicant_doc = {
            "ufl_email": row_dict['ufl_email'].lower(),
            "full_name": row_dict['name'],
            "applications": [{
                "semester_id": semester_id,
                "role": role,
                "survey_responses": survey_responses,
                "status": "pending",
                "submitted_at": datetime.now(timezone.utc),
                "uploaded_by": uploaded_by,
                "embeddings": {
                    "sbert_384": None,
                    "learned_64": None,
                    "sbert_computed_at": None,
                    "learned_computed_at": None
                }
            }],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = self.db.applicants.insert_one(applicant_doc)
        return result.inserted_id
    
    def _add_application_to_existing(
        self,
        applicant_id: ObjectId,
        semester_id: ObjectId,
        role: str,
        row_dict: Dict
    ):
        """Add new semester application to existing applicant"""
        
        metadata_fields = {'role', 'name', 'ufl_email'}
        survey_responses = {
            k: v for k, v in row_dict.items() 
            if k not in metadata_fields and pd.notna(v)
        }
        
        new_application = {
            "semester_id": semester_id,
            "role": role,
            "survey_responses": survey_responses,
            "status": "pending",
            "submitted_at": datetime.now(timezone.utc),
            "embeddings": {
                "sbert_384": None,
                "learned_64": None
            }
        }
        
        self.db.applicants.update_one(
            {"_id": applicant_id},
            {
                "$push": {"applications": new_application},
                "$set": {"updated_at": datetime.now(timezone.utc)}
            }
        )
    
    def get_applicants(
        self,
        semester_id: str,
        role: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get applicants for a semester
        
        Args:
            semester_id: Semester ID
            role: Filter by role ("mentor" or "mentee")
            skip: Pagination offset
            limit: Max results
        
        Returns:
            List of applicant documents with flattened structure
        """
        semester_oid = ObjectId(semester_id)
        
        pipeline = [
            {"$match": {"applications.semester_id": semester_oid}},
            {"$unwind": "$applications"},
            {"$match": {"applications.semester_id": semester_oid}},
        ]
        
        # Add role filter if specified
        if role:
            pipeline.append({"$match": {"applications.role": role}})
        
        # Project to flatten structure
        pipeline.extend([
            {"$project": {
                "applicant_id": {"$toString": "$_id"},
                "ufl_email": 1,
                "full_name": 1,
                "role": "$applications.role",
                "status": "$applications.status",
                "submitted_at": "$applications.submitted_at",
                "survey_responses": "$applications.survey_responses"
            }},
            {"$skip": skip},
            {"$limit": limit}
        ])
        
        applicants = list(self.db.applicants.aggregate(pipeline))
        
        # Remove MongoDB _id, keep applicant_id
        for app in applicants:
            app.pop("_id", None)
        
        return applicants
    
    def update_applicant(
        self,
        applicant_id: str,
        semester_id: str,
        updates: Dict
    ) -> bool:
        """
        Update applicant profile for a specific semester
        
        Args:
            applicant_id: Applicant ID
            semester_id: Semester ID
            updates: Fields to update in survey_responses
        
        Returns:
            True if updated
        """
        # Protected fields that shouldn't be updated
        protected = {'role', 'submitted_at', 'embeddings', 'status'}
        safe_updates = {k: v for k, v in updates.items() if k not in protected}
        
        if not safe_updates:
            return False
        
        # Update survey_responses for specific semester application
        set_updates = {
            f"applications.$[elem].survey_responses.{k}": v 
            for k, v in safe_updates.items()
        }
        set_updates["updated_at"] = datetime.now(timezone.utc)
        
        result = self.db.applicants.update_one(
            {"_id": ObjectId(applicant_id)},
            {"$set": set_updates},
            array_filters=[{"elem.semester_id": ObjectId(semester_id)}]
        )
        
        return result.modified_count > 0
    
    def delete_applicant(
        self,
        applicant_id: str,
        semester_id: str
    ) -> bool:
        """
        Remove applicant's application for a semester
        
        Note: Does not delete the applicant document, just removes
        the application for this semester
        
        Args:
            applicant_id: Applicant ID
            semester_id: Semester ID
        
        Returns:
            True if deleted
        """
        result = self.db.applicants.update_one(
            {"_id": ObjectId(applicant_id)},
            {
                "$pull": {
                    "applications": {"semester_id": ObjectId(semester_id)}
                },
                "$set": {"updated_at": datetime.now(timezone.utc)}
            }
        )
        
        logger.info(f"Removed application for semester {semester_id} from applicant {applicant_id}")
        
        return result.modified_count > 0