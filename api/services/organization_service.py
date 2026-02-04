"""
Organization Service - Business logic for multi-tenant organization management

Responsibilities:
- Create/Read/Update organizations
- Manage organization settings and quotas
- Handle organization membership
"""

from typing import List, Optional, Dict
from bson import ObjectId
from pymongo.database import Database
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class OrganizationService:
    """
    Handle organization-level operations
    
    Multi-tenancy strategy: Strict organization isolation
    - Users can only access resources within their organization
    - Admin role can access all organizations 
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_organization(
        self,
        name: str,
        subdomain: str,
        owner_email: str,
        plan: str = "free"
    ) -> str:
        """
        Create new organization
        
        Args:
            name: Organization name (e.g., "Vietnamese Student Organization - UFL")
            subdomain: Unique subdomain identifier (e.g., "vso-ufl")
            owner_email: Email of organization owner/admin
            plan: Subscription plan ("free", "premium", "enterprise")
        
        Returns:
            organization_id (str): MongoDB ObjectId as string
        
        Raises:
            ValueError: If subdomain already exists
        """
        # Validate subdomain uniqueness
        existing = self.db.organizations.find_one({"subdomain": subdomain})
        if existing:
            raise ValueError(f"Subdomain '{subdomain}' already exists")
        
        # Set quotas based on plan
        quotas = self._get_plan_quotas(plan)
        
        org_doc = {
            "name": name,
            "subdomain": subdomain,
            "plan": plan,
            "settings": {
                "max_applicants_per_semester": quotas["max_applicants"],
                "allowed_semesters": quotas["allowed_semesters"],
                "enable_faiss": quotas["enable_faiss"],
                "max_concurrent_matching_jobs": 5
            },
            "owner_email": owner_email,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "is_active": True
        }
        
        result = self.db.organizations.insert_one(org_doc)
        
        logger.info(f"Created organization '{name}' with ID {result.inserted_id}")
        
        return str(result.inserted_id)
    
    def _get_plan_quotas(self, plan: str) -> Dict:
        """Get resource quotas based on subscription plan"""
        quotas = {
            "free": {
                "max_applicants": 1000,
                "allowed_semesters": 4,
                "enable_faiss": False
            },
            "premium": {
                "max_applicants": 5000,
                "allowed_semesters": -1,  # Unlimited
                "enable_faiss": True
            },
            "enterprise": {
                "max_applicants": -1,  # Unlimited
                "allowed_semesters": -1,
                "enable_faiss": True
            }
        }
        
        return quotas.get(plan, quotas["free"])
    
    def get_organization(self, organization_id: str) -> Dict:
        """
        Get organization details
        
        Returns:
            Organization document with _id converted to string
        
        Raises:
            ValueError: If organization not found
        """
        org = self.db.organizations.find_one({"_id": ObjectId(organization_id)})
        
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Convert ObjectId to string for JSON serialization
        org["_id"] = str(org["_id"])
        
        return org
    
    def update_organization(
        self,
        organization_id: str,
        updates: Dict
    ) -> bool:
        """
        Update organization settings
        
        Args:
            organization_id: Organization ID
            updates: Dict of fields to update (e.g., {"name": "New Name"})
        
        Returns:
            True if updated, False if no changes
        """
        # Prevent updating protected fields
        protected_fields = {"_id", "subdomain", "created_at", "owner_email"}
        safe_updates = {k: v for k, v in updates.items() if k not in protected_fields}
        
        if not safe_updates:
            return False
        
        safe_updates["updated_at"] = datetime.now(timezone.utc)
        
        result = self.db.organizations.update_one(
            {"_id": ObjectId(organization_id)},
            {"$set": safe_updates}
        )
        
        if result.modified_count > 0:
            logger.info(f"Updated organization {organization_id}: {list(safe_updates.keys())}")
            return True
        
        return False
    
    def list_organizations(
        self,
        skip: int = 0,
        limit: int = 50,
        is_active: Optional[bool] = None
    ) -> List[Dict]:
        """
        List all organizations (admin-only endpoint)
        
        Args:
            skip: Pagination offset
            limit: Max results
            is_active: Filter by active status
        
        Returns:
            List of organization documents
        """
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        orgs = list(
            self.db.organizations
            .find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        
        # Convert ObjectIds
        for org in orgs:
            org["_id"] = str(org["_id"])
        
        return orgs
    
    def get_organization_stats(self, organization_id: str) -> Dict:
        """
        Get organization statistics
        
        Returns:
            {
              "total_users": 5,
              "total_semesters": 8,
              "total_applicants": 3600,
              "total_matches": 1800,
              "quota_usage": {
                "applicants": 0.72,  # 720/1000 for free tier
                "semesters": 0.5     # 4/8 allowed
              }
            }
        """
        org_oid = ObjectId(organization_id)
        
        # Count users
        total_users = self.db.applicants.count_documents(
            {"organization_id": org_oid} 
        )
        
        # Count semesters
        total_semesters = self.db.semesters.count_documents(
            {"organization_id": org_oid}
            )
        
        # Count applicants (across all semesters)
        total_applicants = self.db.applicants.count_documents(
            {"applications.semester_id": {"$exists": True}}
        )
        
        # Count match groups ------------------------------------->
        match_count_pipeline = [
            # Find semesters for this org
            {"$match": {"organization_id": org_oid}}, 
            # Join with match_groups to count them
            {"$lookup": {
                "from": "match_groups",
                "localField": "_id",
                "foreignField": "semester_id",
                "as": "matches"
            }},
            # Sum the size of the matches array
            {"$project": {"count": {"$size": "$matches"}}},
            {"$group": {"_id": None, "total": {"$sum": "$count"}}}
        ]
        match_agg = list(self.db.semesters.aggregate(match_count_pipeline))
        total_matches = match_agg[0]['total'] if match_agg else 0


        # Get organization settings for quota calculation
        org = self.get_organization(str(org_oid))
        max_applicants = org["settings"]["max_applicants_per_semester"]
        allowed_semesters = org["settings"]["allowed_semesters"]
        
        quota_usage = {}
        if max_applicants > 0:
            quota_usage["applicants"] = total_applicants / max_applicants
        if allowed_semesters > 0:
            quota_usage["semesters"] = total_semesters / allowed_semesters
        
        return {
            "total_users": total_users,
            "total_semesters": total_semesters,
            "total_applicants": total_applicants,
            "total_matches": total_matches,
            "quota_usage": quota_usage
        }
    
    def check_quota(
        self,
        organization_id: str,
        resource_type: str,
        requested_amount: int = 1
    ) -> bool:
        """
        Check if organization has quota available
        
        Args:
            organization_id: Organization ID
            resource_type: "applicants", "semesters", "matching_jobs"
            requested_amount: Amount to check (default 1)
        
        Returns:
            True if quota available, False otherwise
        """
        org = self.get_organization(organization_id)
        
        if resource_type == "applicants":
            max_allowed = org["settings"]["max_applicants_per_semester"]
            if max_allowed == -1:  # Unlimited
                return True
            
            # Get current semester's applicant count
            # This would need semester_id context - simplified for now
            return requested_amount <= max_allowed
        
        elif resource_type == "semesters":
            max_allowed = org["settings"]["allowed_semesters"]
            if max_allowed == -1:
                return True
            
            current_count = self.db.semesters.count_documents({
                "organization_id": ObjectId(organization_id)
            })
            
            return (current_count + requested_amount) <= max_allowed
        
        elif resource_type == "matching_jobs":
            max_concurrent = org["settings"]["max_concurrent_matching_jobs"]
            
            # Count active jobs
            active_jobs = self.db.matching_jobs.count_documents({
                "organization_id": ObjectId(organization_id),
                "status": {"$in": ["pending", "preprocessing", "embedding", "matching"]}
            })
            
            return (active_jobs + requested_amount) <= max_concurrent
        
        return True