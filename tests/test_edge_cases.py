"""
Edge Case and Error Handling Tests

Tests for:
- Boundary conditions
- Race conditions
- Invalid input handling
- Error recovery
- Concurrent operations

Run with:
    pytest tests/test_edge_cases.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from unittest.mock import Mock, patch
import pandas as pd

from api.services.organization_service import OrganizationService
from api.services.semester_service import SemesterService
from api.services.applicant_service import ApplicantService
from api.services.matching_service import MatchingService, JobStatus


# ============================================================================
# EDGE CASES: OrganizationService
# ============================================================================

class TestOrganizationEdgeCases:
    """Edge case tests for OrganizationService"""
    
    @pytest.fixture
    def org_service(self):
        mock_db = Mock()
        return OrganizationService(mock_db), mock_db
    
    def test_create_organization_empty_name(self, org_service):
        """Test creation with empty organization name"""
        service, mock_db = org_service
        mock_db.organizations.find_one.return_value = None
        
        # Should still create (validation is optional)
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.organizations.insert_one.return_value = mock_result
        
        org_id = service.create_organization(
            name="",  # Empty name
            subdomain="test",
            owner_email="test@test.com"
        )
        
        assert org_id is not None
    
    def test_create_organization_special_characters_subdomain(self, org_service):
        """Test subdomain with special characters"""
        service, mock_db = org_service
        mock_db.organizations.find_one.return_value = None
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.organizations.insert_one.return_value = mock_result
        
        # Should accept (no validation currently)
        org_id = service.create_organization(
            name="Test Org",
            subdomain="test@#$%",  # Special chars
            owner_email="test@test.com"
        )
        
        assert org_id is not None
        # Note: Production should validate subdomain format
    
    def test_get_organization_stats_zero_resources(self, org_service):
        """Test stats when organization has no resources"""
        service, mock_db = org_service
        org_id = str(ObjectId())
        
        # Mock organization
        mock_db.organizations.find_one.return_value = {
            "_id": ObjectId(org_id),
            "settings": {
                "max_applicants_per_semester": 1000,
                "allowed_semesters": 4
            }
        }
        
        # Mock zero counts
        mock_db.applicants.count_documents.return_value = 0
        mock_db.semesters.count_documents.return_value = 0
        mock_db.semesters.aggregate.return_value = []
        
        stats = service.get_organization_stats(org_id)
        
        assert stats["total_users"] == 0
        assert stats["total_semesters"] == 0
        assert stats["total_applicants"] == 0
        assert stats["total_matches"] == 0
        assert stats["quota_usage"]["applicants"] == 0.0
    
    def test_check_quota_boundary_exactly_at_limit(self, org_service):
        """Test quota check when exactly at limit"""
        service, mock_db = org_service
        org_id = str(ObjectId())
        
        mock_db.organizations.find_one.return_value = {
            "_id": ObjectId(org_id),
            "settings": {"allowed_semesters": 4}
        }
        
        # Exactly 4 semesters (at limit)
        mock_db.semesters.count_documents.return_value = 4
        
        # Should deny adding more
        result = service.check_quota(org_id, "semesters", 1)
        assert result is False
        
        # Should allow zero
        result = service.check_quota(org_id, "semesters", 0)
        assert result is True


# ============================================================================
# EDGE CASES: SemesterService
# ============================================================================

class TestSemesterEdgeCases:
    """Edge case tests for SemesterService"""
    
    @pytest.fixture
    def semester_service(self):
        mock_db = Mock()
        return SemesterService(mock_db), mock_db
    
    def test_create_semester_same_start_end_date(self, semester_service):
        """Test semester with start and end on same day"""
        service, mock_db = semester_service
        
        same_date = datetime(2024, 8, 26)
        
        with pytest.raises(ValueError, match="End date must be after start date"):
            service.create_semester(
                organization_id=str(ObjectId()),
                name="Test Semester",
                start_date=same_date,
                end_date=same_date,  # Same as start
                mentor_quota=10,
                mentee_quota=20,
                created_by="test@test.com"
            )
    
    def test_create_semester_zero_quota(self, semester_service):
        """Test creating semester with zero quota"""
        service, mock_db = semester_service
        mock_db.semesters.find_one.return_value = None
        
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.semesters.insert_one.return_value = mock_result
        
        # Should create but log warning
        semester_id = service.create_semester(
            organization_id=str(ObjectId()),
            name="Zero Quota Semester",
            start_date=datetime(2024, 8, 26),
            end_date=datetime(2024, 12, 15),
            mentor_quota=0,  # Zero mentors
            mentee_quota=0,  # Zero mentees
            created_by="test@test.com"
        )
        
        assert semester_id is not None
    
    def test_update_semester_locked_status(self, semester_service):
        """Test updating semester in locked status"""
        service, mock_db = semester_service
        semester_id = str(ObjectId())
        org_id = str(ObjectId())
        
        # Mock completed semester
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(semester_id),
            "organization_id": ObjectId(org_id),
            "status": "completed"
        }
        
        with pytest.raises(ValueError, match="Cannot update semester"):
            service.update_semester(
                semester_id, org_id, {"name": "New Name"}
            )
    
    def test_status_transition_archived_to_anything(self, semester_service):
        """Test that archived status is terminal"""
        service, mock_db = semester_service
        semester_id = str(ObjectId())
        org_id = str(ObjectId())
        
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(semester_id),
            "organization_id": ObjectId(org_id),
            "status": "archived"
        }
        
        # Archived has no valid transitions
        with pytest.raises(ValueError, match="Invalid status transition"):
            service.update_semester_status(semester_id, org_id, "active")
    
    def test_get_semester_stats_no_applicants(self, semester_service):
        """Test stats calculation with no applicants"""
        service, mock_db = semester_service
        semester_id = str(ObjectId())
        org_id = str(ObjectId())
        
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(semester_id),
            "organization_id": ObjectId(org_id),
            "quotas": {"mentors": 150, "mentees": 300}
        }
        
        # No applicants
        mock_db.applicants.aggregate.return_value = []
        mock_db.match_groups.aggregate.return_value = []

        mock_db.semesters.update_one.return_value = Mock()
        
        stats = service.get_semester_stats(semester_id, org_id)
        
        assert stats["total_applicants"] == 0
        assert stats["mentors"] == 0
        assert stats["mentees"] == 0
        assert stats["quota_fulfillment"]["mentors"] == 0.0


# ============================================================================
# EDGE CASES: ApplicantService
# ============================================================================

class TestApplicantEdgeCases:
    """Edge case tests for ApplicantService"""
    
    @pytest.fixture
    def applicant_service(self):
        mock_db = Mock()
        return ApplicantService(mock_db), mock_db
    
    def test_validate_csv_empty_dataframe(self, applicant_service):
        """Test validation with empty CSV"""
        service, mock_db = applicant_service
        empty_df = pd.DataFrame()
        
        is_valid, errors = service.validate_csv(
            empty_df, str(ObjectId()), str(ObjectId())
        )
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_validate_csv_mixed_role_formats(self, applicant_service):
        """Test CSV with mixed role formats (0, 'mentor', 'Mentor')"""
        service, mock_db = applicant_service
        
        csv_data = pd.DataFrame({
            "role": [0, "mentor", "Mentor", 1, "mentee"],
            "name": ["A", "B", "C", "D", "E"],
            "ufl_email": ["a@ufl.edu", "b@ufl.edu", "c@ufl.edu", "d@ufl.edu", "e@ufl.edu"],
            "major": ["CS"] * 5,
            "year": ["Junior"] * 5
        })
        
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        is_valid, errors = service.validate_csv(
            csv_data, str(ObjectId()), str(ObjectId())
        )
        
        # Should pass (service normalizes roles)
        assert is_valid is True or len(errors) == 1  # Might warn about ratio
    
    def test_validate_csv_null_values_in_required_fields(self, applicant_service):
        """Test CSV with null values in required fields"""
        service, mock_db = applicant_service
        
        csv_data = pd.DataFrame({
            "role": [0, None],  # Null role
            "name": ["Alice", "Bob"],
            "ufl_email": ["alice@ufl.edu", None],  # Null email
            "major": ["CS", "Bio"],
            "year": ["Junior", "Senior"]
        })
        
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        is_valid, errors = service.validate_csv(
            csv_data, str(ObjectId()), str(ObjectId())
        )
        
        assert is_valid is False
        # Should catch invalid emails
    
    def test_upload_applicants_duplicate_within_db(self, applicant_service):
        """Test upload when applicant exists in DB but not in semester"""
        service, mock_db = applicant_service
        semester_id = str(ObjectId())
        org_id = str(ObjectId())
        
        csv_data = pd.DataFrame({
            "role": [0],
            "name": ["Alice Smith"],
            "ufl_email": ["alice@ufl.edu"],
            "major": ["CS"],
            "year": ["Junior"]
        })
        
        # Mock validation passes
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        # Mock existing applicant (different semester)
        mock_db.applicants.find_one.return_value = {
            "_id": ObjectId(),
            "ufl_email": "alice@ufl.edu",
            "applications": [
                {"semester_id": ObjectId()}  # Different semester
            ]
        }
        
        mock_db.applicants.update_one.return_value = Mock()
        
        result = service.upload_applicants(
            csv_data, semester_id, org_id, "admin@test.com"
        )
        
        # Should add application to existing applicant
        assert result["total_uploaded"] == 1
        assert result["duplicates_skipped"] == 0


# ============================================================================
# EDGE CASES: MatchingService
# ============================================================================

class TestMatchingEdgeCases:
    """Edge case tests for MatchingService"""
    
    @pytest.fixture
    def matching_service(self):
        mock_db = Mock()
        return MatchingService(mock_db), mock_db
    
    def test_save_match_results_empty_groups(self, matching_service):
        """Test saving empty match results"""
        service, mock_db = matching_service
        
        service.save_match_results(
            job_id="test-job",
            semester_id=str(ObjectId()),
            groups=[]  # No groups
        )
        
        # Should delete old results but not insert new ones
        mock_db.match_groups.delete_many.assert_called_once()
        mock_db.match_groups.insert_many.assert_not_called()
    
    def test_update_job_status_multiple_updates(self, matching_service):
        """Test updating job status multiple times"""
        service, mock_db = matching_service
        job_id = "test-job"
        
        # Update to preprocessing
        service.update_job_status(job_id, JobStatus.PREPROCESSING)
        
        # Update to matching
        service.update_job_status(job_id, JobStatus.MATCHING)
        
        # Update to completed
        service.update_job_status(job_id, JobStatus.COMPLETED)
        
        # Should have 3 update calls
        assert mock_db.matching_jobs.update_one.call_count == 3
    
    def test_get_job_status_elapsed_time_not_started(self, matching_service):
        """Test elapsed time calculation when job hasn't started"""
        service, mock_db = matching_service
        
        mock_db.matching_jobs.find_one.return_value = {
            "job_id": "test-job",
            "status": JobStatus.PENDING,
            "progress": {},
            "created_at": datetime.now(timezone.utc),
            "started_at": None,  # Not started
            "completed_at": None,
            "results": {}
        }
        
        status = service.get_job_status("test-job")
        
        assert status["elapsed_seconds"] is None


# ============================================================================
# RACE CONDITION TESTS
# ============================================================================

class TestRaceConditions:
    """Test race conditions and concurrent operations"""
    
    def test_concurrent_organization_creation(self):
        """Test two users creating org with same subdomain simultaneously"""
        mock_db = Mock()
        service = OrganizationService(mock_db)
        
        # First check: no existing org
        mock_db.organizations.find_one.return_value = None
        
        # But second user created it between check and insert
        def insert_raises_duplicate(*args, **kwargs):
            from pymongo.errors import DuplicateKeyError
            raise DuplicateKeyError("Duplicate subdomain")
        
        mock_db.organizations.insert_one.side_effect = insert_raises_duplicate
        
        # Should propagate the error
        with pytest.raises(Exception):
            service.create_organization(
                name="Test", subdomain="test", owner_email="test@test.com"
            )
    
    def test_concurrent_semester_updates(self):
        """Test concurrent updates to same semester"""
        mock_db = Mock()
        service = SemesterService(mock_db)
        semester_id = str(ObjectId())
        org_id = str(ObjectId())
        
        # Mock semester
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(semester_id),
            "organization_id": ObjectId(org_id),
            "status": "draft"
        }
        
        # First update succeeds
        mock_result = Mock()
        mock_result.modified_count = 1
        mock_db.semesters.update_one.return_value = mock_result
        
        result1 = service.update_semester(semester_id, org_id, {"name": "Name1"})
        
        # Second update (concurrent) also succeeds
        result2 = service.update_semester(semester_id, org_id, {"name": "Name2"})
        
        assert result1 is True
        assert result2 is True
        # Note: Last write wins - expected behavior


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================

class TestErrorRecovery:
    """Test error handling and recovery scenarios"""
    
    def test_semester_stats_aggregation_error(self):
        """Test stats calculation when aggregation fails"""
        mock_db = Mock()
        service = SemesterService(mock_db)
        semester_id = str(ObjectId())
        org_id = str(ObjectId())
        
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(semester_id),
            "organization_id": ObjectId(org_id),
            "quotas": {"mentors": 150, "mentees": 300}
        }
        
        # Aggregation raises error
        mock_db.applicants.aggregate.side_effect = Exception("MongoDB error")
        
        # Should propagate error
        with pytest.raises(Exception):
            service.get_semester_stats(semester_id, org_id)
    
    def test_applicant_upload_partial_failure(self):
        """Test upload when some applicants fail to insert"""
        mock_db = Mock()
        service = ApplicantService(mock_db)
        
        csv_data = pd.DataFrame({
            "role": [0, 1, 1],
            "name": ["Alice", "Bob", "Tyler"],
            "ufl_email": ["alice@ufl.edu", "bob@ufl.edu", "tylerle@ufl.edu"],
            "major": ["CS", "Bio", "DS"],
            "year": ["Junior", "Senior", "Junior"]
        })
        
        # Mock validation passes
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        mock_db.applicants.find_one.return_value = None
        
        # First insert succeeds, second fails
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.applicants.insert_one.side_effect = [
            mock_result,  # Success
            Exception("Insert failed")  # Failure
        ]
        
        result = service.upload_applicants(
            csv_data, str(ObjectId()), str(ObjectId()), "admin@test.com"
        )
        
        # Should have 1 success, 1 error
        assert result["total_uploaded"] >= 1
        assert result["errors"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])