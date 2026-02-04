"""
Unit Tests for Backend Services

Test Coverage:
- OrganizationService (organization_service.py)
- SemesterService (semester_service.py)
- ApplicantService (applicant_service.py)
- MatchingService (matching_service.py)
- FeedbackService (feedback_service.py)

Run with:
    pytest tests/test_services.py -v
    pytest tests/test_services.py::TestOrganizationService -v
    pytest tests/test_services.py -k "test_create_organization" -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from io import StringIO

# Import services
import sys
sys.path.append("..")
from api.services.organization_service import OrganizationService
from api.services.semester_service import SemesterService
from api.services.applicant_service import ApplicantService
from api.services.matching_service import MatchingService, JobStatus
from api.services.feedback_service import FeedbackService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_db():
    """Mock MongoDB database"""
    db = Mock()
    
    # Mock collections
    db.organizations = Mock()
    db.semesters = Mock()
    db.applicants = Mock()
    db.matching_jobs = Mock()
    db.match_groups = Mock()
    db.feedback = Mock()
    db.users = Mock()
    
    return db


@pytest.fixture
def organization_service(mock_db):
    """OrganizationService instance with mocked DB"""
    return OrganizationService(mock_db)


@pytest.fixture
def semester_service(mock_db):
    """SemesterService instance with mocked DB"""
    return SemesterService(mock_db)


@pytest.fixture
def applicant_service(mock_db):
    """ApplicantService instance with mocked DB"""
    return ApplicantService(mock_db)


@pytest.fixture
def matching_service(mock_db):
    """MatchingService instance with mocked DB"""
    return MatchingService(mock_db)


@pytest.fixture
def feedback_service(mock_db):
    """FeedbackService instance with mocked DB"""
    return FeedbackService(mock_db)


@pytest.fixture
def sample_org_id():
    """Sample organization ObjectId"""
    return str(ObjectId())


@pytest.fixture
def sample_semester_id():
    """Sample semester ObjectId"""
    return str(ObjectId())


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing applicant upload"""
    csv_string = """role,name,ufl_email,major,year,bio,interests
0,Alice Smith,alice@ufl.edu,Computer Science,Junior,Love AI,Machine Learning
0,Bob Jones,bob@ufl.edu,Biology,Senior,Pre-med,Healthcare
1,Charlie Brown,charlie@ufl.edu,Physics,Freshman,New student,Research
1,Diana Prince,diana@ufl.edu,Math,Sophomore,Enjoys tutoring,Teaching
1,Eve Williams,eve@ufl.edu,Chemistry,Freshman,Lab work,Science
1,Frank Miller,frank@ufl.edu,Engineering,Sophomore,Robotics,Technology"""
    
    return pd.read_csv(StringIO(csv_string))


# ============================================================================
# TEST: OrganizationService
# ============================================================================

class TestOrganizationService:
    """Test suite for OrganizationService"""
    
    def test_create_organization_success(self, organization_service, mock_db):
        """Test successful organization creation"""
        # Mock insert result
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.organizations.find_one.return_value = None  # No existing subdomain
        mock_db.organizations.insert_one.return_value = mock_result
        
        # Create organization
        org_id = organization_service.create_organization(
            name="Test VSO",
            subdomain="test-vso",
            owner_email="owner@test.com",
            plan="free"
        )
        
        # Assertions
        assert org_id is not None
        assert isinstance(org_id, str)
        mock_db.organizations.insert_one.assert_called_once()
        
        # Verify document structure
        call_args = mock_db.organizations.insert_one.call_args[0][0]
        assert call_args["name"] == "Test VSO"
        assert call_args["subdomain"] == "test-vso"
        assert call_args["plan"] == "free"
        assert call_args["settings"]["max_applicants_per_semester"] == 1000
        assert call_args["is_active"] is True
    
    def test_create_organization_duplicate_subdomain(self, organization_service, mock_db):
        """Test creation fails with duplicate subdomain"""
        # Mock existing organization
        mock_db.organizations.find_one.return_value = {"subdomain": "test-vso"}
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            organization_service.create_organization(
                name="Test VSO",
                subdomain="test-vso",
                owner_email="owner@test.com"
            )
    
    def test_get_plan_quotas(self, organization_service):
        """Test quota calculation for different plans"""
        free_quotas = organization_service._get_plan_quotas("free")
        assert free_quotas["max_applicants"] == 1000
        assert free_quotas["allowed_semesters"] == 4
        assert free_quotas["enable_faiss"] is False
        
        premium_quotas = organization_service._get_plan_quotas("premium")
        assert premium_quotas["max_applicants"] == 5000
        assert premium_quotas["allowed_semesters"] == -1  # Unlimited
        assert premium_quotas["enable_faiss"] is True
        
        enterprise_quotas = organization_service._get_plan_quotas("enterprise")
        assert enterprise_quotas["max_applicants"] == -1  # Unlimited
        assert enterprise_quotas["enable_faiss"] is True
    
    def test_get_organization_success(self, organization_service, mock_db, sample_org_id):
        """Test retrieving organization by ID"""
        # Mock organization document
        mock_db.organizations.find_one.return_value = {
            "_id": ObjectId(sample_org_id),
            "name": "Test VSO",
            "subdomain": "test-vso",
            "plan": "free"
        }
        
        org = organization_service.get_organization(sample_org_id)
        
        assert org["_id"] == sample_org_id  # Converted to string
        assert org["name"] == "Test VSO"
        mock_db.organizations.find_one.assert_called_once()
    
    def test_get_organization_not_found(self, organization_service, mock_db, sample_org_id):
        """Test error when organization doesn't exist"""
        mock_db.organizations.find_one.return_value = None
        
        with pytest.raises(ValueError, match="not found"):
            organization_service.get_organization(sample_org_id)
    
    def test_update_organization(self, organization_service, mock_db, sample_org_id):
        """Test updating organization settings"""
        # Mock successful update
        mock_result = Mock()
        mock_result.modified_count = 1
        mock_db.organizations.update_one.return_value = mock_result
        
        updates = {"name": "Updated VSO", "plan": "premium"}
        result = organization_service.update_organization(sample_org_id, updates)
        
        assert result is True
        mock_db.organizations.update_one.assert_called_once()
    
    def test_update_organization_protected_fields(self, organization_service, mock_db, sample_org_id):
        """Test that protected fields cannot be updated"""
        updates = {"_id": "new_id", "subdomain": "new-subdomain"}
        result = organization_service.update_organization(sample_org_id, updates)
        
        # Should return False (no safe updates)
        assert result is False
        mock_db.organizations.update_one.assert_not_called()
    
    def test_list_organizations(self, organization_service, mock_db):
        """Test listing organizations with pagination"""
        mock_cursor = Mock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = [
            {"_id": ObjectId(), "name": "Org 1"},
            {"_id": ObjectId(), "name": "Org 2"}
        ]
        mock_db.organizations.find.return_value = mock_cursor
        
        orgs = organization_service.list_organizations(skip=0, limit=10)
        
        assert len(orgs) == 2
        assert all(isinstance(org["_id"], str) for org in orgs)
    
    def test_check_quota_semesters(self, organization_service, mock_db, sample_org_id):
        """Test semester quota checking"""
        # Mock organization with 4 semester limit
        mock_db.organizations.find_one.return_value = {
            "_id": ObjectId(sample_org_id),
            "settings": {"allowed_semesters": 4}
        }
        
        # Mock current count
        mock_db.semesters.count_documents.return_value = 3
        
        # Should allow (3 + 1 = 4 <= 4)
        result = organization_service.check_quota(sample_org_id, "semesters", 1)
        assert result is True
        
        # Should deny (3 + 2 = 5 > 4)
        result = organization_service.check_quota(sample_org_id, "semesters", 2)
        assert result is False
    
    def test_check_quota_unlimited(self, organization_service, mock_db, sample_org_id):
        """Test unlimited quota (-1)"""
        mock_db.organizations.find_one.return_value = {
            "_id": ObjectId(sample_org_id),
            "settings": {"allowed_semesters": -1}
        }
        
        result = organization_service.check_quota(sample_org_id, "semesters", 999)
        assert result is True
    
    def test_check_quota_matching_jobs(self, organization_service, mock_db, sample_org_id):
        """Test concurrent matching job quota"""
        mock_db.organizations.find_one.return_value = {
            "_id": ObjectId(sample_org_id),
            "settings": {"max_concurrent_matching_jobs": 5}
        }
        
        # Mock 3 active jobs
        mock_db.matching_jobs.count_documents.return_value = 3
        
        # Should allow (3 + 1 = 4 <= 5)
        result = organization_service.check_quota(sample_org_id, "matching_jobs", 1)
        assert result is True
        
        # Should deny (3 + 3 = 6 > 5)
        result = organization_service.check_quota(sample_org_id, "matching_jobs", 3)
        assert result is False


# ============================================================================
# TEST: SemesterService
# ============================================================================

class TestSemesterService:
    """Test suite for SemesterService"""
    
    def test_create_semester_success(self, semester_service, mock_db, sample_org_id):
        """Test successful semester creation"""
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.semesters.find_one.return_value = None  # No duplicate
        mock_db.semesters.insert_one.return_value = mock_result
        
        start_date = datetime(2024, 8, 26)
        end_date = datetime(2024, 12, 15)
        
        semester_id = semester_service.create_semester(
            organization_id=sample_org_id,
            name="Fall 2024",
            start_date=start_date,
            end_date=end_date,
            mentor_quota=150,
            mentee_quota=300,
            created_by="admin@test.com"
        )
        
        assert semester_id is not None
        mock_db.semesters.insert_one.assert_called_once()
        
        # Verify document structure
        call_args = mock_db.semesters.insert_one.call_args[0][0]
        assert call_args["name"] == "Fall 2024"
        assert call_args["status"] == "draft"
        assert call_args["quotas"]["mentors"] == 150
        assert call_args["quotas"]["mentees"] == 300
    
    def test_create_semester_invalid_dates(self, semester_service, mock_db, sample_org_id):
        """Test creation fails with invalid date range"""
        start_date = datetime(2024, 12, 15)
        end_date = datetime(2024, 8, 26)  # Before start date
        
        with pytest.raises(ValueError, match="End date must be after start date"):
            semester_service.create_semester(
                organization_id=sample_org_id,
                name="Fall 2024",
                start_date=start_date,
                end_date=end_date,
                mentor_quota=150,
                mentee_quota=300,
                created_by="admin@test.com"
            )
    
    def test_create_semester_duplicate_name(self, semester_service, mock_db, sample_org_id):
        """Test creation fails with duplicate semester name in org"""
        # Mock existing semester
        mock_db.semesters.find_one.return_value = {"name": "Fall 2024"}
        
        with pytest.raises(ValueError, match="already exists"):
            semester_service.create_semester(
                organization_id=sample_org_id,
                name="Fall 2024",
                start_date=datetime(2024, 8, 26),
                end_date=datetime(2024, 12, 15),
                mentor_quota=150,
                mentee_quota=300,
                created_by="admin@test.com"
            )
    
    def test_get_semester_with_isolation(self, semester_service, mock_db, 
                                        sample_semester_id, sample_org_id):
        """Test organization-scoped semester retrieval"""
        # Mock semester document
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(sample_semester_id),
            "organization_id": ObjectId(sample_org_id),
            "name": "Fall 2024",
            "status": "active"
        }
        
        semester = semester_service.get_semester(sample_semester_id, sample_org_id)
        
        assert semester["_id"] == sample_semester_id
        assert semester["organization_id"] == sample_org_id
        
        # Verify query enforced org isolation
        call_args = mock_db.semesters.find_one.call_args[0][0]
        assert call_args["organization_id"] == ObjectId(sample_org_id)
    
    def test_get_semester_access_denied(self, semester_service, mock_db, 
                                       sample_semester_id, sample_org_id):
        """Test access denied when semester belongs to different org"""
        mock_db.semesters.find_one.return_value = None  # Not found
        
        with pytest.raises(ValueError, match="not found or access denied"):
            semester_service.get_semester(sample_semester_id, sample_org_id)
    
    def test_update_semester_status_valid_transition(self, semester_service, mock_db,
                                                     sample_semester_id, sample_org_id):
        """Test valid status transition"""
        # Mock existing semester
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(sample_semester_id),
            "organization_id": ObjectId(sample_org_id),
            "status": "draft"
        }
        
        mock_result = Mock()
        mock_result.modified_count = 1
        mock_db.semesters.update_one.return_value = mock_result
        
        # draft -> active is valid
        result = semester_service.update_semester_status(
            sample_semester_id, sample_org_id, "active"
        )
        
        assert result is True
    
    def test_update_semester_status_invalid_transition(self, semester_service, mock_db,
                                                       sample_semester_id, sample_org_id):
        """Test invalid status transition"""
        # Mock existing semester
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(sample_semester_id),
            "organization_id": ObjectId(sample_org_id),
            "status": "draft"
        }
        
        # draft -> completed is invalid (must go through active -> matching)
        with pytest.raises(ValueError, match="Invalid status transition"):
            semester_service.update_semester_status(
                sample_semester_id, sample_org_id, "completed"
            )
    
    def test_delete_semester_soft(self, semester_service, mock_db,
                                  sample_semester_id, sample_org_id):
        """Test soft delete (archive)"""
        mock_db.match_groups.count_documents.return_value = 0
        
        mock_result = Mock()
        mock_result.modified_count = 1
        mock_db.semesters.update_one.return_value = mock_result
        
        result = semester_service.delete_semester(
            sample_semester_id, sample_org_id, soft_delete=True
        )
        
        assert result is True
        mock_db.semesters.update_one.assert_called_once()
    
    def test_delete_semester_hard_with_matches(self, semester_service, mock_db,
                                              sample_semester_id, sample_org_id):
        """Test hard delete fails when matches exist"""
        mock_db.match_groups.count_documents.return_value = 150  # Has matches
        
        with pytest.raises(ValueError, match="Cannot permanently delete"):
            semester_service.delete_semester(
                sample_semester_id, sample_org_id, soft_delete=False
            )
    
    def test_get_semester_stats(self, semester_service, mock_db,
                                sample_semester_id, sample_org_id):
        """Test semester statistics calculation"""
        # Mock semester
        mock_db.semesters.find_one.return_value = {
            "_id": ObjectId(sample_semester_id),
            "organization_id": ObjectId(sample_org_id),
            "quotas": {"mentors": 150, "mentees": 300}
        }
        
        # Mock the 'applicants' collection aggregation
        mock_db.applicants.aggregate.return_value = [
            {"_id": "mentor", "count": 150},
            {"_id": "mentee", "count": 300}
        ]
        
        # 3. Mock the 'match_groups' collection aggregation (This was missing!)
        mock_db.match_groups.aggregate.return_value = [
            {"count": 150, "avg_compatibility": 0.85}
        ]
        
        stats = semester_service.get_semester_stats(sample_semester_id, sample_org_id)
        
        assert stats["total_applicants"] == 450
        assert stats["mentors"] == 150
        assert stats["mentees"] == 300
        assert stats["matched_groups"] == 150
        assert stats["unmatched_mentees"] == 0
        assert stats["average_compatibility"] == 0.85
        assert stats["quota_fulfillment"]["mentors"] == 1.0
        assert stats["quota_fulfillment"]["mentees"] == 1.0


# ============================================================================
# TEST: ApplicantService
# ============================================================================

class TestApplicantService:
    """Test suite for ApplicantService"""
    
    def test_validate_csv_success(self, applicant_service, mock_db, 
                                  sample_csv_data, sample_semester_id, sample_org_id):
        """Test successful CSV validation"""
        # Mock no existing applicants
        mock_db.applicants.aggregate.return_value = []
        
        # Mock organization with high quota
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 10000}
        }
        
        # Mock no existing semester applicants
        mock_db.applicants.count_documents.return_value = 0
        
        is_valid, errors = applicant_service.validate_csv(
            sample_csv_data, sample_semester_id, sample_org_id
        )
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_csv_missing_columns(self, applicant_service, mock_db,
                                         sample_semester_id, sample_org_id):
        """Test validation fails with missing required columns"""
        # Create CSV without required column
        csv_data = pd.DataFrame({
            "name": ["Alice"],
            "major": ["CS"]
            # Missing: role, ufl_email, year
        })
        
        is_valid, errors = applicant_service.validate_csv(
            csv_data, sample_semester_id, sample_org_id
        )
        
        assert is_valid is False
        assert any("Missing required columns" in err for err in errors)
    
    def test_validate_csv_invalid_email_format(self, applicant_service, mock_db,
                                              sample_semester_id, sample_org_id):
        """Test validation fails with invalid email format"""
        csv_data = pd.DataFrame({
            "role": [0],
            "name": ["Alice"],
            "ufl_email": ["not-an-email"],  # Invalid
            "major": ["CS"],
            "year": ["Junior"]
        })
        
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        is_valid, errors = applicant_service.validate_csv(
            csv_data, sample_semester_id, sample_org_id
        )
        
        assert is_valid is False
        assert any("Invalid email format" in err for err in errors)
    
    def test_validate_csv_duplicate_emails(self, applicant_service, mock_db,
                                          sample_semester_id, sample_org_id):
        """Test validation fails with duplicate emails in CSV"""
        csv_data = pd.DataFrame({
            "role": [0, 1],
            "name": ["Alice", "Bob"],
            "ufl_email": ["alice@ufl.edu", "alice@ufl.edu"],  # Duplicate
            "major": ["CS", "Bio"],
            "year": ["Junior", "Senior"]
        })
        
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        is_valid, errors = applicant_service.validate_csv(
            csv_data, sample_semester_id, sample_org_id
        )
        
        assert is_valid is False
        assert any("Duplicate emails in CSV" in err for err in errors)
    
    def test_validate_csv_quota_exceeded(self, applicant_service, mock_db,
                                        sample_csv_data, sample_semester_id, sample_org_id):
        """Test validation fails when organization quota exceeded"""
        mock_db.applicants.aggregate.return_value = []
        
        # Mock organization with low quota
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 100}
        }
        
        # Mock 95 existing applicants
        mock_db.applicants.count_documents.return_value = 95
        
        is_valid, errors = applicant_service.validate_csv(
            sample_csv_data, sample_semester_id, sample_org_id
        )
        
        assert is_valid is False
        assert any("quota exceeded" in err for err in errors)
    
    def test_validate_csv_insufficient_mentees(self, applicant_service, mock_db,
                                              sample_semester_id, sample_org_id):
        """Test validation warns about insufficient mentees"""
        # 3 mentors, 3 mentees (should be at least 6)
        csv_data = pd.DataFrame({
            "role": [0, 0, 0, 1, 1, 1],
            "name": ["M1", "M2", "M3", "E1", "E2", "E3"],
            "ufl_email": ["m1@ufl.edu", "m2@ufl.edu", "m3@ufl.edu", 
                         "e1@ufl.edu", "e2@ufl.edu", "e3@ufl.edu"],
            "major": ["CS"] * 6,
            "year": ["Junior"] * 6
        })
        
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        is_valid, errors = applicant_service.validate_csv(
            csv_data, sample_semester_id, sample_org_id
        )
        
        assert is_valid is False
        assert any("Insufficient mentees" in err for err in errors)
    
    def test_upload_applicants_success(self, applicant_service, mock_db,
                                      sample_csv_data, sample_semester_id, sample_org_id):
        """Test successful applicant upload"""
        # Mock validation passes
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 10000}
        }
        mock_db.applicants.count_documents.return_value = 0
        
        # Mock no existing applicants
        mock_db.applicants.find_one.return_value = None
        
        # Mock successful inserts
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.applicants.insert_one.return_value = mock_result
        
        result = applicant_service.upload_applicants(
            sample_csv_data, sample_semester_id, sample_org_id, "admin@test.com"
        )
        
        assert result["total_uploaded"] == 6
        assert result["mentors"] == 2
        assert result["mentees"] == 4
        assert result["duplicates_skipped"] == 0
    
    def test_create_new_applicant(self, applicant_service, mock_db, sample_semester_id):
        """Test creating new applicant document"""
        row_dict = {
            "role": 0,
            "name": "Alice Smith",
            "ufl_email": "alice@ufl.edu",
            "major": "Computer Science",
            "year": "Junior",
            "bio": "Love AI"
        }
        
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.applicants.insert_one.return_value = mock_result
        
        applicant_id = applicant_service._create_new_applicant(
            row_dict, ObjectId(sample_semester_id), "mentor", "admin@test.com"
        )
        
        assert applicant_id is not None
        
        # Verify document structure
        call_args = mock_db.applicants.insert_one.call_args[0][0]
        assert call_args["ufl_email"] == "alice@ufl.edu"
        assert call_args["full_name"] == "Alice Smith"
        assert len(call_args["applications"]) == 1
        assert call_args["applications"][0]["role"] == "mentor"
        assert call_args["applications"][0]["survey_responses"]["major"] == "Computer Science"
    
    def test_get_applicants_with_role_filter(self, applicant_service, mock_db, sample_semester_id):
        """Test retrieving applicants with role filter"""
        mock_db.applicants.aggregate.return_value = [
            {
                "applicant_id": str(ObjectId()),
                "ufl_email": "mentor@ufl.edu",
                "full_name": "Alice Smith",
                "role": "mentor",
                "status": "pending"
            }
        ]
        
        applicants = applicant_service.get_applicants(
            sample_semester_id, role="mentor", skip=0, limit=100
        )
        
        assert len(applicants) == 1
        assert applicants[0]["role"] == "mentor"


# ============================================================================
# TEST: MatchingService
# ============================================================================

class TestMatchingService:
    """Test suite for MatchingService"""
    
    def test_create_matching_job(self, matching_service, mock_db,
                                 sample_semester_id, sample_org_id):
        """Test creating a new matching job"""
        config = {"use_faiss": False, "top_k": 10}
        
        job_id = matching_service.create_matching_job(
            semester_id=sample_semester_id,
            organization_id=sample_org_id,
            triggered_by="admin@test.com",
            config=config
        )
        
        assert job_id is not None
        assert len(job_id) == 36  # UUID format
        
        mock_db.matching_jobs.insert_one.assert_called_once()
        
        # Verify document structure
        call_args = mock_db.matching_jobs.insert_one.call_args[0][0]
        assert call_args["job_id"] == job_id
        assert call_args["status"] == JobStatus.PENDING
        assert call_args["config"] == config
        assert call_args["progress"]["total_steps"] == 5
    
    def test_update_job_status_to_preprocessing(self, matching_service, mock_db):
        """Test updating job status to preprocessing"""
        job_id = "test-job-123"
        progress = {"current_step": "loading_data", "completed_steps": 1, "total_steps": 5}
        
        matching_service.update_job_status(
            job_id, JobStatus.PREPROCESSING, progress=progress
        )
        
        mock_db.matching_jobs.update_one.assert_called_once()
        
        # Verify update document
        call_args = mock_db.matching_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == JobStatus.PREPROCESSING
        assert update_doc["progress"] == progress
        assert "started_at" in update_doc
    
    def test_update_job_status_to_completed(self, matching_service, mock_db):
        """Test updating job status to completed"""
        job_id = "test-job-123"
        
        matching_service.update_job_status(job_id, JobStatus.COMPLETED)
        
        call_args = mock_db.matching_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == JobStatus.COMPLETED
        assert "completed_at" in update_doc
    
    def test_update_job_status_to_failed(self, matching_service, mock_db):
        """Test updating job status to failed with error message"""
        job_id = "test-job-123"
        error_msg = "Insufficient mentors"
        
        matching_service.update_job_status(
            job_id, JobStatus.FAILED, error_message=error_msg
        )
        
        call_args = mock_db.matching_jobs.update_one.call_args
        update_doc = call_args[0][1]["$set"]
        assert update_doc["status"] == JobStatus.FAILED
        assert update_doc["error_message"] == error_msg
    
    def test_get_job_status(self, matching_service, mock_db):
        """Test retrieving job status"""
        job_id = "test-job-123"
        
        # Mock job document
        mock_db.matching_jobs.find_one.return_value = {
            "job_id": job_id,
            "status": JobStatus.MATCHING,
            "progress": {"current_step": "hungarian_algorithm", "completed_steps": 4},
            "created_at": datetime.now(timezone.utc),
            "started_at": datetime.now(timezone.utc) - timedelta(seconds=120),
            "completed_at": None,
            "error_message": None,
            "results": {"total_groups": None}
        }
        
        status = matching_service.get_job_status(job_id)
        
        assert status["job_id"] == job_id
        assert status["status"] == JobStatus.MATCHING
        assert status["elapsed_seconds"] is not None
        assert status["elapsed_seconds"] > 100  # Approximately 120 seconds
    
    def test_get_job_status_not_found(self, matching_service, mock_db):
        """Test error when job doesn't exist"""
        mock_db.matching_jobs.find_one.return_value = None
        
        with pytest.raises(ValueError, match="not found"):
            matching_service.get_job_status("nonexistent-job")
    
    def test_save_match_results(self, matching_service, mock_db, sample_semester_id):
        """Test saving match results"""
        job_id = "test-job-123"
        groups = [
            {
                "group_id": 0,
                "mentor": {"name": "Alice", "email": "alice@ufl.edu"},
                "mentees": [
                    {"name": "Bob", "major": "CS"},
                    {"name": "Charlie", "major": "Bio"}
                ],
                "compatibility_score": 0.85,
                "individual_scores": [0.83, 0.87]
            },
            {
                "group_id": 1,
                "mentor": {"name": "Diana", "email": "diana@ufl.edu"},
                "mentees": [
                    {"name": "Eve", "major": "Math"},
                    {"name": "Frank", "major": "Physics"}
                ],
                "compatibility_score": 0.90,
                "individual_scores": [0.88, 0.92]
            }
        ]
        
        matching_service.save_match_results(job_id, sample_semester_id, groups)
        
        # Verify old results deleted
        mock_db.match_groups.delete_many.assert_called_once()
        
        # Verify new results inserted
        mock_db.match_groups.insert_many.assert_called_once()
        call_args = mock_db.match_groups.insert_many.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["group_id"] == 0
        assert call_args[1]["compatibility_score"] == 0.90
        
        # Verify job updated with summary
        mock_db.matching_jobs.update_one.assert_called_once()


# ============================================================================
# TEST: FeedbackService
# ============================================================================

class TestFeedbackService:
    """Test suite for FeedbackService"""
    
    def test_submit_feedback_success(self, feedback_service, mock_db):
        """Test submitting feedback"""
        match_id = str(ObjectId())
        semester_id = str(ObjectId())
        
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.feedback.insert_one.return_value = mock_result
        
        feedback_id = feedback_service.submit_feedback(
            match_id=match_id,
            semester_id=semester_id,
            submitted_by="mentor@ufl.edu",
            role="mentor",
            rating=5,
            comment="Great match!",
            tags=["good_communication", "shared_interests"]
        )
        
        assert feedback_id is not None
        
        # Verify document structure
        call_args = mock_db.feedback.insert_one.call_args[0][0]
        assert call_args["rating"] == 5
        assert call_args["role"] == "mentor"
        assert call_args["comment"] == "Great match!"
        assert "good_communication" in call_args["tags"]
    
    def test_submit_feedback_invalid_rating(self, feedback_service, mock_db):
        """Test feedback fails with invalid rating"""
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            feedback_service.submit_feedback(
                match_id=str(ObjectId()),
                semester_id=str(ObjectId()),
                submitted_by="test@ufl.edu",
                role="mentor",
                rating=6  # Invalid
            )
        
        with pytest.raises(ValueError, match="Rating must be between 1 and 5"):
            feedback_service.submit_feedback(
                match_id=str(ObjectId()),
                semester_id=str(ObjectId()),
                submitted_by="test@ufl.edu",
                role="mentor",
                rating=0  # Invalid
            )
    
    def test_get_semester_feedback_summary(self, feedback_service, mock_db, sample_semester_id):
        """Test getting feedback summary for semester"""
        # Mock total feedback count
        mock_db.feedback.count_documents.return_value = 300
        
        # Mock aggregation results
        mock_db.feedback.aggregate.side_effect = [
            # Average rating
            [{"avg_rating": 4.2, "total": 300}],
            # Rating distribution
            [
                {"_id": 5, "count": 120},
                {"_id": 4, "count": 100},
                {"_id": 3, "count": 50},
                {"_id": 2, "count": 20},
                {"_id": 1, "count": 10}
            ],
            # By role
            [
                {"_id": "mentor", "avg": 4.3, "count": 150},
                {"_id": "mentee", "avg": 4.1, "count": 150}
            ],
            # Top tags
            [
                {"_id": "good_communication", "count": 85},
                {"_id": "shared_interests", "count": 72}
            ]
        ]
        
        # Mock total applicants
        mock_db.applicants.count_documents.return_value = 450
        
        summary = feedback_service.get_semester_feedback_summary(
            sample_semester_id, "org123"
        )
        
        assert summary["total_feedback"] == 300
        assert summary["response_rate"] == 0.67
        assert summary["average_rating"] == 4.2
        assert summary["rating_distribution"]["5"] == 120
        assert summary["by_role"]["mentor"]["avg"] == 4.3
        assert len(summary["top_tags"]) == 2
    
    def test_export_labeled_dataset(self, feedback_service, mock_db, sample_semester_id):
        """Test exporting labeled dataset for ML retraining"""
        # Mock aggregation result
        mock_db.feedback.aggregate.return_value = [
            {
                "_id": ObjectId(),
                "avg_rating": 4.5,
                "feedback_count": 2,
                "comments": ["Great!", "Perfect match"],
                "match": {
                    "_id": ObjectId(),
                    "mentor": {"email": "mentor@ufl.edu"},
                    "mentees": [
                        {"email": "mentee1@ufl.edu"},
                        {"email": "mentee2@ufl.edu"}
                    ]
                }
            },
            {
                "_id": ObjectId(),
                "avg_rating": 2.0,
                "feedback_count": 1,
                "comments": ["Not compatible"],
                "match": {
                    "_id": ObjectId(),
                    "mentor": {"email": "mentor2@ufl.edu"},
                    "mentees": [
                        {"email": "mentee3@ufl.edu"},
                        {"email": "mentee4@ufl.edu"}
                    ]
                }
            }
        ]
        
        dataset = feedback_service.export_labeled_dataset(
            sample_semester_id, min_rating_threshold=4
        )
        
        assert len(dataset) == 2
        
        # First match: good (rating >= 4)
        assert dataset[0]["compatibility_label"] == 1
        assert dataset[0]["rating"] == 4.5
        assert dataset[0]["feedback_count"] == 2
        
        # Second match: poor (rating < 4)
        assert dataset[1]["compatibility_label"] == 0
        assert dataset[1]["rating"] == 2.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestServiceIntegration:
    """Integration tests combining multiple services"""
    
    def test_full_semester_workflow(self, organization_service, semester_service,
                                    applicant_service, matching_service, feedback_service,
                                    mock_db, sample_csv_data):
        """Test complete workflow: org -> semester -> applicants -> matching -> feedback"""
        
        # 1. Create organization
        mock_db.organizations.find_one.return_value = None
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        mock_db.organizations.insert_one.return_value = mock_result
        
        org_id = organization_service.create_organization(
            name="Test VSO", subdomain="test-vso", owner_email="admin@test.com"
        )
        
        # 2. Create semester
        mock_db.semesters.find_one.return_value = None
        mock_db.semesters.insert_one.return_value = mock_result
        
        semester_id = semester_service.create_semester(
            organization_id=org_id,
            name="Fall 2024",
            start_date=datetime(2024, 8, 26),
            end_date=datetime(2024, 12, 15),
            mentor_quota=2,
            mentee_quota=4,
            created_by="admin@test.com"
        )
        
        # 3. Upload applicants
        mock_db.applicants.aggregate.return_value = []
        mock_db.organizations.find_one.return_value = {
            "settings": {"max_applicants_per_semester": 1000}
        }
        mock_db.applicants.count_documents.return_value = 0
        mock_db.applicants.find_one.return_value = None
        mock_db.applicants.insert_one.return_value = mock_result
        
        upload_result = applicant_service.upload_applicants(
            sample_csv_data, semester_id, org_id, "admin@test.com"
        )
        
        assert upload_result["total_uploaded"] == 6
        
        # 4. Create matching job
        job_id = matching_service.create_matching_job(
            semester_id=semester_id,
            organization_id=org_id,
            triggered_by="admin@test.com",
            config={"use_faiss": False}
        )
        
        assert job_id is not None
        
        # 5. Submit feedback
        match_id = str(ObjectId())
        mock_db.feedback.insert_one.return_value = mock_result
        
        feedback_id = feedback_service.submit_feedback(
            match_id=match_id,
            semester_id=semester_id,
            submitted_by="mentor@test.com",
            role="mentor",
            rating=5
        )
        
        assert feedback_id is not None


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
