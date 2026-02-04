import pytest
from unittest.mock import MagicMock, ANY
from datetime import datetime
from bson import ObjectId
from api.services.matching import MatchingService, JobStatus
from api.services.feedback_service import FeedbackService

class TestMatchingService:
    def test_create_matching_job(self, mock_db):
        # Setup
        service = MatchingService(mock_db)
        
        # Action
        job_id = service.create_matching_job(
            semester_id     = "65b1234567890abcdef12345",
            organization_id = "65b9876543210fedcb543210",
            triggered_by    = "admin@test.com",
            config          = {"use_faiss": True}
        )
        
        # Assert
        assert isinstance(job_id, str)
        # Verify DB insert was called correctly
        mock_db.matching_jobs.insert_one.assert_called_once()
        call_args = mock_db.matching_jobs.insert_one.call_args[0][0]
        assert call_args["status"] == JobStatus.PENDING
        assert call_args["progress"]["current_step"] == "pending"

    def test_get_job_status_not_found(self, mock_db):
        # Setup
        service = MatchingService(mock_db)
        mock_db.matching_jobs.find_one.return_value = None  # Simulate no doc found
        
        # Assert
        with pytest.raises(ValueError, match="Job .* not found"):
            service.get_job_status("non_existent_id")

class TestFeedbackService:
    def test_submit_valid_feedback(self, mock_db):
        # Setup
        service = FeedbackService(mock_db)
        mock_db.feedback.insert_one.return_value.inserted_id = ObjectId()
        
        # Action
        result = service.submit_feedback(
            match_id="65b1234567890abcdef12345",
            semester_id="65b9876543210fedcb543210",
            submitted_by="student@ufl.edu",
            role="mentee",
            rating=5
        )
        
        # Assert
        assert isinstance(result, str)
        mock_db.feedback.insert_one.assert_called_once()
        
    def test_submit_invalid_rating(self, mock_db):
        service = FeedbackService(mock_db)
        with pytest.raises(ValueError):
            service.submit_feedback(
                match_id="...", semester_id="...", submitted_by="...", role="...", 
                rating=6 # <--- Invalid
            )

    def test_feedback_summary_aggregation(self, mock_db):
        """Test complex aggregation logic by mocking the pipeline response"""
        service = FeedbackService(mock_db)
        
        # Mock the Aggregation Pipeline Responses
        # 1. Average Rating Mock
        mock_db.feedback.aggregate.side_effect = [
            [{"avg_rating": 4.5}],          # avg_pipeline
            [{"_id": 5, "count": 10}],      # dist_pipeline
            [{"_id": "mentor", "avg": 4.5, "count": 10}], # role_pipeline
            [{"_id": "friendly", "count": 5}] # tag_pipeline
        ]
        mock_db.feedback.count_documents.return_value = 20  # total_feedback
        mock_db.applicants.count_documents.return_value = 100 # total_applicants

        # Action
        summary = service.get_semester_feedback_summary("65b...", "org...")

        # Assert
        assert summary["average_rating"] == 4.5
        assert summary["response_rate"] == 0.2  # 20/100
        assert summary["top_tags"][0]["tag"] == "friendly"