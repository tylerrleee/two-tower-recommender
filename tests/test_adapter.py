"""
python3 -m pytest tests/test_adapter.py"""

from database.adapter import DataAdapter
import pytest
from unittest.mock import patch, MagicMock
from pymongo.errors import ConnectionFailure
from bson import ObjectId

import pandas as pd

class TestDataAdapter:
    """
    Tests for database/adapter.py
    Covers: Initialization, Fetching, Aggregation, and Flattening
    """
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database object to inject into DataAdapter."""
        return MagicMock()

    @pytest.fixture
    def adapter(self, mock_db):
        """Initialize DataAdapter with the mock DB."""
        return DataAdapter(db=mock_db)

    # 1) Is it getting the database?
    @patch("database.adapter.get_database")
    def test_init_gets_database_automatically(self, mock_get_db):
        """Test that adapter calls get_database() if no db is provided."""
        _ = DataAdapter()
        mock_get_db.assert_called_once()

    # 2) Is training data being fetched? & 4) Is it flattened?
    def test_fetch_training_data_success(self, adapter, mock_db):
        """
        Test the full flow: Fetch -> Flatten -> DataFrame Split
        """
        semester_id_str = "507f1f77bcf86cd799439011" # Valid 24-char hex
        
        # 1. Mock the Semester check (must return something)
        mock_db.semesters.find_one.return_value = {"_id": ObjectId(semester_id_str)}
        
        # 2. Mock the Aggregation Result (Nested BSON-like structure)
        # This simulates what MongoDB returns BEFORE flattening
        mock_db.applicants.aggregate.return_value = [
            {
                "applicant_id": "1",
                "role": "mentor",
                "ufl_email": "mentor@ufl.edu",
                "survey_responses": {"major": "CS", "year": "Junior"}, # Nested
                "full_name" : "mentor_name1",
                "status": "pending"
            },
            {
                "applicant_id": "2",
                "role": "mentee",
                "ufl_email": "mentee@ufl.edu",
                "survey_responses": {"major": "Bio", "year": "Freshman"}, # Nested
                "full_name" : "mentor_name2",
                "status": "pending"
            }
        ]

        # Action
        df_mentors, df_mentees = adapter.fetch_training_data(semester_id_str)

        # Assertions
        # Did we find the semester?
        mock_db.semesters.find_one.assert_called_once()
        
        # Did we aggregate applicants?
        mock_db.applicants.aggregate.assert_called_once()

        # Did flattening happen? (Check if 'major' is a top-level column)
        assert "major" in df_mentors.columns
        assert "major" in df_mentees.columns
        
        # Check data integrity
        assert len(df_mentors) == 1
        assert df_mentors.iloc[0]["major"] == "CS"
        assert len(df_mentees) == 1
        assert df_mentees.iloc[0]["major"] == "Bio"

    # 3) Is the pipeline being aggregated?
    def test_aggregation_pipeline_structure(self, adapter):
        """
        Test internal _build_aggregation_pipeline logic to ensure 
        it constructs the correct MongoDB stages.
        """
        semester_id = ObjectId("507f1f77bcf86cd799439011") 
        
        pipeline = adapter._build_aggregation_pipeline(
            semester_id=semester_id, 
            include_embeddings=True
        )

        # We expect 4 specific stages: Match -> Project -> Unwind -> Project
        assert len(pipeline) == 4
        
        # Check Stage 1: Match Semester
        assert pipeline[0]["$match"]["applications.semester_id"] == semester_id
        
        # Check Stage 3: Unwind
        assert pipeline[2]["$unwind"] == "$relevant_app"
        
        # Check Stage 4: Final Projection includes embeddings
        final_proj = pipeline[3]["$project"]
        assert "sbert_embedding" in final_proj
        assert "learned_embedding" in final_proj

    def test_fetch_raises_if_semester_not_found(self, adapter, mock_db):
        """Test error handling when semester doesn't exist."""
        semester_id_str = "507f1f77bcf86cd799439011"
        mock_db.semesters.find_one.return_value = None # Not found
        
        with pytest.raises(ValueError) as exc:
            adapter.fetch_training_data(semester_id_str)
        
        assert "not found" in str(exc.value)

    def test_flatten_documents_logic(self, adapter):
        """
        Unit test specifically for _flatten_documents 
        to ensure edge cases (missing fields) are handled.
        """
        raw_docs = [
            {
                "applicant_id": "1", 
                "role": "mentor", 
                "ufl_email": "test@ufl.edu",
                "survey_responses": {"gym_freq": 5},
                "full_name" : "mentor_name1",
                # Missing 'status', missing 'submitted_at'
            }
        ]
        
        flat = adapter._flatten_documents(raw_docs, include_embeddings=False)
        
        # Should default status to 'pending' (as per code)
        assert flat[0]["status"] == "pending"
        # Should un-nest survey responses
        assert flat[0]["gym_freq"] == 5

if __name__ == '__main__':

    print("Tested DB Connection")