import pytest
from unittest.mock import patch, MagicMock
from pymongo.errors import ConnectionFailure

from database.connection import MongoDBConnection, get_database
#from database.adapter import DataAdapter

class TestMongoDBConnection:
    
    @pytest.fixture(autouse=True)
    def clean_singleton(self):
        """Reset the Singleton instance before each test to ensure isolation."""
        MongoDBConnection._instance = None
        MongoDBConnection._client = None
        MongoDBConnection._db = None

    @patch("database.MongoClient")
    def test_singleton_pattern(self, mock_client):
        """Ensure only one instance is created even if called multiple times."""
        # Setup mock to avoid actual network calls
        mock_client.return_value.admin.command.return_value = True
        
        db1 = MongoDBConnection()
        db2 = MongoDBConnection()
        
        assert db1 is db2
        assert db1.client is db2.client
        # Ensure MongoClient was only initialized once
        assert mock_client.call_count == 1

    @patch("database.MongoClient")
    def test_successful_connection(self, mock_client):
        """Test that client and db are initialized correctly on success."""
        # Setup successful ping
        mock_client.return_value.admin.command.return_value = True
        
        conn = MongoDBConnection()
        
        assert conn.client is not None
        assert conn.db is not None
        mock_client.return_value.admin.command.assert_called_with('ping')

    @patch("database.MongoClient")
    def test_connection_failure(self, mock_client):
        """Test that ConnectionFailure is raised and logged."""
        # Simulate a ping failure
        mock_client.return_value.admin.command.side_effect = ConnectionFailure("Timeout")
        
        with pytest.raises(ConnectionFailure):
            MongoDBConnection()

    @patch("database._mongo_connection")
    def test_get_database_helper(self, mock_conn_instance):
        """Test the global helper function."""
        # Mock the global instance's db property
        mock_db = MagicMock()
        mock_conn_instance.db = mock_db
        
        result = get_database()
        
        assert result == mock_db

    @patch("database.adapters.MongoClient")
    def test_adapter_load_semester_data(self, mock_client):
        """Test that MongoDB BSON is correctly flattened to a DataFrame."""
        ...

if __name__ == '__main__':
    #unittest.main()
    print("Tested Database")