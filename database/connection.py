"""
MongoDB connection manager

- Creates a single instance of MongoDB connection
-- Atlas has connection limits so our goal is to limit & timeout failed connections
- Auto fileover
"""

import os 
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from pymongo.database import Database
import logging
import db_config


logger = logging.getLogger(__name__)

class MongoDBConnection:
    """
    Singleton MongoDB connection manager
    
    client: MongoClient instance
    db: database instance for matching
    """
    _instance   : Optional['MongoDBConnection'] = None
    _client     : Optional[MongoClient]         = None
    _db         : Optional[Database]            = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        
        return cls._instance

    def __init__(self):
        """ Initialize connection"""
        if self._client is None:
            self._connect()
    
    def _connect(self) -> None:
        """
        Establish connection to MongoDB Atlas w/ auth

        Raises:
            ConnectionFailure: if cannot conenct to MongoDB
            ConfigurationError: If connection string is invalid
        """
        try:
            connection_string = os.getenv(
                "MONGODB_URL",
                "mongodb://localhost:27017/" # fallback
            )

            self._client = MongoClient(
                connection_string,
                maxPoolSize              = db_config.maxPoolSize,
                minPoolSize              = db_config.minPoolSize,
                maxIdleTimeMS            = db_config.maxIdleTimeMS,
                serverSelectionTimeoutMS = db_config.serveSelectionTimeoutMS,
                connectTimeoutMS         = db_config.connectTimeoutMS,
                socketTimeoutMS          = db_config.socketTimeoutMS
            )
            # Test connection
            self._client.admin.command('ping')

            db_name = os.getenv("MONGODB_DATABASE", "two-tower")
            self._db = self._client[db_name]

        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        except ConfigurationError as e:
            logger.error(f"Invalid MongoDB URI: {e}")
            raise

    @property
    def client(self) -> MongoClient:
        """ Get instance"""
        if self._client is None:
            self._connect()
        return self._client
    
    @property
    def db(self) -> Database:
        """ Get Database instance"""
        if self._db is None:
            self._connect()
        return self._db
    
    def close(self) -> None:
        """ Close Connection"""
        if self._client is not None:
            self._client.close()
            logger.info("MongoDB connection closed")
            self._client = None
            self._db = None

#_mongo_connection = MongoDBConnection() - when imported, it gets called immediately
# could fail if cluster's down? 

def get_database() -> Database:
    """
    get an instance of MongoDB database instance.

    Returns:
        Database: PyMongo Database object

    Example:
        >>> db = get_database()
        >>> db.applicants.count_documents({})
        Output: 450
    """
    return MongoDBConnection().db

def close_connection() -> None:
    """ Close MongoDB connection (call on app shutdown)"""
    MongoDBConnection().close()

if __name__ == '__main__':
    _mongo_connection = MongoDBConnection()
    get_database()