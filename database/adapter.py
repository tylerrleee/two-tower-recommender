"""
Adapter to fetch training data from MongoDB and transform for feature engineering format (/src)
"""

from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np
from bson import ObjectId
from pymongo.database import Database
import datetime
import logging

from database.connection import get_database
import config

logger = logging.getLogger(__name__)

class DataAdapter:
    """
    Convert MongoDB documents -> Pandas Dataframes

    """
    def __init__(self, db: Optional[Database] = None):
        """
        Initialize adapter w/ MongoDB
        """
        self.db = db or get_database()
        self.current_semester_id: Optional[ObjectId] = None
    
    def fetch_training_data(
            self,
            semester_id : str,
            include_embeddings: bool = True
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        fetch & transform applicant data for a given semester

        Args:
            semester_id: MongoDB ObjectId string for the semester
            include_embeddings: If True, include cached S-BERT embeddings
        
        Returns:
            Tuple of (df_mentors, df_mentees) as Pandas DataFrames
            
        Raises:
            ValueError: If semester not found or insufficient data
            
        Example:
            >>> adapter = DataAdapter()
            >>> df_mentors, df_mentees = adapter.fetch_training_data(
                                                    semester_id="67a1b2c3...", 
                                                    include_embeddings = True) # semester_id
            >>> df_mentors.shape
            (150, 25)
            >>> df_mentees.shape
            (300, 25)
        """
        self.current_semester_id = ObjectId(semester_id)
        logger.info(f"Fetching training data for semester: {semester_id}")

        semester = self.db.semesters.find_one({"_id": self.current_semester_id})
        if not semester:
            raise ValueError(f"Semester {semester_id} not found")

        pipeline = self._build_aggregation_pipeline(
                semester_id         = self.current_semester_id,
                include_embeddings  = include_embeddings
        )
        
        results = list(self.db.applicants.aggregate(pipeline))

        if not results:
            raise ValueError(f"No applicants found for semester {semester_id}")
        
        # Transform to flat records
        flattened_data = self._flatten_documents(results, include_embeddings)
        
        df_all = pd.DataFrame(flattened_data)
        
        # Split by role
        df_mentors = df_all[df_all["role"] == "mentor"].copy()
        df_mentees = df_all[df_all["role"] == "mentee"].copy()
        
        # Validate data
        self._validate_dataframes(df_mentors, df_mentees)

        logger.info(f"Loaded {len(df_mentors)} mentors, {len(df_mentees)} mentees")
        
        return df_mentors, df_mentees
    
    def _build_aggregation_pipeline(
                self, 
                semester_id: ObjectId,
                include_embeddings: bool
            ) -> List[Dict]:
        """
        Build MongoDB aggregation pipeline to extract semester-specific data.
        
        Pipeline stages:
        1. Match    : Filter to applicants with the target semester
        2. Project  : Extract relevant application + flatten survey_responses
        3. Unwind   : Convert applications array to individual documents
        4. Match    : Filter to the specific semester application
        5. Project  : Final field mapping
        """

        projection = {
            "applicant_id": {"$toString": "$_id"}, # find id & convert to string
            "ufl_email": 1,
            "full_name": 1,
            "role": "$relevant_app.role",
            "submitted_at": "$relevant_app.submitted_at",
            "status": "$relevant_app.status",
            "survey_responses": "$relevant_app.survey_responses"
        }

        # Embeddings
        if include_embeddings:
            projection["sbert_embedding"] = "$relevant_app.embeddings.sbert_384"
            projection["learned_embedding"] = "$relevant_app.embeddings.learned_64"
        
        pipeline = [
            # 1.Filter to applicants with this semester
            {
                "$match": {
                    "applications.semester_id": semester_id
                }},
            
            # 2. Project only the relevant application
            {
                "$project": {
                    "ufl_email": 1,
                    "full_name": 1,
                    "relevant_app": {
                        "$filter": {
                            "input": "$applications",
                            "as": "app",
                            "cond": {"$eq": ["$$app.semester_id", semester_id]}
                        }
                    }
                }
            },
            
            # 3. Unwind to get single application per document
            {
                "$unwind": "$relevant_app"
            },
            
            # 4. Final projection with flattened fields
            {
                "$project": projection
            }
        ]
        return pipeline
    
    def _flatten_documents(
                self, 
                documents: List[Dict],
                include_embeddings: bool
            ) -> List[Dict]:
        """
        Flatten MongoDB documents to match CSV-like structure
        
        Transforms:
        {
          "survey_responses": {"year": "Junior", "major": "CS", ...} # JSON
        }
        -> {
            "year": "Junior",  # Dict
            "major": "CS",  
            ...
            }
        """
        flattened = []
        
        for doc in documents:
            flat_record = {
                "applicant_id"  : doc["applicant_id"],
                "ufl_email"     : doc["ufl_email"],
                "full_name"     : doc["full_name"],
                "role"          : doc["role"],
                "submitted_at"  : doc.get("submitted_at"),
                "status"        : doc.get("status", "pending")
            }
            
            # Unpack survey_responses to top-level fields
            survey_data = doc.get("survey_responses", {})
            flat_record.update(survey_data)
            
            # Include embeddings if requested
            if include_embeddings:
                flat_record["sbert_embedding"] = doc.get("sbert_embedding")
                flat_record["learned_embedding"] = doc.get("learned_embedding")
            
            flattened.append(flat_record)
        
        return flattened

    def _validate_dataframes(
                self, 
                df_mentors: pd.DataFrame, 
                df_mentees: pd.DataFrame
            ) -> None:
        """
        Validate that DataFrames have sufficient data for training.
        
        Raises:
            ValueError: If validation fails
        """
        if df_mentors.empty:
            raise ValueError("No mentors found for this semester")
        
        if df_mentees.empty:
            raise ValueError("No mentees found for this semester")
        
        # Check 2:1 ratio
        required_mentees = len(df_mentors) * 2
        if len(df_mentees) < required_mentees:
            logger.warning(
                f" Insufficient mentees: Need {required_mentees}, got {len(df_mentees)}"
            )
        
        # Check required columns exist
        required_cols = {'applicant_id', 'ufl_email', 'full_name', 'role'}
        missing_mentor = required_cols - set(df_mentors.columns)
        missing_mentee = required_cols - set(df_mentees.columns)
        
        if missing_mentor or missing_mentee:
            raise ValueError(
                f"Missing required columns. "
                f"Mentors: {missing_mentor}, Mentees: {missing_mentee}"
            )

    def save_sbert_embeddings(
                    self, 
                    embeddings_data: List[Dict[str, any]]
                ) -> int:
        """
        Batch save computed S-BERT embeddings back to MongoDB.
                
        Args:
            embeddings_data: List of dicts with keys:
                - applicant_id: str (MongoDB ObjectId string)
                - embedding: List[float] (384-dim array)
        
        Returns:
            Number of successfully updated documents
            
        Example:
            >>> adapter.save_sbert_embeddings([
            ...     {"applicant_id": "67a1...", "embedding": [0.1, 0.2, ...]},
            ...     {"applicant_id": "67a2...", "embedding": [0.3, 0.4, ...]}
            ... ])
            2
        """
        logger.info(f"Saving {len(embeddings_data)} S-BERT embeddings...")
        
        updated_count = 0
        
        for data in embeddings_data:
            result = self.db.applicants.update_one(
                {
                    "_id": ObjectId(data["applicant_id"])
                },
                {
                    "$set": {
                        "applications.$[elem].embeddings.sbert_384": data["embedding"],
                        "applications.$[elem].embeddings.sbert_computed_at": datetime.datetime.now(datetime.UTC),
                        "applications.$[elem].embeddings.sbert_model_version": "all-MiniLM-L6-v2"
                    }
                },
                array_filters=[
                    {"elem.semester_id": self.current_semester_id}
                ]
            )
            
            if result.modified_count > 0:
                updated_count += 1
        
        logger.info(f"Updated {updated_count}/{len(embeddings_data)} embeddings")
        
        return updated_count
    
    def save_learned_embeddings(
                    self, 
                    embeddings_data: List[Dict[str, any]],
                    model_checkpoint: str
                ) -> int:
        """
        Batch save learned 64-dim embeddings from TwoTowerModel.
        
        This implements generate_mentor_embedding step storage.
        
        Args:
            embeddings_data: List of dicts with keys:
                - applicant_id: str
                - embedding: List[float] (64-dim array)
            model_checkpoint: Path to saved model weights
        
        Returns:
            Number of successfully updated documents
        """
        logger.info(f" Saving {len(embeddings_data)} learned embeddings...")
        
        updated_count = 0
        
        for data in embeddings_data:
            result = self.db.applicants.update_one(
                {
                    "_id": ObjectId(data["applicant_id"])
                },
                {
                    "$set": {
                        "applications.$[elem].embeddings.learned_64": data["embedding"],
                        "applications.$[elem].embeddings.learned_computed_at": datetime.datetime.now(datetime.UTC),
                        "applications.$[elem].embeddings.learned_model_checkpoint": model_checkpoint
                    }
                },
                array_filters=[
                    {"elem.semester_id": self.current_semester_id}
                ]
            )
            
            if result.modified_count > 0:
                updated_count += 1
        
        logger.info(f"Updated {updated_count}/{len(embeddings_data)} learned embeddings")
        
        return updated_count
