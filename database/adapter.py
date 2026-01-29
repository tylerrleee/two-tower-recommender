from typing import Tuple
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

class MondogDBAdapter:
    def __init__(self, connection_string: str):
        self.client = MongoClient(connection_string)
        self.db     = self.client.two_tower
    
    def load_semester_data(
                        self, 
                        semester_id: str
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            
        semester_oid = ObjectId(semester_id)
        pipeline = [
                    # Filter to the semester_iod (index)
                    {"$match": {"applications.semester_id": semester_oid}},
                    
                    # Project only the specific application needed 
                    {"$project": {
                        "ufl_email": 1,
                        "full_name": 1,
                        "relevant_app": {
                            "$filter": {
                                "input": "$applications",
                                "as": "app",
                                "cond": { "$eq": ["$$app.semester_id", semester_oid] }
                            }
                        }
                    }},

                    {"$unwind": "$relevant_app"},
                    
                    # 4. Final Projection to flatten
                    {"$project": {
                        "applicant_id": {"$toString": "$_id"},
                        "ufl_email": 1,
                        "full_name": 1,
                        "role": "$relevant_app.role",  # Applying the fix from Point 1
                        "survey_responses": "$relevant_app.survey_responses",
                        # ... rest of fields
                    }}
                ]       
        results = list(self.db.applicants.aggregate(pipeline))
        
        flattened_data = []
        for record in results:
            flat_record = {
                "applicant_id"  : record["applicant_id"],
                "ufl_email"     : record["ufl_email"],
                "full_name"     : record["full_name"],
                "role"          : record["role"],
                "sbert_embedding": record.get("sbert_embedding"),
                "learned_embedding": record.get("learned_embedding"),
                **record["survey_responses"]  # Unpack dynamic fields
            }
            flattened_data.append(flat_record)
        
        df_all = pd.DataFrame(flattened_data)
        
        df_mentors = df_all[df_all["role"] == "mentor"].copy()
        df_mentees = df_all[df_all["role"] == "mentee"].copy()
        
        # Apply column renaming using config.RENAME_MAP
        
        return df_mentors, df_mentees