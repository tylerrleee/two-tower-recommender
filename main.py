"""
Docstring for two-tower-recommender.main

1. Load Data (CSV format)
2. Engineer Features - standardize inputs
3. Generate Embeddings
4. Initialize model
5. Train Model
6. Match Groups 
7. Formatting for readability
"""

import numpy as np
import pandas as pd


from feature_engineer import FeatureEngineer
import config

class End2EndMatching:
    def __init__(
            self,
            data_path: str,
            sbert_pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            use_pretrained_model: bool  = False,
            model_checkpoint_path: str  = None
            ):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to CSV file with applicant data
            sbert_model: Sentence transformer model name
            use_pretrained_model: Whether to load a pre-trained two-tower model
            model_checkpoint_path: Path to saved model weights
        """
        self.data_path              = data_path
        self.sbert_pretrained_model = sbert_pretrained_model 
        self.use_pretrained_model   = use_pretrained_model
        self.model_checkpoint_path  = model_checkpoint_path

        # COMPONENTS
        self.feature_engineer   = None
        self.embedding_engineer = None
        self.model              = None
        self.matcher            = None

        # STORAGE
        self.df                 = None
        self.mentor_data        = None
        self.mentee_data        = None
        self.mentor_embeddings  = None
        self.mentee_embedding   = None

        #TODO cmd print when initialized
    
    def load_csv_data(self):
        """
        Load CSV data and split into mentors and mentees
        Assuming there should be 2 mentees for every 1 mentor
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} total applicants")

        # TODO change logic for classifying bigs and little
        self.df_mentors = self.df[self.df['Role (0=Big,1=Little)'] == 0].copy()
        self.df_mentees = self.df[self.df['Role (0=Big,1=Little)'] == 1].copy()

        print(f"Mentors: {len(self.df_mentors)}")
        print(f"Mentees: {len(self.df_mentees)}") 

        if len(self.df_mentees) != 2 * len(self.df_mentors):
            print(f"⚠ Warning: Ideal ratio is 2 mentees per mentor")
            print(f"  Current ratio: {len(self.df_mentees) / len(self.df_mentors):.2f}:1")
        
        return self
    
    def load_mentee_embeddings(self, mentee_emb: pd.DataFrame):
        self.df_mentees = mentee_emb
        return self
    
    def load_mentor_embeddings(self, mentor_emb: pd.DataFrame):
        self.df_mentors = mentor_emb
        return self
    
    def engineer_features(self):
        """
        Engineer Features by 

            1. Standardize all columns w/ rename
            2. Clean all entries w/ trailing white space, string types, and NaN values
            3. Clean profile text
        
        Performing:
            1. OneHotEncoding
            2. Standard Scaler
            3. Column Transformer
        """
        self.feature_engineer = FeatureEngineer(
            profile_text        = config.DEFAULT_PROFILE_TEXT,
            categorical_fields  = config.DEFAULT_CATEGORICALS,
            numeric_fields      = config.DEFAULT_NUMERICS
        )

        # Encoding, Scaling
        self.feature_engineer.fit(df = self.df, rename_map=config.RENAME_MAP)

        #




        




        
        