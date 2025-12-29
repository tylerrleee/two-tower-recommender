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

class End2EndMatching:
    def __init__(
            self,
            data_path: str,
            sbert_pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            use_pretrained_model: bool = False,
            model_checkpoint_path: str = None
            ):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to CSV file with applicant data
            sbert_model: Sentence transformer model name
            use_pretrained_model: Whether to load a pre-trained two-tower model
            model_checkpoint_path: Path to saved model weights
        """
        self.data_path = data_path
        self.sbert_pretrained_model = sbert_pretrained_model 
        self.use_pretrained_model = use_pretrained_model
        self.model_checkpoint_path = model_checkpoint_path

        # COMPONENTS
        self.feature_engineer = None
        self.embedding_engineer = None
        
        
        