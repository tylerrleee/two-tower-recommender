import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path
from functools import lru_cache
import hashlib
import datetime
import re

from src.features import FeatureEngineer
from src.embedding import EmbeddingEngineer
from src.model import TwoTowerModel
from src.matcher import GroupMatcher
import config


logger = logging.getLogger(__name__)

class MatchingInference:
    """ Wrapper for matching in production 
    
        1. Load model
        2. Load data
        3. engineer features
        4. engineer embeddings
        5. Fit transform into Model
        6. Generate matched groups 
        7. Format groups and metadata
    
    """
    def __init__(
            self,
            model_path  : str,
            sbert_model : str = "sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim : int = 384,
            device: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.sbert_model = sbert_model
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model              : Optional[TwoTowerModel]       = None
        self.feature_engineer   : Optional[FeatureEngineer]     = None
        self.embedding_engineer : Optional[EmbeddingEngineer]   = None
        self.matcher            : Optional[GroupMatcher]        = None

        self.model_metadata     : Optional[GroupMatcher]        = None
        self._preprocessing_cache                                = {}

    def load_model(self) -> None:
        """ Load trained model from checkpoint
        
        checkpoint format:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'loss_type': loss_type
                    }
        Initiates:
        - model_metadata : 'epoch' , 'train_loss' , 'val_loss', 'loss_type'
        - self.feature_engingeer : FeatureEngineer
        - self.embedding_engineer : Embedding Engineer
        - self.model : TwoTowerModel
        - self.matcher : GroupMatcher
        """

        try:
            logger.info(f"Loading model from {self.model_path}")

            # Load check point
            checkpoint = torch.load(
                self.model_path,
                map_location = torch.device(self.device),
                weights_only = False
            )

            # Extract metadata
            self.model_metadata = {
                'epoch'     : checkpoint.get('epoch', 'unknown'),
                'train_loss': checkpoint.get('train_loss', None),
                'val_loss'  : checkpoint.get('val_loss', None),
                'loss_type' : checkpoint.get('loss_type', 'unknown')
            }

            logger.info(f"Model metadata: {self.model_metadata}")

            self.feature_engineer = FeatureEngineer(
                profile_text        = config.DEFAULT_PROFILE_TEXT,
                categorical_fields  = config.DEFAULT_CATEGORICALS,
                numeric_fields      = config.DEFAULT_NUMERICS
            )

            self.embedding_engineer = EmbeddingEngineer(
                sbert_model_name        = self.sbert_model,
                embedding_batch_size    = 64,
                use_gpu                 = (self.device == 'cuda')
            )

            # Extract first layer weight
            state_dict          = checkpoint['model_state_dict']
            first_layer_weight  = state_dict['mentor_tower.0.weight']

            input_dim           = first_layer_weight.shape[1]
            meta_feature_dim    = input_dim - self.embedding_dim

            logger.info(f"Inferred input_dim={input_dim}, meta_feature_dim={meta_feature_dim}")

            self.model = TwoTowerModel(
                embedding_dim       = self.embedding_dim,
                meta_feature_dim    = meta_feature_dim,
                tower_hidden_dims    = [256, 128, 64],
                dropout_rate        = 0.3
            )

            self.model.load_state_dict(state_dict = state_dict)
            self.model.to(device = self.device)
            self.model.eval()

            self.matcher = GroupMatcher(
                model                = self.model,
                compatibility_weight = 0.6,
                diversity_weight     = 0.4
            )

            self.model_metadata.update({
            'model_path': str(self.model_path),
            'loaded_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'version': self._infer_model_version()
        })
            
            logger.info("Model loaded done.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _infer_model_version(self) -> str:
        """Extract model version from checkpoint"""
        # Check if model path contains version
        # e.g., "models/best_model_v2.1.0.pt"

        match = re.search(r'v(\d+\.\d+\.\d+)', str(self.model_path))
        if match:
            return match.group(1)
        
        # Fallback: Use training date
        return f"1.0.{self.model_metadata.get('epoch', 0)}"

    def _compute_df_hash(self, df: pd.DataFrame) -> str:
        """ Generate hash of DataFrame for cache key"""
        # Calculate hash values for dataframe | each hashed int is a series/df_row
        hashed_binary_of_df = pd.util.hash_pandas_object(df, index = True)

        # condense binary -- reduce size
        condensed_binary = hashlib.md5(hashed_binary_of_df.values) # return binary

        # Convert binary into hexadecimal string (e.g. "5d41402abc4b2a76b9719d911017c592")
        hex_string_representation = condensed_binary.hexdigest()
        return hex_string_representation
    

    def preprocess_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """ Preprocess applicant data into embeddings """
        try:
            
            # check cache
            df_hash = self._compute_df_hash(df)

            if df_hash in self._preprocessing_cache:
                logger.info("Using cache preprocessing result")
                return self._preprocessing_cache[df_hash]

            self.feature_engineer.fit(df = df,
                                      rename_map = config.RENAME_MAP)
            df_mentors = df[df['role'] == 0].copy()
            df_mentees = df[df['role'] == 1].copy()

            mentor_data = self.feature_engineer.transform(df_mentors)
            mentee_data = self.feature_engineer.transform(df_mentees)

            mentor_emb = self.embedding_engineer.combine_features(
                text_features = mentor_data['profile_text'],
                meta_features = mentor_data['meta_features']
            )

            mentee_emb = self.embedding_engineer.combine_features(
                text_features = mentee_data['profile_text'],
                meta_features = mentee_data['meta_features']
            )
            result =  {
                'mentor_embeddings' : mentor_emb,
                'mentee_embeddings' : mentee_emb,
                'df_mentors'        : df_mentors,
                'df_mentees'        : df_mentees
            }
            
            # Cache Result
            if len(self._preprocessing_cache) > 10: # 10 is placeholder for now TODO
                self._preprocessing_cache.clear()  
        
            self._preprocessing_cache[df_hash] = result
            return result
        except Exception as e:
            logger.error(f"Preprocessing failed: {(e)}")
            raise
    
    def generate_learned_embeddings(
                self, 
                mentor_emb: np.ndarray, 
                mentee_emb: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    
        """Generate learned embeddings from raw features"""

        self.model.eval()
        
        with torch.no_grad():
            mentor_tensor = torch.FloatTensor(mentor_emb).to(self.device)
            mentee_tensor = torch.FloatTensor(mentee_emb).to(self.device)
            
            mentor_learned = self.model.get_mentor_embedding(mentor_tensor)
            mentee_learned = self.model.get_mentee_embedding(mentee_tensor)
            
            return mentor_learned.cpu().numpy(), mentee_learned.cpu().numpy()
    
    def match(
            self, 
            df: pd.DataFrame, 
            use_faiss: bool = False, 
            top_k: int = 10
        ) -> Dict:

        """ Matching pipeline"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first!")
        
        processed = self.preprocess_data(df)
        
        mentor_learned, mentee_learned = self.generate_learned_embeddings(
            processed['mentor_embeddings'],
            processed['mentee_embeddings']
        )
        
        # Match groups
        if use_faiss:
            groups = self.matcher.find_best_groups_faiss(
                mentor_emb=mentor_learned,
                mentee_emb=mentee_learned,
                top_k=top_k
            )
        else:
            groups = self.matcher.find_best_groups_base(
                mentor_emb=mentor_learned,
                mentee_emb=mentee_learned
            )
        
        # Format results
        results = self._format_results(
            groups, 
            processed['df_mentors'], 
            processed['df_mentees']
        )
        
        return results

    def _format_results(
                self, 
                groups: Dict, 
                df_mentors: pd.DataFrame, 
                df_mentees: pd.DataFrame
            ) -> Dict:

            """Format matching results for API response"""
            
            formatted_groups = []
            
            for mentor_idx, group_info in groups.items():
                mentor_row = df_mentors.iloc[mentor_idx]
                
                mentees_info = []
                for mentee_idx in group_info['mentees']:
                    mentee_row = df_mentees.iloc[mentee_idx]
                    mentees_info.append({
                        'name': mentee_row['name'],
                        'major': mentee_row['major'],
                        'year': mentee_row['year']
                    })
                
                avg_score = float(
                    group_info['total_compatibility_score'] / len(group_info['mentees'])
                )
                
                formatted_groups.append({
                    'group_id': int(mentor_idx),
                    'mentor': {
                        'name': mentor_row['name'],
                        'major': mentor_row['major'],
                        'email': mentor_row['ufl_email']
                    },
                    'mentees': mentees_info,
                    'compatibility_score': avg_score,
                    'individual_scores': [float(s) for s in group_info['individual_scores']]
                })
            
            # Sort by compatibility score
            formatted_groups.sort(key=lambda x: x['compatibility_score'], reverse=True)
            
            avg_compatibility = np.mean([g['compatibility_score'] for g in formatted_groups])
            
            return {
                'status'        : 'success',
                'total_groups'  : len(formatted_groups),
                'average_compatibility': float(avg_compatibility),
                'groups'        : formatted_groups
            }
