import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path

from src.features import FeatureEngineer
from src.embedding import EmbeddingEngineer
from src.model import TwoTowerModel
from src.matcher import GroupMatcher
import config

logger = logging.getLogger(__name__)

class MatchingIneference:
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
        self.embedding_eingeer  : Optional[EmbeddingEngineer]   = None
        self.matcher            : Optional[GroupMatcher]        = None

        self.model_metadata     : Optional[GroupMatcher]        = None


    def load_model(self) -> None:
        """ Load trained model from checkpoint"""

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
                'epoch': checkpoint.get('epoch', 'unknown'),
                'train_loss': checkpoint.get('train_loss', None),
                'val_loss': checkpoint.get('val_loss', None),
                'loss_type': checkpoint.get('loss_type', 'unknown')
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
            state_dict = checkpoint['model_state_dict']
            first_layer_weight = state_dict['mentor_tower.0.weight']
            input_dim = first_layer_weight.shape[1]
            meta_feature_dim = input_dim - self.embedding_dim

            logger.info(f"Inferred input_dim={input_dim}, meta_feature_dim={meta_feature_dim}")

            self.model = TwoTowerModel(
                embedding_dim       = self.embedding_dim,
                meta_feature_dim    = meta_feature_dim,
                tower_hidden_dim    = [256, 128, 64],
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

            logger.info("Model loaded done.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
