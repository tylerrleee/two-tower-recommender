from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import Optional, Dict, Any, List
import faiss
import joblib

class EmbeddingPipeline:
    def __init__(self,
                 sbert_model_name: str = "",
                 embedding_batch_size: int = 64,
                 use_gpu: bool = False,
                 precision: Optional[np.dtype] = np.float32
            ):  
                self.model_name = sbert_model_name
                self.batch_size = embedding_batch_size
                self._model = None
                self.device = 'cuda' if use_gpu else 'cpu'
                self.precision = precision 
                
    # Define our model 
    @property
    def model(self):
        """
        Load our SentenceTransformer Model with validation and error catching
        """
        try:
            print("Loading Sentence Transformer Model ...")
            if self._model is None:
                self._model = SentenceTransformer(self.model_name, 
                                                device=self.device)
        except TypeError:
              print(" SentenceTransforemr Model do not exist: Insert another.")
        return self._model
    # Embedd text -- profile text
    def embed_texts(self, texts: List[str], show_progress_bar: bool = True):
        """
        Calculate embeddings through :
            1. Tokenization: break text into sub words
            2. tokens are flatten and run through layers
            3. Pooling: single vector for each sentence/text
            4. Normalize vector for cosine similarity or dot prod
        Args:
            texts: corpus with semantic meaning, usually called profile_text
            show_progress_bar: show embedding progress bar
        Return:
            np.ndarray of shape (len(texts), embed_dim) 
                - e.g. [3, 384] means 3 entries with 384 dimensions
        """
        print("Encoding Corpus...")
        embeddings = self.model.encode(
            sentences  = texts,
            batch_size = self.batch_size,
            show_progress_bar = show_progress_bar,
            convert_to_numpy = True,
            normalize_embeddings = True # L2 Norm text embeddings
        )
        return embeddings
    
    def combine_features(self, text_embeddings: np.ndarray, meta_features: np.ndarray):
        """
        Concatenate text embeddings with meta_features | 
        Args:
            text_embeddings: Our encoded corpus embeddings from embed_texts
            meta_features  : Features from FeatureEngineer to characterize the target user
        Return:
            A combined array of both embedding and features, ready to run similarity search
        """
        print("Combining text embeddings & meta features...")
        # Check if meta feature exists
        if meta_features is None or meta_features.size == 0:
             return text_embeddings
        # check meta features does not match text embeddings
        if text_embeddings.shape[0] != meta_features.shape[0]:
             raise ValueError("Mismatched entries between text embedding & meta features!")
        
        # Cast precision type
        text_embed = text_embeddings.astype(self.precision)
        meta       = meta.astype(self.precision)
        combined   = np.concatenate(arrays=[text_embed, meta], axis=1)
        return combined
    
    
    # Save Embedding
    def save_embedding(self, embeddings: np.ndarray, path: str, filename: str = "embeddings.npy");
        print(f"Saving Embeddings at {path}/{filename} ...")
        os.makedirs(name     = path, 
                    exist_ok = True)
        
        np.save(file = os.path.join(path, filename), 
                arr  = embeddings)
        
    # Load Embedding
    def load_embedding(self, path: str, file: str = "embeddings.npy"):
        return np.load(os.path.join(path,file))
    

    # Build Faiss Index for Cosine Similarity

    # Query Faiss (need normalization)

    # Save model metadata & pipeline config
        