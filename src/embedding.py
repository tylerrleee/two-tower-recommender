from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import Tuple, Optional, List
import faiss
from scipy.optimize import linear_sum_assignment
import logging
import pandas as pd
from database.adapter import DataAdapter

logger = logging.getLogger(__name__)

class EmbeddingEngineer:
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
                self.index = None
                
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
        embeddings = self.model.encode(
            sentences  = texts,
            batch_size = self.batch_size,
            show_progress_bar = show_progress_bar,
            convert_to_numpy = True,
            normalize_embeddings = True # L2 Norm text embeddings
        )
        return embeddings
    
    def combine_features(self, text_features: np.ndarray, meta_features: np.ndarray):
        """
        Concatenate text embeddings with meta_features | 
        Args:
            text_embeddings: Our encoded corpus embeddings from embed_texts
            meta_features  : Features from FeatureEngineer to characterize the target user
        Return:
            A combined array of both embedding and features, ready to run similarity search
        """

        # Check if meta feature exists
        if meta_features is None or meta_features.size == 0:
            raise ValueError("Meta Feature does not exist or Empty")
        if text_features is None or text_features.size == 0:
            raise ValueError("Text Embeddings does not exist or Empty")
        
        # Mismatched entries amount
        if text_features.shape[0] != meta_features.shape[0]:
             raise ValueError("Mismatched entries between text embedding & meta features!")
        


        print("Encoding Text Features...")
        # Case when embed_texts is not yet called
        # ignore when text features is None
        if len(text_features) > 0 and isinstance(text_features[0], str):
            print("Encoding Text Features...")
            text_features = self.embed_texts(texts = text_features, show_progress_bar=True)

        print("Combining text embeddings & meta features...")



        # Cast precision type
        EMBEDDED_TEXT       = text_features.astype(self.precision)
        EMBEDDED_META       = meta_features.astype(self.precision)
        COMBINED_DATA       = np.concatenate((EMBEDDED_TEXT, EMBEDDED_META), axis=1)
        return COMBINED_DATA
    
    
    # Save Embedding
    def save_embedding(self, embeddings: np.ndarray, path: str, filename: str = "embeddings.npy"):
        print(f"Saving Embeddings at {path}/{filename} ...")
        os.makedirs(name     = path, 
                    exist_ok = True)
        
        np.save(file = os.path.join(path, filename), 
                arr  = embeddings)
        
    # Load Embedding
    def load_embedding(self, path: str, file: str = "embeddings.npy"):
        return np.load(os.path.join(path,file))
    


    # Build Faiss Index for Cosine Similarity
    def build_faiss_index(self, combined_embeddings: np.ndarray):
        """
        Builds a Flat Inner Product index for Cosine Similarity search

        Note: OpemMP threads causes segmentation fault
         - FAISS compiles with IntelOpenMP, whereas the local system (Apple) uses a different process
         - Hence, causes runtime conflict, then segfault

        IndexFlatIP:
        - Why don't we normalize the embeddings?
        1. The metadata is alredy normalized in features.py using StandardScaler
        2. The text_data is normalized through embed_text() using S-BERT model
        """
        # get dimensions
        #combined_embeddings_norm = faiss.normalize_L2(combined_embeddings)
        dimensions = combined_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimensions)

        # Add vector to index
        self.index.add(combined_embeddings.astype('float32'))
        print(f"FAISS index built with {self.index.ntotal} vectors!")
    # Query Faiss (need normalization)
    def query_index(self, query_vector: np.ndarray, top_k:int = 4):
         """
         Find the top_k most similar vectors to query
         """
         if not hasattr(self.index, 'search'):
            raise RuntimeError("Index not built. Call build_faiss_index first.")     
         if self.index is None:
            raise RuntimeError("Index not built. Call build_faiss_index first.")   
         
         query_vector = query_vector.astype('float32')    

         distances, indices = self.index.search(query_vector, top_k)
         
         return distances, indices
    
    def generate_embeddings_with_cache(
            mentor_data: pd.DataFrame,
            mentee_data: pd.DataFrame,
            db_adapter: DataAdapter
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings with MongoDB caching (Step 3 with cache).
        
        Flow:
        1. Check if S-BERT embeddings exist in DataFrame (from DB)
        2. Compute missing embeddings
        3. Save new embeddings back to MongoDB
        4. Concatenate text + meta features
        """
        ...