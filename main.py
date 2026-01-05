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
import torch

from features import FeatureEngineer
from embedding import EmbeddingEngineer
from model import TwoTowerModel
from train import MentorMenteeDataset, train_epoch
from matcher import GroupMatcher
from loss import DiversityLoss
import config

class End2EndMatching:
    def __init__(
            self,
            data_path: str,
            sbert_pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            use_pretrained_model: bool  = False,
            embedding_dimensions:       int = None,
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
        self.embedding_dimensions   = embedding_dimensions

        # COMPONENTS
        self.feature_engineer   = None
        self.embedding_engineer = None
        self.model              = None
        self.matcher            = None

        # STORAGE
        self.df                 = None
        self.mentor_data        = None
        self.mentee_data        = None
        self.mentor_embedding   = None
        self.mentee_embedding   = None
        self.mentee_embeddings_learned = None
        self.mentor_embeddings_learned = None

        # MATCHING
        self.groups             = None

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

        # Ratio Check
        if len(self.df_mentees) != 2 * len(self.df_mentors):
            print(f"Current ratio: {len(self.df_mentees) / len(self.df_mentors):.2f}:1")
        
        return self
    
    def load_mentee_embeddings(self, mentee_emb: pd.DataFrame):
        self.df_mentees = mentee_emb
        return self
    
    def load_mentor_embedding(self, mentor_emb: pd.DataFrame):
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
        self.feature_engineer   = FeatureEngineer(
            profile_text        = config.DEFAULT_PROFILE_TEXT,
            categorical_fields  = config.DEFAULT_CATEGORICALS,
            numeric_fields      = config.DEFAULT_NUMERICS
        )

        # Encoding, Scaling on ALL data
        self.feature_engineer.fit(df = self.df, rename_map=config.RENAME_MAP)

        # Transform for each role
        self.mentee_data = self.feature_engineer.transform(self.df_mentees)
        print(f"Mentor profile texts: {self.mentor_data['profile_text'].shape}")
        print(f"Mentor meta features: {self.mentor_data['meta_features'].shape}")

        self.mentee_data = self.feature_engineer.transform(self.df_mentees)
        print(f"Mentee profile texts: {self.mentee_data['profile_text'].shape}")
        print(f"Mentee meta features: {self.mentee_data['meta_features'].shape}")

        return self
    
    def generate_embeddings(self):

        self.embedding_engineer = EmbeddingEngineer(
            sbert_model_name    =self.sbert_pretrained_model,
            embedding_batch_size=64,
            use_gpu             =torch.cuda.is_available()
        )

        # Generate embeddings
        self.mentor_embedding = self.embedding_engineer.combine_features(
            text_features=self.mentor_data['profile_text'],
            meta_features=self.mentor_data['meta_features']
        )
        print(f"Mentor embeddings shape: {self.mentor_embedding.shape}")


        self.mentee_embedding = self.embedding_engineer.combine_features(
            text_features=self.mentee_data['profile_text'],
            meta_features=self.mentee_data['meta_features']
        )
        print(f"Mentee embeddings shape: {self.mentee_embeddings.shape}")

        return self

    def initialize_model(self):
        meta_feature_dim = self.mentor_embedding.shape[1] - self.embedding_dimensions

        print(f"Text embedding dim: {self.embedding_dimensions}")
        print(f"Meta feature dim: {meta_feature_dim}")

        self.model = TwoTowerModel(
            embedding_dim=self.embedding_dimensions,
            meta_feature_dim=meta_feature_dim,
            tower_hidden_dims = [256, 128, 64],
            dropout_rate = 0.3
        )

        # Load Pretrained weights if true
        if self.use_pretrained_model and self.model_checkpoint_path:
            print(f"Loading pretrained model from: {self.model_checkpoint_path}")
            self.model.load_state_dict(torch.load(self.model_checkpoint_path))
        # TODO What to do when we don't use a pretrained model?

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        

        return self

    def train_model(self, num_epochs: int, batch_size: int, learning_rate: float):
        """
        Training the model

        Args:
            num_epochs: Number of training epochs
            batch_sizes: batch size for training
            learning_rate: Learning rate for optimizer
        
        """
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get synthetic positive pairs
        n_mentors    = len(self.mentor_embedding)
        n_mentees    = len(self.mentee_embedding)
        n_pairs      = min(n_mentees, n_mentors)

        pos_pairs    = np.arange(n_pairs)
        
        # Compute diversity features 
        mentee_diversity = self.feature_engineer.compute_diversity_features(
            df = self.df_mentees.head(n_pairs) # Compute for n_pairs of mentees
        )

        mentor_diversity = np.zeros((n_pairs, mentee_diversity.shape[1])) # Dummy
 
        # Create dataset 
        dataset = MentorMenteeDataset(
            mentee_features     =   self.mentee_embedding[:n_pairs],
            mentor_features     =   self.mentor_embedding[:n_pairs],
            mentee_diversity    =   mentee_diversity, 
            mentor_diversity    =   mentor_diversity,
            positive_pairs      =   pos_pairs
        )

        dataLoader = torch.utils.data.DataLoader(dataset = dataset,  
                                    batch_size=batch_size,
                                    shuffle=True) 
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr = learning_rate)
        criterion = DiversityLoss(
            compatibility_weight=0.7,
            diversity_weight=0.3,
            temperature=0.1
        )

        print("Starting training...")
        # Batch Norm updated running statistics + Dropout layers perform random drop out on input 
        self.model.train() 

        for epoch in range(num_epochs):
            avg_loss = train_epoch(model        = self.model, 
                                   dataloader   = dataLoader,
                                   optimizer    = optimizer,
                                   criterion    = criterion,
                                   device       = device)
            # We could use 
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        print("Training completed!")

        save_path = f"model_checkpoint_epoch{num_epochs}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

        return self

    def generate_mentor_embeddings(self):
        """
        Generate learned embeddings for all mentors using train model
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

        # turn off gradianet computation
        with torch.no_grad():
            # Convert to sentors
            mentor_tensor = torch.FloatTensor(self.mentor_embedding).to(device)
            mentee_tensor = torch.FloatTensor(self.mentee_embedding).to(device)

            # Get learned embeddings
            learned_mentor_embedding = self.model.get_mentor_embedding(mentor_tensor)
            learned_mentee_embedding = self.model.get_mentee_embedding(mentee_tensor)

            # Numpy Array conversion + create new variable
            self.mentor_embeddings_learned = learned_mentor_embedding.cpu().numpy()
            self.mentee_embeddings_learned = learned_mentee_embedding.cpu().numpy()
        
        print(f"Learned mentor embeddings: {self.mentor_embeddings_learned.shape}")
        print(f"Learned mentee embeddings: {self.mentee_embeddings_learned.shape}")
        
        return self

    def match_groups(self, use_faiss: bool = False, top_k: int = 10):
        """
        Generate matched groups
        
        Args:
            use_faiss: FAISS accelerated matching -- sacrifise precision for effiency
            top_k: Number of candidates for FAISS
        
        """

        # If model was trained, otherwise use raw embeddings
        if hasattr(self, 'mentee_embeddings_learned'):
            mentor_emb = self.mentor_embeddings_learned
            mentee_emb = self.mentee_embeddings_learned
        else:
            mentor_emb = self.mentor_embedding
            mentee_emb = self.mentee_embedding

        # Initialize Matcher
        self.matcher = GroupMatcher(
            model=self.model,
            compatibility_weight=0.6,
            diversity_weight=0.4
        )

        # Choose matching algo
        if use_faiss:
            self.groups = self.matcher.find_best_groups_faiss(
                mentee_emb  =   mentor_emb,
                mentee_emb  =   mentee_emb,
                top_k       =   top_k
            )
        else:
            self.groups = self.matcher.find_best_groups_base(
                mentor_emb  =   mentor_emb,
                mentee_emb  =   mentee_emb
            )
        print(f"Created {len(self.groups)} mentor-mentee groups")
        return self
def main():
    """
    TODO
    1. Load data 
    2. Add features | standardize names, norm, scaling
    3. Generate embeddings | remove filler words, vectorize corpus, standardize dimensions
    4. Initialize model | with & w/o FAISS index for tests
    5. Training pipeline (optional)
    """
        

# TODO import logging and replace all print statements - 
# TODO Test at different DiversityLoss weights - training
# TODO Test parallel loading w/ num_workers - training

        




        
        