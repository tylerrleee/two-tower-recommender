"""
Docstring for two-tower-recommender.main

1. Load Data (CSV format)
2. Engineer Features - standardize inputs
3. Generate Embeddings
4. Initialize model
5. Train Model
6. Match Groups 
7. Formatting for readability

Note:
- return self on methods without a return value for chaining purposes
- Type hinting follows the PEP 589 - TypeDict | https://peps.python.org/pep-0589/
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
from scipy.optimize import linear_sum_assignment
from typing import Optional, Dict, List, Any, TypedDict, Tuple

from src.features      import FeatureEngineer
from src.embedding    import EmbeddingEngineer
from src.boostrap_positive_pairs    import bootstrap_topk_pairs_from_embeddings
from src.model            import TwoTowerModel
from src.train            import MentorMenteeDataset, train_epoch, train_model_with_validation, create_mentor_level_split
from src.matcher        import GroupMatcher
from src.loss              import DiversityLoss
from src.pairwise_margin_loss import PairwiseMarginLoss
from src.saving_csv import *

import config
import traceback

# Type definitions for structured data
class TransformedFeatures(TypedDict):
    """Output from FeatureEngineer.transform()"""
    profile_text: npt.NDArray[np.str_]
    meta_features: npt.NDArray[np.float64]
    index: npt.NDArray[np.int64]
    raw_df: pd.DataFrame


class GroupInfo(TypedDict):
    """Matching result for a single mentor-mentee group"""
    mentees: List[int]
    individual_scores: List[float]
    total_compatibility_score: float


class MentorInfo(TypedDict):
    """Mentor metadata for final output"""
    name: str
    major: str
    email: str


class MenteeInfo(TypedDict):
    """Mentee metadata for final output"""
    name: str
    major: str
    year: str


class MatchResult(TypedDict):
    """Final formatted matching result"""
    group_id: int
    mentor: MentorInfo
    mentees: List[MenteeInfo]
    compatibility_score: float
    individual_scores: List[float]


class TrainingHistory(TypedDict):
    """Training history from train_model_with_validation"""
    train_loss: List[float]
    val_loss: List[float]
    train_metrics: List[Dict[str, float]]
    val_metrics: List[Dict[str, float]]

class End2EndMatching:
    def __init__(
            self,
            data_path: str,
            sbert_pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            use_pretrained_model: bool  = False,
            embedding_dimensions:   int = 384,
            model_checkpoint_path: str  = None
            ) -> None:
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to CSV file with applicant data
            sbert_pretrained_model: Sentence transformer model name
            use_pretrained_model: Whether to load a pre-trained two-tower model
            embedding_dimensions: Dimension of S-BERT text embeddings (default: 384)
            model_checkpoint_path: Path to saved model weights (optional)
        """
        # Configuration
        self.data_path: str = data_path
        self.sbert_pretrained_model: str = sbert_pretrained_model 
        self.use_pretrained_model: bool = use_pretrained_model
        self.model_checkpoint_path: Optional[str] = model_checkpoint_path
        self.embedding_dimensions: int = embedding_dimensions

        # COMPONENTS
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.embedding_engineer: Optional[EmbeddingEngineer] = None
        self.model: Optional[TwoTowerModel] = None
        self.matcher: Optional[GroupMatcher] = None

        # STORAGE - Raw DataFrames
        self.df: Optional[pd.DataFrame] = None
        self.df_mentors: Optional[pd.DataFrame] = None
        self.df_mentees: Optional[pd.DataFrame] = None

        # STORAGE - Transformed Features
        self.mentor_data: Optional[TransformedFeatures] = None
        self.mentee_data: Optional[TransformedFeatures] = None

        # STORAGE - Combined Embeddings (S-BERT text + meta features)
        # Shape: (n_samples, embedding_dim + meta_feature_dim)
        self.mentor_embedding: Optional[npt.NDArray[np.float32]] = None
        self.mentee_embedding: Optional[npt.NDArray[np.float32]] = None    
        
        # STORAGE - Learned Embeddings (from Two-Tower model)
        # Shape: (n_samples, output_dim) where output_dim = 64
        self.mentor_embeddings_learned: Optional[npt.NDArray[np.float32]] = None
        self.mentee_embeddings_learned: Optional[npt.NDArray[np.float32]] = None 
        # MATCHING - Results
        self.groups: Optional[Dict[int, GroupInfo]] = None
        self.results: Optional[List[MatchResult]] = None

    
    def load_csv_data(self):
        """
        Load CSV data and split into mentors and mentees
        1. Check if required columns exists
        """
        self.df = pd.read_csv(self.data_path)
        self.df = FeatureEngineer.rename_column(self.df, config.RENAME_MAP)

        # Column Check
        #required_cols = {'role', 'name', 'major', 'year', 'ufl_email'}
        required_cols = config.DEFAULT_IDENTIFIER
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        
        # Validate role values (0=mentor, 1=mentee)
        if not self.df['role'].isin([0, 1]).all():
            raise ValueError("'role' column must contain only 0 (mentor) or 1 (mentee)")
        
        # Check for empty DataFrame
        if len(self.df) == 0:
            raise ValueError("Empty Dataframe - CSV file is empty")
        
        print(f"Loaded {len(self.df)} total applicants")

        self.df_mentors = self.df[self.df['role'] == 0].copy()
        self.df_mentees = self.df[self.df['role'] == 1].copy()

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
        self.mentor_data = self.feature_engineer.transform(self.df_mentors)
        print(f"Mentor profile texts: {self.mentor_data['profile_text'].shape}")
        print(f"Mentor meta features: {self.mentor_data['meta_features'].shape}")

        self.mentee_data = self.feature_engineer.transform(self.df_mentees)
        print(f"Mentee profile texts: {self.mentee_data['profile_text'].shape}")
        print(f"Mentee meta features: {self.mentee_data['meta_features'].shape}")

        return self
    
    def generate_embeddings(self):

        self.embedding_engineer = EmbeddingEngineer(
            sbert_model_name    = self.sbert_pretrained_model,
            embedding_batch_size= 64,
            use_gpu             = torch.cuda.is_available()
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
        print(f"Mentee embeddings shape: {self.mentee_embedding.shape}")

        return self

    def initialize_model(self):

        self.embedding_dimensions = self.embedding_dimensions

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
            
            # Load the full checkpoint
            checkpoint = torch.load(self.model_checkpoint_path, map_location=torch.device('cpu'))
            
            # Extract just the model weights
            if 'model_state_dict' in checkpoint:

                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Training loss: {checkpoint.get('train_loss', 'N/A')}")
                print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A')}")
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("No pretrained model found. Initializing with default random weights.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        return self



    def train_model(
            self,
            num_epochs          : int,
            batch_size          : int,
            learning_rate       : float,
            loss_type           : str = 'margin',
            use_multi_positive  : bool = True,
            k_positives         : int = 3,
            use_hard_negatives  : bool = True,
            val_split           : float = 0.2,
            early_stopping_patience: int = 5,
            checkpoint_path     : str = 'best_model.pt'
                ) -> Tuple['End2EndMatching', Dict[str, Any]]:
        '''
        Training the model with multi-positive bootstrap and hard negatives
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            loss_type: 'margin' or 'diversity'
            use_multi_positive: Use top-K positives per mentor (default: True)
            k_positives: Number of positive candidates per mentor (default: 3)
            use_hard_negatives: Include unused mentees as hard negatives (default: True)
            val_split: Validation split ratio (default: 0.2)
            early_stopping_patience: Patience for early stopping (default: 5)
            checkpoint_path: Path to save best model (default: 'best_model.pt')
        
        Returns:
            Tuple of (self, training_history)
        '''

        print(f"\n{'='*70}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        print(f"Loss type: {loss_type}")
        print(f"Multi-positive: {use_multi_positive} (k={k_positives if use_multi_positive else 1})")
        print(f"Hard negatives: {use_hard_negatives}")
        print(f"Validation split: {val_split}")
        print(f"{'='*70}\n")
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_mentors = len(self.mentor_embedding)
        n_mentees = len(self.mentee_embedding)
    
        print(f"Dataset Info:")
        print(f"  Mentors: {n_mentors}")
        print(f"  Mentees: {n_mentees}")

        # Step 1: Bootstrap positive pairs
        if use_multi_positive:
            pos_pairs, unused_mentees = bootstrap_topk_pairs_from_embeddings(
                mentor_embeddings=self.mentor_embedding,
                mentee_embeddings=self.mentee_embedding,
                k=k_positives,
                method='hungarian_topk'
            )
            print(f"  Multi-positive pairs shape: {pos_pairs.shape}")
            print(f"  Unused mentees: {len(unused_mentees)}")
        else: 
            similarity_matrix = np.dot(self.mentor_embedding, self.mentee_embedding.T)
            _, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
            pos_pairs = col_ind  # Shape: (n_mentors,)
            
            used_mentees = set(pos_pairs)
            all_mentees = set(range(n_mentees))
            unused_mentees = np.array(sorted(all_mentees - used_mentees), dtype=np.int64)
            print(f"  Single positive pairs: {len(pos_pairs)}")
            print(f"  Unused mentees: {len(unused_mentees)}")

        if loss_type == 'diversity':
            mentor_diversity = self.feature_engineer.compute_diversity_features(
                self.mentor_data['raw_df']
            )
            mentee_diversity = self.feature_engineer.compute_diversity_features(
                self.mentee_data['raw_df']
            )
            print(f"  Diversity features: {mentor_diversity.shape}")
        elif loss_type == 'margin':
            # Dummy diversity features for margin loss
            mentor_diversity = np.zeros((n_mentors, 1), dtype=np.float32)
            mentee_diversity = np.zeros((n_mentees, 1), dtype=np.float32)

            # Step 3: Prepare hard negative pool
        if use_hard_negatives and len(unused_mentees) > 0:
            hard_negative_pool = self.mentee_embedding[unused_mentees]
            hard_negative_diversity = mentee_diversity[unused_mentees]
            print(f"  Hard negative pool: {hard_negative_pool.shape}")
        else:
            hard_negative_pool = None
            hard_negative_diversity = None   

            # Step 4: Create mentor-level train/val split
        train_mentor_indices, val_mentor_indices = create_mentor_level_split(
            n_mentors=n_mentors,
            train_ratio=1.0 - val_split,
            random_seed=42
        )
        
            # Step 5: Create datasets
        full_dataset = MentorMenteeDataset(
            mentor_features=self.mentor_embedding,
            mentee_features=self.mentee_embedding,
            mentor_diversity=mentor_diversity,
            mentee_diversity=mentee_diversity,
            positive_pairs=pos_pairs,
            hard_negative_pool=hard_negative_pool,
            hard_negative_diversity=hard_negative_diversity
        )

        train_dataset = Subset(full_dataset, train_mentor_indices)
        val_dataset = Subset(full_dataset, val_mentor_indices)

        print(f"\nDataset split:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Step 6: Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        # Step 7: Initialize loss function
        if loss_type == 'diversity':
            criterion = DiversityLoss(
                compatibility_weight=0.7,
                diversity_weight=0.3,
                temperature=0.1
            )
        elif loss_type == 'margin':
            criterion = PairwiseMarginLoss(
                margin=0.2,
                similarity='cosine'
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Step 8: Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization
        )

        # Step 9: Train model
        print(f"\nStarting training...")
        history = train_model_with_validation(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            loss_type=loss_type,
            early_stopping_patience=early_stopping_patience,
            max_grad_norm=1.0,
            checkpoint_path=checkpoint_path,
            use_hard_negatives=use_hard_negatives
        )

        print(f"\n{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Best epoch: {history['best_epoch'] + 1}")
        print(f"Best val loss: {min(history['val_loss']):.4f}")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Early stopping: {'Yes' if history['stopped_early'] else 'No'}")
        print(f"{'='*70}\n")

        return self, history


        
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
        if self.mentee_embeddings_learned is not None and self.mentor_embeddings_learned is not None:
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

        # Matching
        assert mentor_emb is not None
        assert mentee_emb is not None
        assert mentor_emb.ndim == 2
        assert mentee_emb.ndim == 2

        if use_faiss:
            self.groups = self.matcher.find_best_groups_faiss(
                mentor_emb  =   mentor_emb,
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

    def set_output_result(self):
        """
        Format groups in readable format
        """

        # Check if our target columns exist
        expected_cols = {'name', 'major', 'year', 'ufl_email'}
        missing = expected_cols - set(self.df_mentors.columns)
        #print(self.df_mentors.columns)
        assert not missing, f"Missing columns: {missing}"


        results = []

        for mentor_idx, group_info in self.groups.items():
            # Get mentor info - assuming all names are unique (for now)
            mentor_row      = self.df_mentors.iloc[mentor_idx]
            mentor_name     = mentor_row['name']
            mentor_major    = mentor_row['major']

            # Get mentee info
            mentee_indices  = group_info['mentees']
            mentees_info    = []

            # mentee candidate checking
            
            for mentee_idx in mentee_indices:
                mentee_row = self.df_mentees.iloc[mentee_idx]
                mentees_info.append({
                    'name': mentee_row['name'],
                    'major': mentee_row['major'],
                    'year': mentee_row['year']
                })
            
            avg_compatibility = float(group_info['total_compatibility_score'] / len(mentee_indices))

            result = {
                'group_id': int(mentor_idx),
                'mentor':{
                    'name'  : mentor_name,
                    'major' : mentor_major,
                    'email' : mentor_row['ufl_email']
                },
                'mentees'   : mentees_info,
                'compatibility_score'   : avg_compatibility,
                'individual_scores'     : [float(s) for s in group_info['individual_scores']]
            }

            results.append(result)

        results.sort(key=lambda x: x['compatibility_score'], reverse=True)  

        self.results = results 
        return self
    
    def display_results(self):
        """
        Command line display on matching results, referencing outputs from set_output_result()
        """
        print("=" * 60)
        print("MENTOR-MENTEE GROUP RECOMMENDATIONS")
        print("=" * 60)

        for i, result in enumerate(self.results, 1):
            print(f"\nGroup {i} | Compatibility Score: {result['compatibility_score']:.2f}")
            print(f"Mentor: {result['mentor']['name']} ({result['mentor']['major']})")
            print(f"Email: {result['mentor']['email']}")

            for j, (mentee, score) in enumerate(
                zip(result['mentees'], result['individual_scores']), start=1
            ):
                print(
                    f" {j}. {mentee['name']} "
                    f"- Year: {mentee['year']} "
                    f"- Major: {mentee['major']} "
                    f"- Score: {score:.3f}"
                )

        return self


    def _save_groups_summary_csv(self, output_path: str) -> None:
        """Save high-level summary of each group."""
        
        filepath = os.path.join(output_path, "groups_summary.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Group ID',
                'Mentor Name',
                'Mentor Major',
                'Mentor Email',
                'Num Mentees',
                'Avg Compatibility Score',
                'Mentee Names'
            ])
            
            # Data rows
            for result in self.results:
                mentee_names = ', '.join([m['name'] for m in result['mentees']])
                
                writer.writerow([
                    result['group_id'],
                    result['mentor']['name'],
                    result['mentor']['major'],
                    result['mentor']['email'],
                    len(result['mentees']),
                    f"{result['compatibility_score']:.4f}",
                    mentee_names
                ])


    def _save_detailed_matches_csv(self, output_path: str) -> None:
        """Save flattened mentor-mentee pairs with individual scores."""
        import csv
        
        filepath = os.path.join(output_path, "detailed_matches.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Group ID',
                'Mentor Name',
                'Mentor Major',
                'Mentor Email',
                'Mentee Name',
                'Mentee Major',
                'Mentee Year',
                'Individual Score',
                'Group Avg Score'
            ])
            
            # Data rows - one row per mentor-mentee pair
            for result in self.results:
                mentor = result['mentor']
                group_id = result['group_id']
                group_score = result['compatibility_score']
                
                for mentee, score in zip(result['mentees'], result['individual_scores']):
                    writer.writerow([
                        group_id,
                        mentor['name'],
                        mentor['major'],
                        mentor['email'],
                        mentee['name'],
                        mentee['major'],
                        mentee['year'],
                        f"{score:.4f}",
                        f"{group_score:.4f}"
                    ])


    def _save_readable_report(self, output_path: str) -> None:
        """Save human-readable text report."""
        filepath = os.path.join(output_path, "readable_report.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("MENTOR-MENTEE GROUP MATCHING RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Groups: {len(self.results)}\n")
            
            if self.results:
                avg_score = np.mean([r['compatibility_score'] for r in self.results])
                f.write(f"Average Compatibility Score: {avg_score:.4f}\n")
            
            f.write("=" * 80 + "\n\n")
            
            # Individual group details
            for i, result in enumerate(self.results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"GROUP {i} | Compatibility Score: {result['compatibility_score']:.4f}\n")
                f.write(f"{'='*80}\n\n")
                
                # Mentor info
                f.write(f"MENTOR:\n")
                f.write(f"  Name:  {result['mentor']['name']}\n")
                f.write(f"  Major: {result['mentor']['major']}\n")
                f.write(f"  Email: {result['mentor']['email']}\n\n")
                
                # Mentees info
                f.write(f"MENTEES ({len(result['mentees'])}):\n")
                for j, (mentee, score) in enumerate(zip(result['mentees'], result['individual_scores']), 1):
                    f.write(f"  {j}. {mentee['name']}\n")
                    f.write(f"     Major: {mentee['major']}\n")
                    f.write(f"     Year:  {mentee['year']}\n")
                    f.write(f"     Compatibility Score: {score:.4f}\n")
                    if j < len(result['mentees']):
                        f.write("\n")
                
                f.write("\n")
            
            # Summary statistics at the end
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            scores = [r['compatibility_score'] for r in self.results]
            f.write(f"Total Groups: {len(self.results)}\n")
            f.write(f"Average Score: {np.mean(scores):.4f}\n")
            f.write(f"Median Score:  {np.median(scores):.4f}\n")
            f.write(f"Min Score:     {np.min(scores):.4f}\n")
            f.write(f"Max Score:     {np.max(scores):.4f}\n")
            f.write(f"Std Dev:       {np.std(scores):.4f}\n")



    def save_results(self, output_dir: str):
        """
        Save matching results to CSV

        1. groups_summary.csv : Overview of all groups
        2. detailed_matches.csv : Mentor-mentee pairs with scores
        
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No results to save. Call set_output_result() first.")
                
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"matching_results_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        print(f"\nSaving results to: {output_path}")
        
        # 1. Save groups summary CSV
        self._save_groups_summary_csv(output_path)
        
        # 2. Save detailed matches CSV (flattened)
        self._save_detailed_matches_csv(output_path)
 
        # 3. Save readable text report
        self._save_readable_report(output_path)
        
        print(f"✓ Saved 4 output files:")
        print(f"  - groups_summary.csv")
        print(f"  - results.json")
        print(f"  - readable_report.txt")
        
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
    DATA_PATH   = "./vso_ratataou_ace_mock_data.csv"
    SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TRAIN_MODEL = True
    USE_FAISS   = False 
    USE_PRETRAINED_MODEL = True
    MODEL_CHECKPOINT = "./best_model.pt"

    # Training model
    NUMB_EPOCHS   = 10
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-3


    pipeline = End2EndMatching(
        data_path               = DATA_PATH,
        sbert_pretrained_model  = SBERT_MODEL,
        use_pretrained_model    = USE_PRETRAINED_MODEL,
        model_checkpoint_path   = MODEL_CHECKPOINT
    )

    try:
        pipeline.load_csv_data()       
        pipeline.engineer_features()   
        pipeline.generate_embeddings()
        pipeline.initialize_model()

        if TRAIN_MODEL:
            pipeline, history = pipeline.train_model(
                num_epochs=20,
                batch_size=32,
                learning_rate=1e-3,
                loss_type='margin',
                use_multi_positive=True,
                k_positives=3,
                use_hard_negatives=True
            )
            pipeline.generate_mentor_embeddings()
        else:
            pipeline.generate_mentor_embeddings()

        pipeline.match_groups(use_faiss=USE_FAISS)
        pipeline.set_output_result()
        pipeline.save_results(output_dir='output')
        #pipeline.display_results()

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        traceback.print_exc()

# TODO import logging and replace all print statements - 
# TODO Test at different DiversityLoss weights - training
# TODO Test parallel loading w/ num_workers - training

        
if __name__ == "__main__":
    main()



        
        