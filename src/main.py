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
"""

import numpy as np
import pandas as pd
import torch

from src.features.features      import FeatureEngineer
from src.embedding.embedding    import EmbeddingEngineer
from src.embedding.boostrap_positive_pairs    import bootstrap_positive_pairs_from_embeddings

from src.model.model            import TwoTowerModel
from src.train.train            import MentorMenteeDataset, train_epoch, train_model_with_validation
from src.matcher.matcher        import GroupMatcher
from src.loss.loss              import DiversityLoss
from src.loss.pairwise_margin_loss import PairwiseMarginLoss

import config
import traceback

class End2EndMatching:
    def __init__(
            self,
            data_path: str,
            sbert_pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            use_pretrained_model: bool  = False,
            embedding_dimensions:   int = 384,
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
        self.feature_engineer: FeatureEngineer      = None
        self.embedding_engineer: EmbeddingEngineer  = None
        self.model: TwoTowerModel                   = None
        self.matcher: GroupMatcher                  = None

        # STORAGE
        self.df: pd.DataFrame   = None
        self.mentor_data        = None
        self.mentee_data        = None
        self.mentor_embedding   = None
        self.mentee_embedding   = None
        self.mentee_embeddings_learned = None
        self.mentor_embeddings_learned = None

        # MATCHING
        self.groups             = None
        self.results            = None

        #TODO cmd print when initialized
    
    def load_csv_data(self):
        """
        Load CSV data and split into mentors and mentees
        Assuming there should be 2 mentees for every 1 mentor
        """
        self.df = pd.read_csv(self.data_path)
        self.df = FeatureEngineer.rename_column(self.df, config.RENAME_MAP)

        print(f"Loaded {len(self.df)} total applicants")

        # TODO change logic for classifying bigs and little
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
            self.model.load_state_dict(torch.load(self.model_checkpoint_path))
        # TODO What to do when we don't use a pretrained model?

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        

        return self

    def train_model(self, 
                    num_epochs: int, 
                    batch_size: int, 
                    learning_rate: float, 
                    loss_type: str = 'margin'):
        '''
        Training the model with loss function 
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            loss_type: 'margin' or 'diversity' - which loss function to use
        '''
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        print(f"Loss type: {loss_type}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get synthetic positive pairs
        n_mentors = len(self.mentor_embedding)
        n_mentees = len(self.mentee_embedding)
        
        n_pairs = min(n_mentees, n_mentors)

        # Synthetic Pairng [1, 2, 3,...n]
        #pos_pairs = np.arange(n_pairs)
        pos_pairs = bootstrap_positive_pairs_from_embeddings(
            mentor_embeddings=self.mentor_embedding,
            mentee_embeddings=self.mentee_embedding,
            top_k=5,
            method='hungarian'
        )
        # Compute diversity features 
        mentee_diversity = self.feature_engineer.compute_diversity_features(
            df=self.df_mentees.head(n_pairs)
        )
        mentor_diversity = np.zeros((n_pairs, mentee_diversity.shape[1]))

        # Create dataset
        dataset = MentorMenteeDataset(
            mentee_features=self.mentee_embedding[:n_pairs],
            mentor_features=self.mentor_embedding[:n_pairs],
            mentee_diversity=mentee_diversity, 
            mentor_diversity=mentor_diversity,
            positive_pairs=pos_pairs
        )

        # Split into train/val (80/20)

        train_size = int(0.80 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Load Data in randomized batches 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Intialize Adam Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Choose loss function
        if loss_type == 'margin':
            criterion = PairwiseMarginLoss(margin=0.2, 
                                           similarity="cosine")
        elif loss_type == 'diversity':
            criterion = DiversityLoss(
                compatibility_weight=0.7,
                diversity_weight=0.3,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        print("Starting training...")
        self.model.train()

        # Use the improved training loop
        history = train_model_with_validation(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            loss_type=loss_type,
            early_stopping_patience=5
        )

        print("Training completed!")

        save_path = f"model_checkpoint_{loss_type}_epoch{num_epochs}.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

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



    def save_results(self, output_dir: str):
        """
        Save matching results to a CSV file
        """
        ...
            


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

    # Training model
    NUMB_EPOCHS   = 10
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-3


    pipeline = End2EndMatching(
        data_path               =DATA_PATH,
        sbert_pretrained_model  =SBERT_MODEL,
        use_pretrained_model    = False
    )

    try:
        pipeline.load_csv_data()       
        pipeline.engineer_features()   
        pipeline.generate_embeddings()
        pipeline.initialize_model()

        if TRAIN_MODEL:
            pipeline.train_model(
                num_epochs      =NUMB_EPOCHS,
                batch_size      =BATCH_SIZE,
                learning_rate   =LEARNING_RATE
            )
            pipeline.generate_mentor_embeddings()

        pipeline.match_groups(use_faiss=USE_FAISS)
        pipeline.set_output_result()
        pipeline.save_results(output_dir='output')
        pipeline.display_results()

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



        
        