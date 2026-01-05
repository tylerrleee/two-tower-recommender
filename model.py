import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(
            self, 
            embedding_dim: int,
            meta_feature_dim: int,
            tower_hidden_dims: list = [256, 128, 64],
            dropout_rate: float = 0.3
    ):
        super().__init__()

        input_dim = embedding_dim + meta_feature_dim
        self.mentor_tower = self._build_tower(input_dim, tower_hidden_dims, dropout_rate)
        self.mentee_tower = self._build_tower(input_dim, tower_hidden_dims, dropout_rate)
        self.output_dim = tower_hidden_dims[-1]
        self.bias = nn.Parameter(torch.zeros(1))
    
    def _build_tower(self, input_dim, hidden_dims, dropout_rate):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, mentor_features, mentee_features):

        # Get Embeddings
        mentor_emb = self.mentor_tower(mentor_features)
        mentee_emb = self.mentee_tower(mentee_features)

        # L2 Normalize
        mentor_emb = F.normalize(mentor_emb, p=2, dim=1)
        mentee_emb = F.normalize(mentee_emb, p=2, dim=1)

        # Compute similarity (Dot prod)
        similarity = torch.sum(mentor_emb * mentee_emb, dim=1, keepdim=True)

        return similarity + self.bias, mentor_emb, mentee_emb
    
    def get_mentor_embedding(self, mentor_features):
        emb = self.mentor_tower(mentor_features)
        return F.normalize(emb, p=2, dim=1)
    
    def get_mentee_embedding(self, mentee_features: torch.FloatTensor):
        emb = self.mentee_tower(mentee_features)
        return F.normalize(emb, p=2, dim=1)