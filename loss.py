import torch
import torch.nn as nn
import torch.nn.functional as F

class DiversityLoss(nn.Module):
    """
    Combined loss that encourages:
    1. Compatible mentor-mentee matching
    2. Diverse mentee pairs in the same group
    """
    def __init__(
            self,
            compatibility_weight: float = 0.7,
            diversity_weight: float = 0.3,
            temperature: float = 0.1
    ):
        super().__init__()
        self.compatibility_weight = compatibility_weight
        self.diversity_weight = diversity_weight
        self.temperature = temperature
    
    def forward(
            self, 
            mentor_emb,
            mentee_emb,
            positive_pairs,
            mentee_diversity_features = None
    ):
        """
        Args:
            :param mentor_emb: (batch_size, embedding_dim)
            :param mentee_emb: (batch_size, embedding_dim)
            :param positive_pairs: (batch_size,) - indices of positive mentor-mentee pairs
            :param mentee_diversity_features:  (batch_size, diversity_dim) - features like extroversion
        
        Returns:
            Total Loss
            Compatibility Loss
            Diversity Loss
        """
        # 1. Compatiblity loss 
        # A * B^T / temperature
        similarity_matrix = torch.matmul(mentor_emb, mentee_emb.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)

        # Cross Entropy Loss (infoNCE)
        compatability_loss = F.cross_entropy(similarity_matrix, labels)

        # 2. Diversity Loss (mentees should be diverse)
        diversity_loss = 0
        if mentee_diversity_features:
            # Compute pairwise diversity (-cosine sim)
            mentee_sim = torch.matmul(
                F.normalize(mentee_diversity_features, dim=1),
                F.normalize(mentee_diversity_features, dim=1).T
            )

            # Mask out diagonals | A mentee will be 100% similar to themselves
            mask = torch.eye(mentee_sim.size(0), device = mentee_sim.device).bool()

            # Penalize high similarity scores
            # If a mentee is more than 50% similar, than we penalize
            penalty_threshold = 0.5 # ADJUST
            diversity_loss = torch.mean(F.rely(mentee_sim[~mask] - penalty_threshold))
        
        total_loss = (
            self.compatibility_weight * compatability_loss + 
            self.diversity_weight * diversity_loss
        )

        return total_loss, compatability_loss, diversity_loss