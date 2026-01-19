import torch
import torch.nn as nn
import torch.nn.functional as F 

class PairwiseMarginLoss(nn.Module):
    """
    Mentor anchored pairwise margin ranking loss
    Note:
    - https://gombru.github.io/2019/04/03/ranking_loss/
    """
    def __init__(self,
                 margin: float = 0.2,
                 similarity: str = 'cosine'
                ):
        super().__init__()
        self.margin     = margin
        self.similarity = similarity
    
    def compute_similarity(self, a, b):
        if self.similarity == "cosine":
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
        
        # Dot Product
        return torch.matmul(a, b.T)
    
    def forward(self, 
                mentor_emb: torch.Tensor, # (B, D)
                mentee_emb: torch.Tensor, # (B, D)
                positive_pairs: torch.Tensor
                ):
        """
        positive_pairs[i] = index of positive mentee for mentor ith
        """

        B = mentor_emb.size(dim=0)

        # Similarity matrix (mentor x mentee)
        sim = self.compute_similarity(mentor_emb, mentee_emb)

        # Positive Similarities
        pos_sim = sim[torch.arange(B), positive_pairs] # (B, )

        # Expand for broadcasting
        pos_sim = pos_sim.unsqueeze(dim=1) # (B, 1)

        # Compute triplet margin loss formulation
        ##  max(0, margin - sim(anchor, positive) + sim(anchor, negative))

        loss_matrix = F.relu(self.margin - pos_sim + sim)

        # Mask out positive pairs (mentee ith vs mentee ith)
        mask = torch.ones_like(loss_matrix, dtype=torch.bool)
        mask[torch.arange(B), positive_pairs] = False

        loss = loss_matrix[mask].mean()

        print("Avg positive sim:", pos_sim.mean().item())
        print("Avg negative sim:", sim.mean().item())
        print("Margin violations:", (loss_matrix > 0).float().mean().item())

        return loss

    