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
                 similarity: str = 'cosine',
                 temperature: float = 0.1
                ):
        super().__init__()
        self.margin = margin
        self.similarity = similarity
        self.temperature = temperature
    
    def compute_similarity(self, a, b):
        """Compute similarity matrix between two embedding sets"""
        if self.similarity == "cosine":
            a = F.normalize(a, dim=1)
            b = F.normalize(b, dim=1)
        
        # Dot product similarity
        return torch.matmul(a, b.T) / self.temperature
    
    def forward(self, 
                mentor_emb: torch.Tensor,      # (B, D)
                mentee_emb: torch.Tensor,      # (B, D)
                positive_pairs: torch.Tensor = None,  # Not used in this version
                hard_negatives: torch.Tensor = None   # Optional (B, D)
                ):
        """
        Compute pairwise margin loss with in-batch negatives
        
        Args:
            mentor_emb: Mentor embeddings (B, D)
            mentee_emb: Mentee embeddings (B, D) - positive mentees
            positive_pairs: Ignored (kept for API compatibility)
            hard_negatives: Optional hard negative embeddings (B, D)
        
        The positive pair for mentor i is mentee i (diagonal of similarity matrix).
        All other mentees in the batch serve as negatives.
        """
        B = mentor_emb.size(0)
        
        # Similarity matrix: mentor_i × mentee_j
        sim = self.compute_similarity(mentor_emb, mentee_emb)  # (B, B)
        
        # Positive similarities (diagonal)
        pos_sim = torch.diag(sim)  # (B,)
        pos_sim = pos_sim.unsqueeze(1)  # (B, 1) for broadcasting
        
        # Compute margin loss: max(0, margin - pos_sim + neg_sim)
        # For each mentor, penalize when: neg_sim > pos_sim - margin
        loss_matrix = F.relu(self.margin - pos_sim + sim)  # (B, B)
        
        # Mask out diagonal (positive pairs don't contribute to loss)
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        in_batch_loss = loss_matrix[mask].mean()
        
        total_loss = in_batch_loss
        
        if hard_negatives is not None:
            hard_neg_sim = self.compute_similarity(mentor_emb, hard_negatives)  # (B, B)
            hard_neg_loss_matrix = F.relu(self.margin - pos_sim + hard_neg_sim)
            hard_neg_loss = hard_neg_loss_matrix.mean()
            total_loss = total_loss + 0.5 * hard_neg_loss  # Weight hard negatives
        
        # Metrics for monitoring
        with torch.no_grad():
            avg_pos_sim = pos_sim.mean().item()
            avg_neg_sim = sim[mask].mean().item()
            margin_violations = (loss_matrix[mask] > 0).float().mean().item()
            
            # Only print occasionally to avoid spam
            if torch.rand(1).item() < 0.1:  # 10% of batches
                print(f"  [Loss] Pos: {avg_pos_sim:.3f}, Neg: {avg_neg_sim:.3f}, "
                      f"Violations: {margin_violations:.2%}")
        
        return total_loss
    