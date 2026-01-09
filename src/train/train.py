import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MentorMenteeDataset(Dataset):
    def __init__(
            self,
            mentor_features,
            mentee_features,
            mentor_diversity,
            mentee_diversity,
            positive_pairs
        ):
        self.mentor_features  = torch.FloatTensor(mentor_features)
        self.mentee_features  = torch.FloatTensor(mentee_features)
        self.mentor_diversity = torch.FloatTensor(mentor_diversity)
        self.mentee_diversity = torch.FloatTensor(mentee_diversity)
        self.positive_pairs   = torch.LongTensor(positive_pairs)

    def __len__(self):
        return len(self.mentor_features)
    
    def __getitem__(self, index):
        return {
            'mentor': self.mentor_features[index],
            'mentee': self.mentee_features[index],
            'mentor_div': self.mentor_diversity[index],
            'mentee_div': self.mentee_diversity[index],
            'pair_idx': self.positive_pairs[index]
        }
    
def train_epoch(model, dataloader, optimizer, criterion, device, loss_type: str='margin'):
    """
    Train for one epoch with flexible loss function support.
    
    Args:
        model: The two-tower model
        dataloader: PyTorch DataLoader
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (DiversityLoss or PairwiseMarginLoss)
        device: torch.device
        loss_type: 'diversity' or 'margin' - determines how to handle loss returns
    
    Returns:
        avg_loss: Average loss for the epoch
        metrics: Dictionary with additional metrics (comp_loss, div_loss if applicable)
    """
    
    model.train()
    total_loss = 0
    total_comp_loss = 0
    total_div_loss = 0
    
    num_batches = 0

    for batch in dataloader:
        mentor_feat = batch['mentor'].to(device)
        mentee_feat = batch['mentee'].to(device)
        mentee_div  = batch['mentee_div'].to(device)
        labels      = batch['pair_idx'].to(device) # Batch-specific positive pairs

        optimizer.zero_grad()

        # Forward pass
        similarity, mentor_emb, mentee_emb = model(mentor_feat, mentee_feat)

        # Compute Loss
       # Compute Loss based on loss type
        if loss_type == 'diversity':
            # DiversityLoss returns tuple: (total_loss, comp_loss, div_loss)
            loss, comp_loss, div_loss = criterion(
                mentor_emb=mentor_emb,
                mentee_emb=mentee_emb,
                positive_pairs=labels,
                mentee_diversity_features=mentee_div
            )
            total_comp_loss += comp_loss.item()
            total_div_loss += div_loss.item()
            
        elif loss_type == 'margin':
            # PairwiseMarginLoss returns single tensor
            loss = criterion(
                mentor_emb=mentor_emb,
                mentee_emb=mentee_emb,
                positive_pairs=labels  
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Backward Pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters()) # Ensure that similar to equal users don't produce an explosive or NaN gradient
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Compute averages
        avg_loss = total_loss / num_batches
        
        metrics = {
            'avg_loss'      : avg_loss,
            'avg_comp_loss' : total_comp_loss / num_batches,
            'avg_div_loss'  : total_div_loss / num_batches
        }
        
        return avg_loss, metrics
