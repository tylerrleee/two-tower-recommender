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
    
    def train_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0

        for batch in dataloader:
            mentor_feat = batch['mentor'].to(device)
            mentee_feat = batch['mentee'].to(device)
            mentee_div  = batch['mentee_div'].to(device)

            optimizer.zero_grad()

            # Forward pass
            similarity, mentor_emb, mentee_emb = model(mentor_feat, mentee_feat)

            # Compute Loss
            labels = batch['pair_idx'].to(device)
            loss, comp_loss, div_loss = criterion(
                mentor_emb,
                mentee_emb,
                labels,
                mentee_div
            )

            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
