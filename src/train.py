"""
Training utilities for Two-Tower Matching Model

"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Any, Optional, Union, List
import os

class MentorMenteeDataset(Dataset):
    """
    PyTorch Dataset for mentor-mentee matching with multi-positive support
    
    Supports two modes:
    1. Single positive: pos_pairs shape (N,)
    2. Multi-positive: pos_pairs shape (N, k)
    """
    
    def __init__(
                self,
                mentor_features     : npt.NDArray[np.float32],
                mentee_features     : npt.NDArray[np.float32],
                mentor_diversity    : npt.NDArray[np.float32],
                mentee_diversity    : npt.NDArray[np.float32],
                positive_pairs      : npt.NDArray[np.int64],
                hard_negative_pool  : Optional[npt.NDArray[np.float32]] = None,
                hard_negative_diversity: Optional[npt.NDArray[np.float32]] = None
            ) -> None:
        """
        Args:
            mentor_features: (N, D) mentor embeddings
            mentee_features: (N, D) OR (N*k, D) matched mentee embeddings
            mentor_diversity: (N, diversity_dim) mentor diversity features
            mentee_diversity: (N, diversity_dim) OR (N*k, diversity_dim) mentee diversity
            positive_pairs: (N,) single positive OR (N, k) multi-positive indices
            hard_negative_pool: (M, D) embeddings of unused mentees for hard negatives
            hard_negative_diversity: (M, diversity_dim) diversity of unused mentees
        """
        self.multi_positive : bool = (positive_pairs.ndim == 2) # True when == (N*k, diversity_dim)

        if self.multi_positive:
            self.top_k = positive_pairs.shape[1]
            print(f"  Dataset mode: Multi-positive (k={self.top_k})")
        else:
            self.top_k = 1
            print(f"  Dataset mode: Single positive")
        
        # Validation
        assert len(mentor_features) == len(mentee_features), \
            f"Mentor-mentee count mismatch: {len(mentor_features)} vs {len(mentee_features)}"
        
        if self.multi_positive:
            assert positive_pairs.max() < len(mentee_features), \
                f"Positive pair index {positive_pairs.max()} exceeds mentee count {len(mentee_features)}"
        else:
            # For single positive, mentee_features should be N (not N*k)
            assert len(positive_pairs) == len(mentor_features)

        self.mentor_features    = torch.FloatTensor(mentor_features)
        self.mentee_features    = torch.FloatTensor(mentee_features)
        self.mentor_diversity   = torch.FloatTensor(mentor_diversity)
        self.mentee_diversity   = torch.FloatTensor(mentee_diversity)
        self.positive_pairs     = torch.LongTensor(positive_pairs)
        
        # Hard negative pool
        self.has_hard_negatives : bool = (hard_negative_pool is not None)

        if self.has_hard_negatives:
            self.hard_negative_pool = torch.FloatTensor(hard_negative_pool)
            self.hard_negative_diversity = torch.FloatTensor(hard_negative_diversity)
            print(f"  Hard negative pool size: {len(self.hard_negative_pool)}")
        else:
            self.hard_negative_pool = None
            self.hard_negative_diversity = None

    def __len__(self) -> int:
        """
        Returns the number of mentor in the dataset. 
        The built-in len() function will call this method.
        """
        return len(self.mentor_features)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample.
        
        For multi-positive mode:
        - Randomly samples ONE positive from the k candidates
        - This provides gradient diversity across epochs
        """
        mentor = self.mentor_features[index]
        mentor_div = self.mentor_diversity[index]
        
        if self.multi_positive:
            # Randomly sample one positive from k candidates
            pos_candidates = self.positive_pairs[index]  # (k,)
            selected_idx = torch.randint(0, self.top_k, (1,)).item()
            pair_idx = pos_candidates[selected_idx]
            
            mentee = self.mentee_features[pair_idx]
            mentee_div = self.mentee_diversity[pair_idx]
        else:
            # Single positive mode
            pair_idx = self.positive_pairs[index]
            mentee = self.mentee_features[index]  # Already aligned
            mentee_div = self.mentee_diversity[index]
        
        sample = {
            'mentor': mentor,
            'mentee': mentee,
            'mentor_div': mentor_div,
            'mentee_div': mentee_div,
            'pair_idx': pair_idx if self.multi_positive else torch.tensor(index)
        }
        
        # Optionally include a hard negative
        if self.has_hard_negatives:
            hard_neg_idx = torch.randint(0, len(self.hard_negative_pool), (1,)).item()
            sample['hard_negative'] = self.hard_negative_pool[hard_neg_idx]
            sample['hard_negative_div'] = self.hard_negative_diversity[hard_neg_idx]
        
        return sample
    
def create_mentor_level_split(
                            n_mentors: int,
                            train_ratio: float = 0.8,
                            random_seed: int = 42
                        ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Split mentors into train/val sets (NOT pairs).
    
    This ensures validation tests generalization to NEW mentors,
    not just memorization of training mentors.
    
    Args:
        n_mentors: Total number of mentors
        train_ratio: Fraction for training (default: 0.8)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        train_mentor_indices: Indices of mentors for training
        val_mentor_indices: Indices of mentors for validation
    """
    np.random.seed(random_seed)

    mentor_indices = np.arange(n_mentors) # [1,2,3,4,5,...] Sequential array
    np.random.shuffle(mentor_indices)     # [5,2,3,1,4,...] Shuffle array in-place

    split_point = int(n_mentors * train_ratio) # 0.8 means we will split 80% of the dataset for train/eval

    train_mentors = mentor_indices[:split_point]

    val_mentors = mentor_indices[split_point:]

    print(f"  Mentor-level split:")
    print(f"    Train mentors: {len(train_mentors)}")
    print(f"    Val mentors: {len(val_mentors)}")

    return train_mentors, val_mentors


def train_epoch(
                model       : torch.nn.Module,
                dataloader  : DataLoader,
                optimizer   : torch.optim.Optimizer,
                criterion   : torch.nn.Module,
                device      : torch.device,
                loss_type   : str = 'margin',
                max_grad_norm: float = 1.0,
                use_hard_negatives: bool = False
            ) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch with in-batch negatives."""
    
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        mentor_feat = batch['mentor'].to(device)
        mentee_feat = batch['mentee'].to(device)  # Already aligned to batch
        
        # The positive is implicit (mentor i matches mentee i)
        
        optimizer.zero_grad()
        
        # Forward pass
        _ , mentor_emb, mentee_emb = model(mentor_feat, mentee_feat)
        
        # Compute loss 
        if loss_type == 'margin':
            hard_neg = None
            if use_hard_negatives and 'hard_negative' in batch:
                hard_neg_feat = batch['hard_negative'].to(device)
                _, _, hard_neg_emb = model(mentor_feat, hard_neg_feat)
                hard_neg = hard_neg_emb
            
            loss = criterion(
                mentor_emb=mentor_emb,
                mentee_emb=mentee_emb,
                hard_negatives=hard_neg
            )
        
        elif loss_type == 'diversity':
            mentee_div = batch['mentee_div'].to(device)
            labels = batch['pair_idx'].to(device)
            loss, comp_loss, div_loss = criterion(
                mentor_emb=mentor_emb,
                mentee_emb=mentee_emb,
                positive_pairs=labels,
                mentee_diversity_features=mentee_div
            )
        
        # Backward pass
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at batch {batch_idx}: {loss.item()}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    metrics = {'avg_loss': avg_loss}
    
    return avg_loss, metrics


def validate_epoch(
                    model: torch.nn.Module,
                    dataloader: DataLoader,
                    criterion: torch.nn.Module,
                    device: torch.device,
                    loss_type: str = 'margin'
                ) -> Tuple[float, Dict[str, float]]:
    
    """Validate model for one epoch."""

    if loss_type not in ['margin', 'diversity']:
        raise ValueError(f"loss_type must be 'margin' or 'diversity', got '{loss_type}'")
    
    if len(dataloader) == 0:
        raise ValueError("Validation dataloader is empty")
    
    model.eval()

    total_loss      : float = 0.0
    total_comp_loss : float = 0.0
    total_div_loss  : float = 0.0
    num_batches     : int   = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mentor_feat = batch['mentor'].to(device)
            mentee_feat = batch['mentee'].to(device)
            mentee_div = batch['mentee_div'].to(device)
            labels = batch['pair_idx'].to(device)

            _, mentor_emb, mentee_emb = model(mentor_feat, mentee_feat)
            
            assert torch.isfinite(mentor_emb).all()
            assert torch.isfinite(mentee_emb).all()

            if loss_type == 'diversity':
                loss, comp_loss, div_loss = criterion(
                    mentor_emb=mentor_emb,
                    mentee_emb=mentee_emb,
                    positive_pairs=labels,
                    mentee_diversity_features=mentee_div
                )
                total_comp_loss += comp_loss.item()
                total_div_loss += div_loss.item()
                
            elif loss_type == 'margin':
                loss = criterion(
                    mentor_emb=mentor_emb,
                    mentee_emb=mentee_emb,
                    positive_pairs=labels
                )
            
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss at batch {batch_idx}")

            total_loss += loss.item()
            num_batches += 1
    
    assert not model.training

    avg_loss = total_loss / num_batches
    metrics: Dict[str, float] = {'avg_loss': avg_loss}
    
    if loss_type == 'diversity':
        metrics['avg_comp_loss'] = total_comp_loss / num_batches
        metrics['avg_div_loss'] = total_div_loss / num_batches
    
    return avg_loss, metrics

def train_model_with_validation(
                                model           : torch.nn.Module,
                                train_loader    : DataLoader,
                                val_loader      : DataLoader,
                                criterion       : torch.nn.Module,
                                optimizer       : torch.optim.Optimizer,
                                device          : torch.device,
                                num_epochs      : int = 10,
                                loss_type       : str = 'margin',
                                early_stopping_patience: int = 5,
                                max_grad_norm   : float = 1.0,
                                checkpoint_path : str = 'best_model.pt',
                                use_hard_negatives: bool = False
                            ) -> Dict[str, Any]:
    """
    Full training loop with validation and early stopping.
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty")
    if len(val_loader) == 0:
        raise ValueError("Validation dataloader is empty")
    if num_epochs < early_stopping_patience:
        raise ValueError("num_epochs must be greater than stopping patience.")

    history = {
        'train_loss'    : [],
        'val_loss'      : [],
        'train_metrics' : [],
        'val_metrics'   : [],
        'best_epoch'    : 0,
        'stopped_early' : False
    }
    
    best_val_loss = np.inf
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {num_epochs} epochs, {loss_type} loss")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Hard negatives: {use_hard_negatives}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            loss_type=loss_type,
            max_grad_norm=max_grad_norm,
            use_hard_negatives=use_hard_negatives
        )
        
        val_loss, val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            loss_type=loss_type
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        if loss_type == 'diversity':
            print(f"  Train Comp: {train_metrics['avg_comp_loss']:.4f}, "
                  f"Div: {train_metrics['avg_div_loss']:.4f}")
            print(f"  Val Comp:   {val_metrics['avg_comp_loss']:.4f}, "
                  f"Div: {val_metrics['avg_div_loss']:.4f}")
        
        if train_loss < val_loss * 0.5:
            print(f"Warning: Possible overfitting")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'loss_type': loss_type
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved!")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n Early stopping triggered")
                history['stopped_early'] = True
                break
        
        print()
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best epoch: {history['best_epoch']+1}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return history
