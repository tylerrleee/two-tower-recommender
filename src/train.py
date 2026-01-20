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
    
    
def train_epoch(model, 
                dataloader,
                optimizer, 
                criterion, 
                device, 
                loss_type: str='margin'):
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
        metrics: Dictionary with additional metrics 
            avg_loss: primary metric used for backpropagation. It is the average of the total loss across all batches.

            avg_comp_loss (Compatibility): tracks how well the pairs fit together purely based on preference/similarity, ignoring the diversity penalty.

            avg_div_loss (Diversity): tracks the penalty applied for selecting cohorts that are too homogeneous (lack diversity).   
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
        # Ensure that similar to equal users don't produce an explosive or NaN gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Compute averages
    if num_batches == 0:
        raise ValueError("Validation Data Loader is empty")
    else:
        avg_loss = total_loss / num_batches
        
    metrics = {'avg_loss': avg_loss}

    if loss_type == 'diversity':
        metrics.update({
            'avg_comp_loss': total_comp_loss / num_batches,
            'avg_div_loss': total_div_loss / num_batches
        })


    assert num_batches == len(dataloader)   

    return avg_loss, metrics

def validate_epoch(model, dataloader, criterion, device, loss_type='margin'):
    """
    Validate model for one epoch.
    
    Args:
        model: The two-tower model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: torch.device
        loss_type: 'diversity' or 'margin'
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dictionary with additional metrics
    """
    # Disable Dropout 
    model.eval()

    total_loss = 0
    total_comp_loss = 0
    total_div_loss = 0
    num_batches = 0
    
    # Prevent model from updating weights
    with torch.no_grad():
        for batch in dataloader:
            mentor_feat = batch['mentor'].to(device)
            mentee_feat = batch['mentee'].to(device)
            mentee_div  = batch['mentee_div'].to(device)
            labels      = batch['pair_idx'].to(device)

            # Forward pass
            _, mentor_emb, mentee_emb = model(mentor_feat, mentee_feat)

            # Compute Loss
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

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    
    metrics = {'avg_loss': avg_loss}
    

    metrics['avg_comp_loss'] = total_comp_loss / num_batches
    metrics['avg_div_loss'] = total_div_loss / num_batches
    
    return avg_loss, metrics

def train_model_with_validation(
    model, 
    train_loader, 
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    loss_type='margin',
    early_stopping_patience=5
):
    """
    Full training loop with validation and early stopping
    
    Args:
        model: Two-tower model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
        num_epochs: Number of training epochs
        loss_type: 'diversity' or 'margin'
        early_stopping_patience: Stop if no improvement for N epochs
    
    Returns:
        history: Dictionary with training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            loss_type=loss_type
        )
        
        # Validation
        val_loss, val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            loss_type=loss_type
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        

        print(f"  Train Comp: {train_metrics['avg_comp_loss']:.4f}, "
                  f"Div: {train_metrics['avg_div_loss']:.4f}")
        print(f"  Val Comp:   {val_metrics['avg_comp_loss']:.4f}, "
                  f"Div: {val_metrics['avg_div_loss']:.4f}")
        
        # Early stopping - after X patience, the training stops as we see no improvement, and to prevent overfitting
        ## TODO default 5, but we need to test at different numbers to see where overfitting occurs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
            print(f" New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return history